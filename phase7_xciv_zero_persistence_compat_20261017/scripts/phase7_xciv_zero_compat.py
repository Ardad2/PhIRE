#!/usr/bin/env python3
"""Additive XCIV compatibility launcher: reconcile canonical zero-persistence pairs.

The 2026-10-17 Spark diagnostic established, for GT sample 0:
 H0 canonical 1452, Phase5E retained 1434, 18 exact zeros dropped;
 H1 canonical 1466, Phase5E retained 1453, 13 exact zeros dropped.
Retained birth/death pairs matched exactly. This launcher checks the same
identity for EVERY XCII source and dimension without changing Phase5E solver.

Pinned original XCIV driver is loaded from disk, not edited. The only runtime
interposition is (a) read-only capture of the archived reader's zero counts,
(b) strict filtering of canonical exact/near-zero diagonal pairs before the
original *coordinate+value* comparison, and (c) adding compatibility lineage
to the atomic result manifest. XCI protocol, XCII/III and solver unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

ORIGINAL = Path('phase7_xciv_update/scripts/phase7_xciv_dev_locality.py')
ORIGINAL_SHA = '71168242dbac580936250122385315c153999e4d5ef27e83b4a7ad6c93da63ad'
SELF = Path('phase7_xciv_zero_persistence_compat_20261017/scripts/phase7_xciv_zero_compat.py')
POLICY = 'retain strictly positive persistence > original Phase5E solver.TOL; require all excluded canonical pairs match reader zero count'
COMPAT_KEY = 'coordinate_compatibility'


def need(condition, msg):
    if not condition:
        raise RuntimeError(msg)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def import_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def filtered_pairs(reference, tol, np):
    arr = np.asarray(reference, dtype=float).reshape(-1, 2)
    need(bool(np.isfinite(arr).all()), 'Non-finite canonical birth/death value')
    persistence = arr[:, 1] - arr[:, 0]
    need(bool(np.all(persistence >= -tol)), 'Negative persistence in corrected canonical input')
    keep = persistence > tol
    excluded = int(np.count_nonzero(~keep))
    return arr[keep], excluded


def install(root, original):
    """Preserve solver and original check; filter canonical diagonal only."""
    import numpy as np
    recorded_zero = {}
    filtered_counts = {}
    original_source = original.source
    original_pairs = original.pairs
    original_verify = original.verify
    checksum_self = sha256(root / SELF)
    orig_sha = sha256(root / ORIGINAL)
    need(orig_sha == ORIGINAL_SHA, 'Original XCIV SHA mismatch')

    def source_with_zero_inventory(r):
        result = original_source(r)
        solver = result[3]
        read_unmodified = solver.read_features
        def read_and_record(path):
            features, skipped = read_unmodified(path)
            name = str(Path(path).resolve())
            value = {str(d): int(skipped[d]) for d in (0, 1)}
            need(all(v >= 0 for v in value.values()), f'Negative skip count: {path}')
            recorded_zero[name] = value
            return features, skipped
        solver.read_features = read_and_record
        return result

    def checked_pairs(features, expected, path):
        label = str(Path(path).resolve())
        need(label in recorded_zero, f'Missing original zero-skip accounting for {label}')
        # Use *original* solver tolerance, not a new or loosened threshold.
        solver = original_solver[0]
        need(solver is not None and math.isfinite(float(solver.TOL)) and 0 < solver.TOL <= 1e-8,
             'Unexpected archived locality solver TOL')
        filtered = {}
        removed = {}
        for dim in (0, 1):
            kept, n_removed = filtered_pairs(expected[dim], float(solver.TOL), np)
            need(int(recorded_zero[label][str(dim)]) == n_removed,
                 f'Zero-pair discrepancy {label} H{dim}: canonical_removed={n_removed}, '
                 f'original_reader_skipped={recorded_zero[label][str(dim)]}')
            filtered[dim] = kept
            removed[str(dim)] = n_removed
        # This invokes the ORIGINAL XCIV validations for persistence, spatial
        # endpoint finiteness, bounds, and sorted birth/death matching.
        matched = original_pairs(features, filtered, path)
        filtered_counts[label] = removed
        # Stable JSON string keys also fix a latent XCIV verify-vs-manifest
        # int-key deserialization mismatch. Count values are not modified.
        return {str(d): int(matched[d]) for d in (0, 1)}

    original_solver = [None]
    def source_with_instrumentation(r):
        result = source_with_zero_inventory(r)
        original_solver[0] = result[3]
        return result

    def compat_record():
        return {
            'schema': 'phase7-xciv-zero-persistence-compat-v1',
            'policy': POLICY,
            'original_driver_sha256': orig_sha,
            'launcher_sha256': checksum_self,
            'solver_tolerance': float(original_solver[0].TOL),
            'excluded_canonical_pairs_by_file': {
                # repo-relative path, stable across Spark mounts
                str(Path(k).relative_to(root.resolve())): filtered_counts[k]
                for k in sorted(filtered_counts)
            },
        }

    def save_with_provenance(r, aggs, dims, counts, fields, src):
        """Original atomic layout, plus provenance fields BEFORE publication."""
        need(len(filtered_counts) == 28, f'Expected 28 individually matched PD sources, got {len(filtered_counts)}')
        target = r / original.OUT
        original.need(not target.exists(), f'XCIV output already exists; never overwrite: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix='.' + target.name + '.staging-', dir=target.parent))
        try:
            original.write_csv(stage / 'locality_aggregate.csv', original.A_FIELDS, aggs)
            original.write_csv(stage / 'locality_dimension.csv', original.D_FIELDS, dims)
            (stage / 'locality_report.md').write_text(original.report(aggs))
            manifest = {
                'format': 'phase7-xciv-locality-pilot-v1',
                'scope': 'DEVELOPMENT_ONLY',
                'n_fields': original.N,
                'arms': list(original.ARMS),
                'radii': [original.rk(x) for x in original.RADII],
                'source_xciii_sha256': original.sha(r / original.XCIII / 'numeric_manifest.json'),
                'xcII_source_sha256': original.sha(r / original.XCII / 'extraction_manifest.json'),
                'xciii_verified_source': src,
                'fields': fields,
                'coord_feature_counts': counts,
                'phase5e_solver_sha256': original.SOLVER_SHA,
                'this_script_sha256': original.sha(r / original.SELF),
                'n_aggregate_rows': len(aggs),
                'n_dimension_rows': len(dims),
                'output_semantics': 'Grid-cell endpoint constrained sparse assignment; not kilometer physical distance',
                COMPAT_KEY: compat_record(),
            }
            (stage / 'locality_manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + '\n')
            items = ['locality_aggregate.csv','locality_dimension.csv','locality_report.md','locality_manifest.json']
            (stage / 'SHA256SUMS').write_text(''.join(f'{original.sha(stage / x)}  {x}\n' for x in items))
            original.need(not target.exists(), 'XCIV output appeared while staging')
            stage.rename(target)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        print(f'[LOCALITY-GATE] PASS: {len(aggs)} aggregate + {len(dims)} dimension rows; '
              f'24 infinite-radius matches XCIII; compatibility provenance sealed; published={target}', flush=True)

    def verify_with_provenance(r, xc, solver, fields, scores, src, pilot, decoder):
        target = r / original.OUT
        need(target.is_dir(), f'XCIV output missing: {target}')
        # First run all unchanged XCIV source checks and 192/384 row recomputation.
        original_verify(r, xc, solver, fields, scores, src, pilot, decoder)
        manifest = json.loads((target / 'locality_manifest.json').read_text())
        observed = manifest.get(COMPAT_KEY)
        need(observed is not None, 'Missing compatibility lineage in immutable XCIV manifest')
        need(len(filtered_counts) == 28, f'Not all 28 sources checked: {len(filtered_counts)}')
        need(observed == compat_record(), 'Zero-persistence compatibility provenance drift')
        print('[COMPAT-VERIFY] PASS: original coordinate reader skip counts match canonical diagonal filtering '
              'on 28 diagrams; 56 H0/H1 numerical pair sets and immutable lineage verified', flush=True)

    original.source = source_with_instrumentation
    original.pairs = checked_pairs
    original.save = save_with_provenance
    original.verify = verify_with_provenance
    return original


def selftest():
    import numpy as np
    x = np.array([[1,1],[1,2],[3,3 + 1e-12],[0,4]], dtype=float)
    kept, removed = filtered_pairs(x,1e-10,np)
    need(removed==2 and kept.shape==(2,2) and (kept[0]==(1,2)).all() and (kept[1]==(0,4)).all(),
         'Canonical zero filter does not match threshold contract')
    for invalid in (np.array([[3,1]]), np.array([[0,np.nan]])):
        try:filtered_pairs(invalid,1e-10,np)
        except RuntimeError:pass
        else:raise RuntimeError('Invalid persistence failed to trigger fail-closed gate')
    print('[COMPAT-SELFTEST] PASS: exact/near-zero filtering, positive pair retention, '
          'negative/nonfinite rejection', flush=True)


def main():
    args = argparse.ArgumentParser(add_help=False)
    args.add_argument('--mode', choices=['selftest','preflight','compute','verify'], required=True)
    args.add_argument('--repo-root', type=Path, default=Path('.'))
    z, _ = args.parse_known_args()
    root = z.repo_root.resolve()
    need((root / ORIGINAL).is_file() and (root / SELF).is_file(), 'Missing original XCIV or compatibility launcher')
    original_digest = sha256(root / ORIGINAL)
    need(original_digest == ORIGINAL_SHA, f'Pinned XCIV script modified: {original_digest}')
    print(f'[COMPAT] Original XCIV SHA256={original_digest} launcher_sha256={sha256(root / SELF)}', flush=True)
    if z.mode=='selftest':
        selftest()
        return
    # Keep original main() argument parsing, mode handling, all source/historical
    # regressions, frozen scope and all solver code unchanged.
    original = import_source(root / ORIGINAL, 'phase7_xciv_original_pinned')
    original = install(root, original)
    original.main()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'[XCIV-COMPAT] ERROR {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
