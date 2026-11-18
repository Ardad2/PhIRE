#!/usr/bin/env python3
"""Additive, fail-closed Phase8 168-field independent-locality verification/report hotfix.

Fixes TWO presentation/verification defects in memory ONLY:
  1. Original verify_locality compares JSON-loaded string dictionary keys with
     live integer keys inside `reader_skipped_zeros`. Canonicalize the live
     result with the same JSON round-trip used by the original `put()` writer
     before the original exhaustive near_struct() numeric comparison.
  2. Original report dictionary comprehension shadows its `metric()` function
     with a string named `metric`. Use `key` as the loop variable instead.

Original script SHA256 is pinned; original on-disk source, 168 scores, 168
locality samples, replay, model weights, and locked geographic tests remain
untouched. The patched module preserves the ORIGINAL __file__ path so all
original source-lineage/manifest checks remain strict and valid.

From ~/PhIRE:
  python3 phase8_original168_verify_report_hotfix_20261118.py --mode selftest
  PYTHONNOUSERSITE=1 "$PHASE8_MICROMAMBA" run -n gudhi-audit python \
      phase8_original168_verify_report_hotfix_20261118.py --mode finalize

If independent locality re-evaluation disagrees in values, finalization stops
without creating a report. Does NOT modify previously sealed artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

ORIGINAL_NAME = 'phase8_original168_dynamic_pd_benchmark_20261118.py'
ORIGINAL_SHA256 = 'c28b2ed444083668472f5527d53cdd4a7cd04e3c2611868062668091f64192d0'

# Exact one-occurrence textual bindings; refuse other revisions and layouts.
PATCHES = (
    (
        'need(near_struct(previous,actual),',
        'need(near_struct(previous,json.loads(json.dumps(actual,allow_nan=False))),',
        'Normalize in-memory integer keys to JSON string keys in independent comparison',
    ),
    (
        'summary={arm:{metric:metric(arm,metric) for metric in metrics} for arm in ARMS}',
        'summary={arm:{key:metric(arm,key) for key in metrics} for arm in ARMS}',
        'Avoid shadowing the mean-metric function when building the report',
    ),
)


def require(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bind_original(root: Path):
    original = root / ORIGINAL_NAME
    require(original.is_file(), f'Missing original driver: {original}')
    raw = original.read_bytes()
    require(digest(raw) == ORIGINAL_SHA256,
            'STOP: original benchmark script SHA256 differs; do not change or bypass the pin. '
            f'Actual={digest(raw)} Expected={ORIGINAL_SHA256}')
    source = raw.decode('utf-8')
    for old, new, description in PATCHES:
        require(source.count(old) == 1,
                f'STOP: expected exactly one source site for: {description}')
        require(old != new, 'No-op patch invalid')
        source = source.replace(old, new, 1)
    require(source.count('need(near_struct(previous,actual),') == 0,
            'Locality comparison fix was not applied')
    require(source.count('metric:metric(arm,metric)') == 0,
            'Report-comprehension fix was not applied')
    code = compile(source, str(original) + '::in_memory_hotfix', 'exec')
    name = 'phase8_original168_verified_inmemory_hotfix_20261118'
    module = types.ModuleType(name)
    # Crucial: all original sha(SOURCE), replay, input and output provenance
    # must still refer to the original unchanged driver, not this launcher.
    module.__file__ = str(original)
    sys.modules[name] = module
    exec(code, module.__dict__)
    require(module.SOURCE.resolve() == original.resolve(),
            'STOP: patched source-path lineage changed')
    require(module.OUT ==
            'ttk_runs_fixed/topology_finetuning/phase8_original168_eightstep_benchmark_20261118_r01',
            'STOP: experimental output path changed')
    require(module.SAMPLE_IDS == tuple(range(168)) and
            module.ARMS == ('cnn', 'control', 'dynamic'),
            'STOP: original experiment selection drifted')
    print('[HOTFIX] PASS: original SHA256 pinned=' + ORIGINAL_SHA256, flush=True)
    print('[HOTFIX] In-memory two-site SHA256=' + digest(source.encode('utf-8')), flush=True)
    print('[HOTFIX] Original driver, 168 scores, 168 locality results and model data unchanged', flush=True)
    return module


def selftest(root: Path) -> None:
    # Regression: JSON loaded from disk has string keys; live features have
    # integer H0/H1 keys. JSON normalization must preserve real metric values.
    reference = {'reader_skipped_zeros': {'gt': {'0': 18, '1': 13}},
                 'arms': {'dynamic': {'16.0': {'w22': 12.75}}}}
    live = {'reader_skipped_zeros': {'gt': {0: 18, 1: 13}},
            'arms': {'dynamic': {'16.0': {'w22': 12.75}}}}
    normalized = json.loads(json.dumps(live, allow_nan=False))
    require(reference == normalized, 'JSON key-normalization regression failed')
    changed = json.loads(json.dumps(live, allow_nan=False))
    changed['arms']['dynamic']['16.0']['w22'] = 12.76
    require(reference != changed, 'Scientific-value drift must remain detectable')
    m = bind_original(root)
    m.selftest()
    print('[HOTFIX-SELFTEST] PASS: JSON key-identity normalization; scientific values preserved; '
          'two exact source sites fixed in memory', flush=True)


def finalize(root: Path) -> None:
    m = bind_original(root)
    require(root.resolve() == m.SOURCE.parent.resolve(),
            'STOP: run from the exact PhIRE repo containing the pinned driver')
    out = root / m.OUT
    require(out.is_dir(), 'STOP: original 168 benchmark output missing')
    require(not (out / 'report').exists(),
            'STOP: report output already exists; do not overwrite immutable results')
    # The original install/preflight/check_replay/check_locality_sample functions
    # re-verify all original SHA256 identities. Only the two in-memory sites
    # described in PATCHES differ from the original implementation.
    p = m.install(root)
    print('[HOTFIX-FINALIZE] Starting independent 168-field locality re-evaluation', flush=True)
    m.verify_locality(root, out, p)
    print('[HOTFIX-FINALIZE] Locality independent verification PASSED; creating immutable report', flush=True)
    m.report(root, out, p)
    require((out/'report'/'manifest.json').is_file(), 'Report manifest missing')
    print('[HOTFIX-FINALIZE] PASS: verified locality and published original-scope 168-field report', flush=True)
    print('[HOTFIX-FINALIZE] Report: ' + str(out/'report'/'report.md'), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('selftest', 'finalize'))
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    a = parser.parse_args()
    root = a.repo_root.resolve()
    require(root.is_dir(), 'Repository root is absent')
    if a.mode == 'selftest':
        selftest(root)
    else:
        finalize(root)


if __name__ == '__main__':
    try:
        main()
    except Exception as ex:
        print('[PHASE8-168-HOTFIX] ERROR ' + type(ex).__name__ + ': ' + str(ex),
              file=sys.stderr, flush=True)
        sys.exit(1)
