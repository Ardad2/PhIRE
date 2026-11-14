#!/usr/bin/env python3
"""Run the strict gradient + CV/PERS 2x2 ablation against tagged PhIRE F1.

Place alongside scripts/run_candidateF_expanded2688_finetune.py in a PhIRE
checkout of topoinvis-2026-arxiv-v1.1. This wrapper modifies ONLY in-memory
variant tables and invokes the existing released runner. It does not modify
historical code, checksums, checkpoints or results.

Variants (all use the released runner's *same* custom E2 training loop):
  grad_pair_zero  : grad=0.05, CV=0,     PERS=0    [procedural control]
  grad_cv_only    : grad=0.05, CV=0.004, PERS=0
  grad_pers_only  : grad=0.05, CV=0,     PERS=0.002
  grad_e2_low     : grad=0.05, CV=0.004, PERS=0.002 [frozen existing F1]

Use --print-config (no TensorFlow needed) then --dry-run for each new variant.
The released runner refuses nonempty existing output paths and keeps the
training/evaluation datasets and fixed GT TTK constraints unchanged.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

BASE_SCRIPT = Path(__file__).resolve().with_name('run_candidateF_expanded2688_finetune.py')
# Git blob SHA of the actual F runner in GitHub tag topoinvis-2026-arxiv-v1.1.
BASE_GIT_BLOB_SHA = 'b7c5359952a4e2cf30e2b6f4341c66bcbe7ee7c8'


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b'blob ' + str(len(data)).encode('ascii') + b'\0' + data).hexdigest()


def _load_frozen_runner():
    if not BASE_SCRIPT.is_file():
        raise SystemExit(f'[error] Missing tagged F runner: {BASE_SCRIPT}')
    actual = _git_blob_sha(BASE_SCRIPT)
    if actual != BASE_GIT_BLOB_SHA:
        raise SystemExit(
            '[error] Candidate F source does not match GitHub release tag '
            'topoinvis-2026-arxiv-v1.1. Refusing a non-comparable run.\n'
            f'  expected blob: {BASE_GIT_BLOB_SHA}\n  actual blob:   {actual}\n'
            'Use a clean checkout of the release tag; never modify the frozen runner.'
        )
    spec = importlib.util.spec_from_file_location('_phase6_frozen_candidate_f', BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit('[error] Could not import frozen Candidate F runner.')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_variants(f):
    base = f.VARIANTS['grad_e2_low']
    assert base['lambda_grad'] == 0.05 and base['lambda_ttkcv'] == 0.004
    assert base['lambda_ttkpers'] == 0.002 and base['requires_e2'] is True
    new = {
        'grad_pair_zero': ('candidateF_grad_pairzero_expanded2688', 0.0, 0.0),
        'grad_cv_only': ('candidateF_grad_CV_only_expanded2688', 0.004, 0.0),
        'grad_pers_only': ('candidateF_grad_PERS_only_expanded2688', 0.0, 0.002),
    }
    for variant, (method, lam_cv, lam_pers) in new.items():
        if variant in f.VARIANTS or any(x['method_name'] == method for x in f.VARIANTS.values()):
            raise SystemExit(f'[error] Experiment name collision: {variant} / {method}')
        cfg = dict(base)
        cfg.update(method_name=method, lambda_ttkcv=lam_cv,
                   lambda_ttkpers=lam_pers, requires_e2=True)
        f.VARIANTS[variant] = cfg
        # The original _validate_variant_configs verifies every EXPECTED_ASSERTIONS entry.
        f.EXPECTED_ASSERTIONS[variant] = {
            key: cfg[key] for key in (
                'lambda_speed', 'lambda_grad', 'lambda_levelset', 'lambda_crit',
                'lambda_ttkcv', 'lambda_ttkpers', 'requires_e2'
            )
        }
        f.PREFERRED_ORDER.append(variant)
    f._validate_variant_configs()
    # Scientific design: every 2x2 arm must take the identical E2 code path.
    arms = ['grad_pair_zero', 'grad_cv_only', 'grad_pers_only', 'grad_e2_low']
    assert all(f.VARIANTS[a]['requires_e2'] is True for a in arms)
    assert len({f.VARIANTS[a]['method_name'] for a in arms}) == 4
    assert {(f.VARIANTS[a]['lambda_ttkcv'], f.VARIANTS[a]['lambda_ttkpers'])
            for a in arms} == {(0.0, 0.0), (0.004, 0.0), (0.0, 0.002), (0.004, 0.002)}


def main() -> int:
    f = _load_frozen_runner()
    _install_variants(f)
    # Allow existing F runner to own argument parsing, preflight, collision checks,
    # TF1 graph, checkpoint restoration, inference and output naming.
    return f.main()


if __name__ == '__main__':
    sys.exit(main())
