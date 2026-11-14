#!/usr/bin/env python3
"""Non-TF smoke test of CV/PERS variant definitions and graph-path invariants.

Place alongside run_candidateF_cv_pers_ablation.py and run from repo root.
"""
from pathlib import Path
import importlib.util

p = Path(__file__).resolve().with_name('run_candidateF_cv_pers_ablation.py')
spec = importlib.util.spec_from_file_location('_phase6_cv_pers', p)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
f = a._load_frozen_runner()
a._install_variants(f)
arms = ('grad_pair_zero', 'grad_cv_only', 'grad_pers_only', 'grad_e2_low')
assert len({f.method_paths(f.VARIANTS[v]['method_name'])['model_dir'] for v in arms}) == 4
assert all(f.constraints_npz_for(v) == f.CONSTRAINTS_NPZ_2688 for v in arms)
assert all(f.VARIANTS[v]['lambda_speed'] == f.VARIANTS[v]['lambda_levelset']
           == f.VARIANTS[v]['lambda_crit'] == 0 for v in arms)
assert all(f.VARIANTS[v]['lambda_grad'] == 0.05 for v in arms)
# A frozen F1 method-name collision must be prevented by the runner.
assert 'candidateF_grad_E2_low_expanded2688' in f.COMPLETED_METHOD_NAMES or \
       any(x['method_name'] == 'candidateF_grad_E2_low_expanded2688'
           for x in f.VARIANTS.values())
print('PASS: four distinct arms; fixed constraints; no unintended auxiliary losses.')
print('NOTE: this checks configuration logic only; TF1 training/evaluation is NOT run.')
