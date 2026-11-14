# Phase 7 Part CVI — seven 2010 development region-weeks, six-model inference

## Scientific and execution boundaries

This extends the original SHA-pinned **Part CI** full500 six-model inference, with week-bound metadata derived from the sealed **Part CV** native input files and **XCVI** hour-0/84 model prediction anchors. It does not rewrite or modify the CI/CI-XCV/Part CV files, or the frozen XCI protocol. Source checks, checkpoint hashes, immutable files and historical prediction regression are required gates. Original February Gulf inference is independently reverified, but **not rerun**. The original model code runs one checkpoint at a time.

**Expected per new week**: 168 ordered MR/GT reference pairs, six arrays each containing 168 full500 U/V predictions, 12 cross-checks against original XCVI model prediction snapshots, full per-hour + pooled reconstruction metrics, SHA256 integrity for 13 output files. Overall **seven weeks × 1,008 = 7,056** additional full-resolution predictions. No topological scores yet. No locked external regions. 2012 replication is separate.

## Install in `~/PhIRE` (flat ZIP layout)

```bash
cd ~/PhIRE
unzip Phase7_CVI_Development_Seven_2010_Week_Inference_20261019.zip -d phase7_cvi_update
( cd phase7_cvi_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_cvi_update/scripts/*.py
PATCH=phase7_cvi_update/patches/phase7_notes_CV_to_CVI.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part ' dataset_generation_and_repair_notes.md | tail -3
```

## Read-only self-test and all-week readiness

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
SCRIPT=phase7_cvi_update/scripts/phase7_cvi_dev2010_sevenweek_inference.py
/usr/bin/python3 "$SCRIPT" --mode selftest --repo-root "$PWD"
/usr/bin/python3 "$SCRIPT" --mode inventory --repo-root "$PWD" \
  2>&1 | tee logs/phase7_cvi_inventory_20261019.log
/usr/bin/python3 "$SCRIPT" --mode preflight-all --repo-root "$PWD" \
  2>&1 | tee logs/phase7_cvi_preflight_all_20261019.log
echo "Preflight exit: $?"
```

**Stop** unless all seven frozen weeks report `[PREFLIGHT] PASS`, original CI/FEB re-verification passes and exit code is 0. The preflight checks source/checkpoint integrity and disk/memory budgets but runs no TF inference.

## First new week: CNN-only smoke test

```bash
TF_CPP_MIN_LOG_LEVEL=2 TF_FORCE_GPU_ALLOW_GROWTH=true \
/usr/bin/python3 "$SCRIPT" --mode infer --repo-root "$PWD" \
  --region gulf_inland --week-start 2010-05-01T00:00:00 \
  --max-new-models 1 \
  2>&1 | tee logs/phase7_cvi_gulf_may_cnn_20261019.log
echo "CNN smoke exit: $?"
```

Require `[MODEL-PASS] cnn`, `[PARTIAL-GATE] PASS: 1/6`, exit 0. The in-memory adapter uses the exact original CI routine and requires both XCVI anchor predictions to agree within its original tolerance.

## Resume all seven in one batch

Run in `tmux` if you wish. Inference occurs **sequentially** per week and per checkpoint; each sealed model/week will be integrity checked before resume. A completed week is independently verified and skipped. Do not start multiple runs into the same output directory concurrently.

```bash
TF_CPP_MIN_LOG_LEVEL=2 TF_FORCE_GPU_ALLOW_GROWTH=true \
/usr/bin/python3 "$SCRIPT" --mode infer-all --repo-root "$PWD" \
  2>&1 | tee logs/phase7_cvi_infer_all_20261019.log
echo "Infer-all exit: $?"
```

Require 7 sealed weeks and the final `[ALL-INFER-ALL] PASS: 7/7`, exit 0.

## Independently verify all seven sealed inference weeks

```bash
/usr/bin/python3 "$SCRIPT" --mode verify-all --repo-root "$PWD" \
  2>&1 | tee logs/phase7_cvi_verify_all_20261019.log
echo "Verify-all exit: $?"
/usr/bin/python3 "$SCRIPT" --mode inventory --repo-root "$PWD"
```

Require seven `[VERIFY-GATE] PASS`, final `[ALL-VERIFY-ALL] PASS: 7/7` and exit 0. Each output folder has 13 checksums; `verify-all` checks them while recomputing metrics. No additional `sha256sum -c` pass is strictly necessary, but it is fine to run one for any individual week.

**Important:** NumPy 1.x/2.x optional binary-import warnings in the existing TensorFlow environment are known; the actual exit codes, model gates, reference/input/anchor numerical checks and SHA checks determine success. Do not change frozen checkpoints or dependency environments merely to hide nonfatal warnings.
