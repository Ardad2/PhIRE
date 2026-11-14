# Phase 7 CI — full 168-hour Gulf inland development-only six-checkpoint inference

**Status:** prepared, tested locally with synthetic/mock inference ONLY. Real Spark TensorFlow execution is pending. Do not call this a locked geographic test, a new training experiment, or topology scoring.

This flat-layout ZIP is designed to be extracted into `phase7_ci_update`. It depends on the successfully verified Part C 168-hour source artifact, XCVI 16-field six-model inference and frozen XCI geographic evaluation protocol. It pins original C and XCVI source-script SHA-256 and uses the original XCVI TFRecord writer, XC frozen checkpoint prefixes, and original PhIRE `test_paired(r=[5],batch_size=1)` inference. No changes to historical source, checkpoints, notes already present, original 16-field results, external-test regions, or hyperparameters.

## Install (Spark PhIRE root)

```bash
cd ~/PhIRE
unzip Phase7_CI_Development_Fullweek_Six_Model_Inference_20261018.zip -d phase7_ci_update
( cd phase7_ci_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_ci_update/scripts/*.py
PATCH=phase7_ci_update/patches/phase7_notes_C_to_CI.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part ' dataset_generation_and_repair_notes.md | tail -3
```

Stop if any checksum, compilation or Git patch check fails. Do not force a patch or re-extract into an existing directory with overwritten members. The last heading should be Part CI.

## Read-only self-test and preflight

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
SCRIPT=phase7_ci_update/scripts/phase7_ci_dev_fullweek_inference.py
/usr/bin/python3 "$SCRIPT" --mode selftest --repo-root "$PWD"
/usr/bin/python3 "$SCRIPT" --mode preflight --repo-root "$PWD" \
  2>&1 | tee logs/phase7_ci_preflight_20261018.log
echo "Preflight exit: $?"
```

Require `[SELFTEST] PASS`, `[SOURCES] PASS`, `[SCOPE] PASS`, `[BUDGET]` and `[PREFLIGHT] PASS`, exit 0. Preflight verifies all 168 original source files and 16-field XCVI reference/checkpoint hashes; it does not write experimental output or import TensorFlow. A full-week source budget of roughly 666.5 MiB of native HR/MR, 3.76 GiB of six raw output arrays, plus TFRecord and scratch space informs the conservative requirement of ≥12 GiB disk free and ≥10 GiB memory available. This does **not** guarantee TensorFlow GPU peak memory or runtime; stop on real resource errors.

## CNN-only pilot (first immutable model)

```bash
cd ~/PhIRE
set -o pipefail
SCRIPT=phase7_ci_update/scripts/phase7_ci_dev_fullweek_inference.py
TF_CPP_MIN_LOG_LEVEL=2 TF_FORCE_GPU_ALLOW_GROWTH=true \
/usr/bin/python3 "$SCRIPT" --mode infer --repo-root "$PWD" --max-new-models 1 \
  2>&1 | tee logs/phase7_ci_cnn_smoke_20261018.log
echo "CNN smoke exit: $?"
```

Expect `[MODEL-PASS] cnn: 168/168`, `[PARTIAL-GATE] PASS: 1/6`, exit 0. The native PhIRE TensorFlow import can emit optional NumPy 2.x warnings about system `numexpr` or `bottleneck`; **do not** downgrade a working global environment just to silence warnings. Actual paired inference and saved array regression gates must pass, not merely TensorFlow import.

## Resume five remaining frozen models

```bash
cd ~/PhIRE
set -o pipefail
TF_CPP_MIN_LOG_LEVEL=2 TF_FORCE_GPU_ALLOW_GROWTH=true \
/usr/bin/python3 "$SCRIPT" --mode infer --repo-root "$PWD" \
  2>&1 | tee logs/phase7_ci_infer_20261018.log
echo "Inference exit: $?"
```

Expect `[RESUME] PASS: 1/6`, successful GAN/gradient/CV/PERS/F1 `[MODEL-PASS]`, then `[INFER-GATE] PASS` and exit 0. This is a multi-model run: use `tmux` if disconnection is a concern. On normal errors, already completed model predictions are checksum protected for resumption. If the script reports unexpected/orphaned staging files, **stop and inspect**; do not delete validated outputs or bypass a gate.

## Independent offline numerical verification

```bash
cd ~/PhIRE
set -o pipefail
/usr/bin/python3 "$SCRIPT" --mode verify --repo-root "$PWD" \
  2>&1 | tee logs/phase7_ci_verify_20261018.log
echo "Verify exit: $?"
OUT=ttk_runs_fixed/topology_finetuning/phase7_ci_dev_gulf_inland_feb2010_inference168_20261018
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/fullweek_inference_report.md"
```

Require `[VERIFY-GATE] PASS`, exit 0, **13** output checksums OK. This reruns the original reconstruction error calculations on every field, and verifies the predictions at hours 0,84 against XCVI 16-field outputs for all six checkpoints. It does not import TensorFlow or recompute the actual neural networks independently.

## Scientific boundaries

- Development Gulf inland February 2010 only; 168 hours are temporally dependent, **one region-week block** under XCI. Do not infer confidence intervals from treating hours as independent blocks.
- All outputs are full500 `[u,v]` reconstruction and pointwise/speed errors. **No topology in Part CI**. The primary topology target for a future stage remains the exact top-left 160×160 speed crop, canonical TTK diagrams and corrected GUDHI/Phase5E calculations.
- Original pretrained PhIRE dataset membership for this geography/time window remains unknown. Locked `upper_midwest` and `pacific_inland` remain untouched.
- Failed preflight/inference/verification must be reported honestly. The local synthetic integration is not evidence of Spark runtime success.
