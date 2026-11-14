# Phase 7 — Part XCVI: 16-field six-model development inference

**Status:** prepared and synthetic-tested. Run against real Spark checkpoints before drawing any conclusions. Does **not** touch locked Upper Midwest/Pacific test predictions or change the frozen XCI protocol.

This package is *flat*: extract once with `-d phase7_xcvi_update` (not into an extra nested directory). The package checksum, Python compiler and notes patch should all pass before executing.

## 1. Install and record research-plan notes

```bash
cd ~/PhIRE
unzip Phase7_XCVI_Development_Seasonal16_Inference_20261018.zip -d phase7_xcvi_update
( cd phase7_xcvi_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_xcvi_update/scripts/*.py
PATCH=phase7_xcvi_update/patches/phase7_notes_XCV_to_XCVI.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part ' dataset_generation_and_repair_notes.md | tail -3
```

If files already exist or the notes patch fails, STOP and inspect; do not overwrite/force/reapply. Last heading should be Part XCVI.

## 2. Selftest and offline preflight (no inference)

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
SCRIPT=phase7_xcvi_update/scripts/phase7_xcvi_dev_seasonal16_infer.py
/usr/bin/python3 "$SCRIPT" --mode selftest --repo-root "$PWD"
/usr/bin/python3 "$SCRIPT" --mode preflight --repo-root "$PWD" 2>&1 | tee logs/phase7_xcvi_preflight_20261018.log
echo "Preflight exit: $?"
```

Require `[SELFTEST] PASS`, `[SOURCES] PASS`, `[DEV-SCOPE] PASS`, `[PREFLIGHT] PASS`, exit 0. The preflight verifies XCV, XC, XCI plus six checkpoints and also recomputes the archived XC numerical metrics. It uses no WTK/HSDS or TensorFlow.

## 3. Six-model inference (development only)

Only if preflight passed:

```bash
cd ~/PhIRE
set -o pipefail
SCRIPT=phase7_xcvi_update/scripts/phase7_xcvi_dev_seasonal16_infer.py
TF_CPP_MIN_LOG_LEVEL=2 TF_FORCE_GPU_ALLOW_GROWTH=true /usr/bin/python3 "$SCRIPT" --mode infer --repo-root "$PWD" 2>&1 | tee logs/phase7_xcvi_infer_20261018.log
echo "Inference exit: $?"
```

Expect six `[MODEL-PASS]` messages and `[INFER-GATE] PASS` (96 full500 model outputs = 6 models × 16 fields), exit 0. The script compares each of the four reused examples against archived XC predictions (`max abs <=1e-3 m/s`). If it fails, stop and share the exact error; **do not** loosen the numerical gate before investigating. No model checkpoint is changed.

## 4. Verify and review

```bash
cd ~/PhIRE
set -o pipefail
SCRIPT=phase7_xcvi_update/scripts/phase7_xcvi_dev_seasonal16_infer.py
/usr/bin/python3 "$SCRIPT" --mode verify --repo-root "$PWD" 2>&1 | tee logs/phase7_xcvi_verify_20261018.log
echo "Verify exit: $?"
OUT=ttk_runs_fixed/topology_finetuning/phase7_xcvi_dev_seasonal16_inference_20261018
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/seasonal16_inference_report.md"
```

Require `[VERIFY-GATE] PASS`, exit 0, eleven checksum successes. `verify` is read-only and does not import TensorFlow. An existing completed inference output must be verified, not rerun or overwritten.

## Important scope limits

- Four 2010 seasons × two development regions × two hours per region-week. These 16 fields do **not** replace the frozen XCI all-168-hour region-week benchmark, the nine-tile sensitivity checks, or 2012 replication.
- No locked-test predictions or hyperparameter changes. Do not use development rankings as final generalization findings.
- XC four-field tolerance is only a checkpoint/preprocessing regression gate, and will not certify numerical equality across all GPU hardware.
- Original PhIRE pretraining overlap with geographic footprints is not established.
- This runs the original native paired PhIRE path, not an independently reconstructed TensorFlow generator.

## Files

- `scripts/phase7_xcvi_dev_seasonal16_infer.py` — full driver
- `xcvi_append.md` — notes section / scope
- `patches/phase7_notes_XCV_to_XCVI.patch` — master-notes append, apply once
- `SHA256SUMS` — workpack files
