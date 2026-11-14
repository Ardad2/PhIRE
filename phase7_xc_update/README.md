# Phase 7 — Part XC: development-only six-model inference pilot

**Status:** Prepared, *not run on Spark*. Zip layout is **flat**, not prefixed with an internal `phase7_xc_update/` folder. Do not extract a second time into an identically named subdirectory. The package itself contains no predictions or datasets.

## Installation and notes patch

```bash
cd ~/PhIRE
unzip Phase7_XC_Development_Six_Model_Inference_20261015.zip -d phase7_xc_update
( cd phase7_xc_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_xc_update/scripts/phase7_dev_six_model_pilot.py
PATCH=phase7_xc_update/patches/phase7_notes_LXXXIX_to_XC.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part ' dataset_generation_and_repair_notes.md | tail -3
```

Expected last heading: `# Part XC ...`. The patch only advances the notes from fully installed Part LXXXIX. If patch check fails, do not force it.

## Preflight — no TensorFlow, no write

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
/usr/bin/python3 phase7_xc_update/scripts/phase7_dev_six_model_pilot.py --mode selftest
/usr/bin/python3 phase7_xc_update/scripts/phase7_dev_six_model_pilot.py --mode preflight --repo-root "$PWD" \
  2>&1 | tee logs/phase7_xc_dev_preflight_20261015.log
echo "Preflight exit: $?"
```

Expect `[SELFTEST] PASS`, `[FROZEN-SPLIT] PASS`, six `[CHECKPOINT]` lines, `[DEV-SCOPE] PASS`, and `[PREFLIGHT] PASS` with exit 0. This step audits all checkpoint `cnn/gan.index` and data shards, which may take a short time; it reads no locked-region previews and imports no TensorFlow.

## Inference — only 4 development previews × 6 archived models

```bash
cd ~/PhIRE
set -o pipefail
TF_CPP_MIN_LOG_LEVEL=2 TF_FORCE_GPU_ALLOW_GROWTH=true /usr/bin/python3 \
  phase7_xc_update/scripts/phase7_dev_six_model_pilot.py \
  --mode infer --repo-root "$PWD" \
  2>&1 | tee logs/phase7_xc_dev_inference_20261015.log
echo "Inference exit: $?"
```

Expect six `[MODEL-PASS]` lines, `[INFER-GATE] PASS` and exit 0. May take time for each TF graph/checkpoint. The TensorFlow optional-extension warnings encountered in Part LXXXVIII might recur; do not mutate system packages. On error share the last useful console lines before taking action.

## Verification and report

```bash
cd ~/PhIRE
set -o pipefail
/usr/bin/python3 phase7_xc_update/scripts/phase7_dev_six_model_pilot.py --mode verify --repo-root "$PWD" \
  2>&1 | tee logs/phase7_xc_dev_verify_20261015.log
echo "Verification exit: $?"
OUT=ttk_runs_fixed/topology_finetuning/phase7_xc_development_inference_pilot_20261015
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/dev_inference_report.md"
```

Expect `[VERIFY-GATE] PASS` and SHA `OK` for every file. This is a four-field development sanity gate only: no model selection, topology inference, or generalization claims. Upper Midwest / Pacific inland are locked external test areas; never run pilot inference on their previews.

### Safety

- No checkpoint writes, training, or read of locked-region model outputs.
- Source/checkpoint/report SHA256 recorded; output refuses overwrite.
- TensorFlow paired inference is delegated to original unchanged `PhIREGANs.test_paired`; MR and GT must match archived previews within 2e-5 after normalization inversion.
- Uses full 500x500 fields throughout and retains 100x100 MR only as input.
- Requires existing frozen split under `ttk_runs_fixed/topology_finetuning/phase7_lxxxix_locked_region_split_20261015`.
