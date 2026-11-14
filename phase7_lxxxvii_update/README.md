# Phase 7 / Part LXXXVII — WTK Native Wind Screening (October 15, 2026)

## Purpose

The LXXXVI geographic inventory and timestamp hotfix **already passed** on Spark. The next step is a **small meteorological screening pilot, not inference or training**: 64 native 500×500 100-m WTK fields, 16 per predeclared region. For each of eight valid 2010/2012 seasonal weeks, select UTC hours at offsets 0 and 84. This avoids choosing hours from model outcomes and provides two diurnal phases per week. Eight full-resolution `[u,v]` previews (February/August 2010 anchor 0 per region), with exact 5× block-average MR arrays, are saved for subsequent model-inference parity tests.

**Important:** The four candidates remain **provisional**. Their labels are geographic intentions, not certified wind regimes; *no* development/test assignments are made. This workpack does not test any pretrained/fine-tuned model, calculate PD/MT, produce TFRecords, or train weights. The original pretraining overlap is still unknown.

## Install (from `~/PhIRE`)

```bash
cd ~/PhIRE
unzip Phase7_LXXXVII_WTK_Wind_Screening_20261015.zip -d phase7_lxxxvii_update
( cd phase7_lxxxvii_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_lxxxvii_update/scripts/*.py
PATCH=phase7_lxxxvii_update/patches/phase7_notes_LXXXVI_to_LXXXVII.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part LX' dataset_generation_and_repair_notes.md | tail -4
```

Require last heading **Part LXXXVII**. This patch assumes the completed LXXXVI section is already in your master notes; don't reapply an earlier patch and don't force through conflicts.

## Offline preflight — no WTK network request

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
python3 phase7_lxxxvii_update/scripts/phase7_wtk_wind_screen.py --mode preflight \
  2>&1 | tee logs/phase7_lxxxvii_preflight_20261015.log
echo "Preflight exit: $?"
```

Require `[INVENTORY] PASS`, `[TIME] PASS`, `[PREFLIGHT] PASS` and exit `0`. Checks LXXXVI report SHA, LXXXVI script and candidate fingerprints, archived builder, geographic separation, exact 64 sample indices and 2007 fine-tuning exclusion.

## Live pilot (HSDS 64 time×window reads)

Run from a login shell whose Python has `h5pyd`, `numpy`, network access to WTK, and sufficient network bandwidth. If first-time WTK connection fails, check environment without modifying existing environments.

```bash
cd ~/PhIRE
export HS_ENDPOINT="https://developer.nlr.gov/api/hsds"
export HSDS_ENDPOINT="$HS_ENDPOINT"
python3 -c 'import h5pyd, numpy; print("h5pyd and numpy OK")'
set -o pipefail
python3 phase7_lxxxvii_update/scripts/phase7_wtk_wind_screen.py --mode pilot \
  2>&1 | tee logs/phase7_lxxxvii_pilot_20261015.log
echo "Pilot exit: $?"
```

The pilot prints one progress line per WTK snapshot, reads `windspeed_100m` and `winddirection_100m` only, computes wind validity, speed distributions and gradient summaries, and writes **only** the isolated final directory after all 64 samples have been processed. Network interruption leaves no *completed* output; restart with the same pilot command. No existing data or checkpoint is changed. At worst, an interrupted run can leave a named `.staging-...` directory from an abrupt process kill; it is not a completed dataset and must be inspected before any manual cleanup.

Successful gate: `[PILOT] PASS: 64 native WTK fields screened; 8 snapshot previews saved ...`, exit `0` (fewer than eight previews signals invalid wind data at a selected preview and should trigger manual review).

## Offline verification and inspection

```bash
cd ~/PhIRE
set -o pipefail
python3 phase7_lxxxvii_update/scripts/phase7_wtk_wind_screen.py --mode verify \
  2>&1 | tee logs/phase7_lxxxvii_verify_20261015.log
echo "Verification exit: $?"
OUT=ttk_runs_fixed/topology_finetuning/phase7_lxxxvii_wind_screen_20261015
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/wind_screen.md"
```

Require verify exit `0` and each SHA `OK`. Artifacts: `sample_screen.csv` (64 records); `screen_summary.json` (bounds, weather statistics, source fingerprints); `wind_screen.md`; and up to eight `preview_<region>_<yyyymmdd>.npz` containing native 500×500 float64 `hr_uv`, 100×100 float64 `mr_uv`, row/col rectangle, WTK index and UTC timestamp. Arrays are in WTK stored row/column orientation, not yet a certified TTK orientation. The numeric summary **cannot** establish geographic independence, climate regimes, unseen pretrained-model geography, model accuracy, or topological generalization.

## Next experiment

After confirming meteorological screening and actual footprint coordinates/corners, select a small, *development-only* frozen checkpoint inference pilot with PhIRE MR→HR. Before model evaluation, declare development vs locked external test regions. Validate original TensorFlow block-averaging against NumPy previews, normalization, GT and inference indices, full field shapes, and compute pointwise metrics only after this parity gate. Do not build large multi-region TFRecords or 500×500 persistence calculations before this step.
