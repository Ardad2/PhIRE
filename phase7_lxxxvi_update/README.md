# Phase 7 — Part LXXXVI: WTK geographic and temporal inventory (October 15, 2026)

**Scope:** Start the agreed evaluation-generalization study. This is a *read-only geographic and time-index inventory*, not a dataset-generation, training, inference or topology run. The first purpose is to determine which 500×500 native WIND Toolkit windows can be obtained without overlap with the archived Mid-Atlantic benchmark and with one another. Outputs are isolated; the project and all previous experiments remain frozen.

## Install and master-notes update

```bash
cd ~/PhIRE
unzip Phase7_LXXXVI_WTK_Region_Inventory_20261015.zip -d phase7_lxxxvi_update
( cd phase7_lxxxvi_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_lxxxvi_update/scripts/phase7_region_inventory.py
PATCH=phase7_lxxxvi_update/patches/phase6_notes_LXXXV_to_LXXXVI.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part LX' dataset_generation_and_repair_notes.md | tail -4
```

Expected final heading: Part LXXXVI. If `git apply --check` fails, do **not** use `--force`, skip, or reapply earlier patches. Inspect the notes' last heading and report the diagnostic. The intended base is the on-host notes through Part LXXXV.

## 1. Offline preflight — no HSDS connection

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
python3 phase7_lxxxvi_update/scripts/phase7_region_inventory.py --mode preflight \
  2>&1 | tee logs/phase7_lxxxvi_preflight_20261015.log
echo "Preflight exit: $?"
```

Require `[PREFLIGHT] PASS`, exit `0`. No data files or inventories are generated in this mode.

## 2. Check the WTK HSDS environment, then probe (read-only)

The original PhIRE extraction code requires `numpy` and `h5pyd`. Run with **the same Python environment that successfully ran `build_example_data_extension_500.py`**. If this environment lacks `h5pyd`, stop and send the error; do not modify the long-lived model environments or install anything blindly.

```bash
cd ~/PhIRE
export HS_ENDPOINT="https://developer.nlr.gov/api/hsds"
export HSDS_ENDPOINT="$HS_ENDPOINT"
python3 -c 'import numpy, h5pyd; print("numpy and h5pyd imports OK")'
```

After the import test passes:

```bash
cd ~/PhIRE
set -o pipefail
python3 phase7_lxxxvi_update/scripts/phase7_region_inventory.py --mode probe \
  2>&1 | tee logs/phase7_lxxxvi_probe_20261015.log
echo "Probe exit: $?"
```

Probe accesses only `/nrel/wtk-us.h5` metadata, `coordinates`, and `datetime`; it reads no full meteorological field. It validates the original 168-field crop against the recovered rows `[720,1220)`, cols `[2102,2602)`. For each candidate, it resolves the **actual nearest grid cell**, the exact 500×500 source window, and the coordinates of its four corners. It enforces an 80-native-grid-cell separation buffer from the existing study and among any provisionally selected regions. Fourteen candidate points and eight possible seasonal weeks are *proposals*, not a locked benchmark. If the remote dataset is unavailable or coordinates differ from the archived reference, it fails without writing a report.

No location is labeled as a true held-out test region until the candidate coordinates and temporal indices are reviewed and the split is frozen. Pretrained CNN/GAN original training-region membership is not established.

## 3. Offline report-integrity verification

```bash
cd ~/PhIRE
set -o pipefail
python3 phase7_lxxxvi_update/scripts/phase7_region_inventory.py --mode verify \
  2>&1 | tee logs/phase7_lxxxvi_verify_20261015.log
echo "Verify exit: $?"

OUT=ttk_runs_fixed/topology_finetuning/phase7_lxxxvi_region_inventory_20261015
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/inventory.md"
```

Verify checks output SHA256 and the local script, candidate-list and legacy builder hashes recorded by the probe; it does not re-query HSDS. The probe **refuses overwrite** of an existing output directory. Do not remove an existing report to rerun; keep its provenance.

## Scientific next actions **after** reviewing the inventory

1. Agree on an evaluation-only geographic development set and genuinely locked external test regions. Check actual footprints, coverage, terrain/coast type, and whether temporal windows conflict with the historical 2,688 training fields. Original pretrained-model geography cannot be proven from the manuscript alone.
2. Perform a small (8–24 hours/region) extraction pilot in a separately named workpack, with original native 500×500 `[u,v]` data, factor-5 *block-average* MR inputs and unchanged pretrained normalization. Validate `dataGT`, `dataIN`, orientation, time indices and full-field inference on the six original/fine-tuned checkpoints (CNN, GAN, gradient-only, CV-only, PERS-only, F1).
3. On new non-training regions, freeze evaluation comparisons before examining model rankings; begin with pointwise metrics and historic 160×160 `[0,0]`, then fixed multiple tiles and a small full-500×500 PD/MT pilot. Full-500 diagram distances are different topological objects and should not be numerically conflated with 160 crops.
4. Only after the frozen-model geographic study decide whether to generate mixed-region training TFRecords and **new aligned E2 constraints** for new training fields. Never reuse the original 2,688 constraint NPZ with new records.

## Known constraints

- The original Stengel et al. (2020) pretrained wind models trained on 37,972 WIND Toolkit patches and were validated on 4,383 more, covering 2007–2013. Their specific training-patch locations are not established from the article. New regions can be *unseen during the user's fine-tuning* but are **not yet proven unseen during initial PhIRE pretraining**.
- The old case uses WTK index `0..167` and rect `[720:1220,2102:2602]`. Preserve it as the historical, previously analyzed benchmark.
- Date candidates are not final evaluations; no historical 2,688-sample time-window exclusion is yet demonstrated.
- Index-space nonoverlap and an 80-cell buffer do not establish meteorological independence.
- This workpack makes **no scientific performance or generalization claims**.
