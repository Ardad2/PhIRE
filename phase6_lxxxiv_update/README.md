# Phase 6 — Part LXXXIV, full-spatial closeout and historical F1 evidence inventory

**Status:** The user-supplied LXXXIII Spark logs show successful full 168-sample run and independent verification with exit `0`. This workpack only *summarizes and fingerprints* those frozen results; it does **not** recompute TTK/GUDHI/matching and does not close F1 training lineage.

## 1. Transfer, check, and patch notes

```bash
cd ~/PhIRE
unzip Phase6_LXXXIV_Spatial_Closeout_And_F1_Evidence_20261014.zip -d phase6_lxxxiv_update
( cd phase6_lxxxiv_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase6_lxxxiv_update/scripts/phase6_lxxxiv_spatial_closeout.py
PATCH=phase6_lxxxiv_update/patches/phase6_notes_LXXXIII_to_LXXXIV.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part LX' dataset_generation_and_repair_notes.md | tail -4
```

The final heading must be **Part LXXXIV**. Do not force patch application or reset unrelated changes.

## 2. Read-only preflight

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
/usr/bin/python3 phase6_lxxxiv_update/scripts/phase6_lxxxiv_spatial_closeout.py \
  --mode preflight 2>&1 | tee logs/phase6_lxxxiv_preflight_20261014.log
echo "Preflight exit: $?"
```

Require `[SOURCE] PASS`, `[PREFLIGHT] PASS` and exit `0`. This hashes all seven frozen LXXXIII files and independently recomputes the 40 macro means from the full 6,720-row sample/radius table. No write is performed.

## 3. Generate isolated closeout and F1 evidence inventory

```bash
cd ~/PhIRE
set -o pipefail
/usr/bin/python3 phase6_lxxxiv_update/scripts/phase6_lxxxiv_spatial_closeout.py \
  --mode report 2>&1 | tee logs/phase6_lxxxiv_report_20261014.log
echo "Report exit: $?"
```

Creates only `ttk_runs_fixed/topology_finetuning/phase6_lxxxiv_spatial_closeout_20261014/` with `spatial_closeout.md`, `f1_evidence_inventory.md`, `source_sha256.json`, and `SHA256SUMS`. Refuses to overwrite an existing completed directory; use `--mode verify` instead. F1 evidence inventory reports presence/hashes of historical canonical and batch training logs, archived evaluation report, and four expected model `cnn.index` checkpoint files. It does **not** infer random-seed/optimizer equivalence from their presence.

## 4. Verify and inspect

```bash
cd ~/PhIRE
set -o pipefail
/usr/bin/python3 phase6_lxxxiv_update/scripts/phase6_lxxxiv_spatial_closeout.py \
  --mode verify 2>&1 | tee logs/phase6_lxxxiv_verify_20261014.log
echo "Verify exit: $?"
OUT=ttk_runs_fixed/topology_finetuning/phase6_lxxxiv_spatial_closeout_20261014
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/spatial_closeout.md"
cat "$OUT/f1_evidence_inventory.md"
```

The report includes all eight radii, both GT- and SR-side persistence-mass coverage, cost paired wins and medians, CV/PERS factorial contrasts, and seven chronological 24-hour cost blocks at selected radii. The original LXXXIII full/verify gates are already complete and should not be rerun merely to generate this report.

## Scientific boundaries and recommended next stage

This is a matched descriptive study of consecutive hourly wind-field patches with a single model run per factorial cell. Coverage is *not* spatial displacement, and radius is in native grid-cell units. The key result is a tight-radius gradient-only vs wider-radius PERS-only ranking reversal; use all radii/paired comparisons before manuscript claims. F1 training/optimizer/seed-lineage closure and new-arm corrected merge-tree analysis are separate outstanding tasks; do not treat this SHA-index as proof of matched training randomness. All archived Phase 5E, LXXIX, LXXXIII, published tag, weights, arrays and TTK outputs remain unchanged.
