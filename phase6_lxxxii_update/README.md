# Phase 6 / Part LXXXII — Original Phase 5E sparse-locality pilot

This package follows the successful user-confirmed LXXXI readiness gate. It is a **new, isolated 8-sample pilot**, not yet executed on Spark. It imports your **existing on-Spark** `spatial_pd/phase5e_sparse_locality_solver.py` unchanged, pinning its SHA-256 to the exact value printed by LXXXI. No copy of that solver or any model/PD data is bundled.

The fixed pilot samples are 0,24,48,69,96,120,144,167; models are CNN, gradient-only, CV-only, PERS-only, F1; dimensions D0/D1; radii 4,8,16,32,64,96,128,infinity. The pilot checks 40 unconstrained distances against frozen LXXIX and 128 CNN/F1 sample×radius rows against Phase 5E, including original coverage definitions. Only after all checks pass are 640 dimension and 320 aggregate rows saved to a new isolated output folder.

## 1. Installation

From `~/PhIRE`, transfer and extract:

```bash
cd ~/PhIRE
unzip Phase6_LXXXII_Phase5E_Locality_Pilot8_20261014.zip -d phase6_lxxxii_update
( cd phase6_lxxxii_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase6_lxxxii_update/scripts/phase6_lxxxii_locality_pilot.py
PATCH=phase6_lxxxii_update/patches/phase6_notes_LXXXI_to_LXXXII.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part LX' dataset_generation_and_repair_notes.md | tail -4
```

Do not force the patch if its check fails; local history may differ. **Do not rerun the Part LXXXI patch** if it is already applied.

## 2. Read-only preflight

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
/usr/bin/python3 phase6_lxxxii_update/scripts/phase6_lxxxii_locality_pilot.py \
  --mode preflight 2>&1 | tee logs/phase6_lxxxii_preflight_20261014.log
echo "Preflight exit: $?"
```

Require `[SOLVER] PASS`, `[REFERENCE] PASS`, and `[PREFLIGHT] PASS` with exit status `0`. This preflight needs no GUDHI or VTK import.

## 3. Matched spatial pilot (only after preflight passes)

Use the known isolated `gudhi-audit` environment with VTK 9.6, GUDHI and SciPy matching:

```bash
cd ~/PhIRE
set -o pipefail
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase6_lxxxii_update/scripts/phase6_lxxxii_locality_pilot.py \
  --mode pilot 2>&1 | tee logs/phase6_lxxxii_pilot_20261014.log
echo "Pilot exit: $?"
```

All 8 samples must pass the original GUDHI dimension-wise infinity check, 40 five-arm aggregate infinity regressions and 128 frozen CNN/F1 spatial regressions. The script refuses to overwrite an existing pilot output directory. Failures before the output stage write no experimental outputs.

## 4. Verify and view the pilot report

```bash
cd ~/PhIRE
set -o pipefail
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase6_lxxxii_update/scripts/phase6_lxxxii_locality_pilot.py \
  --mode verify 2>&1 | tee logs/phase6_lxxxii_verify_20261014.log
echo "Verify exit: $?"
OUT=ttk_runs_fixed/topology_finetuning/phase6_lxxxii_spatial_pilot8_20261014
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/pilot_summary.md"
```

`--mode verify` checks stored SHA/source continuity and output integrity; it does not rerun all 320 spatial matchings. The pilot execution is the numeric solver computation.

## Scientific scope

This is an implementation and data-alignment pilot, not the 168-sample result. Its stdout and report may show qualitative trends but do not support manuscript claims about dataset-wide spatial improvements yet. The exact Phase 5E parser, coordinate convention and matching solver are reused; `r=∞` agrees with corrected GUDHI `W_{2,2}`, **not** `W_{2,∞}`. F1 training lineage and new Phase 6 merge-tree results remain open. Preserve all prior outputs and `topoinvis-2026-arxiv-v1.1`.
