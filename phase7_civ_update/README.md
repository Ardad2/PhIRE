# Part CIV — Gulf inland full-week spatial locality (development only)

Extends the verified 16-field Phase7 XCIX spatial-locality analysis to the immutable 168-hour CIII numerical and CII canonical PD sources. Uses original unchanged XCIV/Phase5E sparse matching and the documented original coordinate/zero-persistence policy. The 168 serially correlated hours belong to **one region-week**, not an external/generalization benchmark. No locked-region access, TTK, TensorFlow, HSDS or training.

**Key row counts:** 168 hours × 6 model arms × 8 original grid-cell radii = **8,064** aggregate rows; **16,128** H0/H1 rows; 7 daily windows × 6 arms = **42** daily descriptive rows. All 1008 infinite-radius costs must agree with Part CIII; at week hours 0 and 84, 96 aggregate and 192 dimension rows must agree with Part XCIX. Model arms: `cnn`, `gan`, `grad`, `cv`, `pers`, `f1`. Radii: `4,8,16,32,64,96,128,inf` WTK grid cells (not km). The original top-left 160×160 scalar-speed crop is mandatory.

## Installation (flat ZIP layout)

```bash
cd ~/PhIRE
unzip Phase7_CIV_Development_Fullweek_Spatial_Locality_20261019.zip -d phase7_civ_update
( cd phase7_civ_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_civ_update/scripts/*.py
PATCH=phase7_civ_update/patches/phase7_notes_CIII_to_CIV.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part ' dataset_generation_and_repair_notes.md | tail -3
```

## Read-only self-test, source and anchor preflight

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
SCRIPT=phase7_civ_update/scripts/phase7_civ_dev_fullweek_locality.py
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$SCRIPT" --mode selftest --repo-root "$PWD"
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$SCRIPT" --mode preflight --repo-root "$PWD" \
  2>&1 | tee logs/phase7_civ_preflight_20261019.log
echo "Preflight exit: $?"
```

Require `[SELFTEST] PASS`, `[SOURCES] PASS`, both `[ANCHOR] PASS`, `[PREFLIGHT] PASS`, exit `0`. Preflight validates all CIII numeric outputs and XCIX lineage before computing original locality values for the 0/84 regression. This may take time. No output stage written.

## Two-hour real locality smoke test

```bash
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$SCRIPT" \
  --mode compute --repo-root "$PWD" --max-new-hours 2 \
  2>&1 | tee logs/phase7_civ_first2_20261019.log
echo "Smoke exit: $?"
```

Require two `[HOUR-LOCALITY]` records and `[PARTIAL-GATE] PASS: 2/168`, exit `0`.

## Resume all 168 hours (recommend tmux)

```bash
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$SCRIPT" \
  --mode compute --repo-root "$PWD" \
  2>&1 | tee logs/phase7_civ_compute_20261019.log
echo "Compute exit: $?"
```

Require `[RESUME] PASS: 2/168`, then `[LOCALITY-GATE] PASS`, exit `0`. It stores one atomic checksum-protected hour at a time. A source hash or coordinate mismatch fails closed. Do not edit/delete the immutable old outputs or staging state.

## Post-publication independent same-method audit

```bash
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$SCRIPT" \
  --mode verify --repo-root "$PWD" \
  2>&1 | tee logs/phase7_civ_verify_20261019.log
echo "Verify exit: $?"
OUT=ttk_runs_fixed/topology_finetuning/phase7_civ_dev_gulf_inland_feb2010_locality168_20261019
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/locality_report.md"
```

Require `[VERIFY-GATE] PASS`, exit 0, and all **five** checksums `OK`. Results should not be interpreted before that gate completes.

**Safety and provenance:** The code SHA-pins Part CIII, XCIX, original XCIV and its compatibility wrapper, original Phase5E solver, and recomputes/validates CIII numerical and XCIX source lineage. Does not modify an existing model, PD, source data file or source code. Verification uses the same Phase5E algorithm, not independently implemented numerical matching. The original pretrained PhIRE WTK exposure remains unverified.
