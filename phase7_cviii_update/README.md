# Phase 7 Parts CVIII + CIX + CX — remaining seven 2010 development weeks

**Purpose:** complete corrected GUDHI persistence distances and the original Phase5E locality computation for the seven development-only 2010 region-weeks, then issue an **equal-weighted eight-region-week summary** (including the already completed Gulf February reference week). Do **not** present this as 2012 replication or locked external geographic generalization.

**Verified input requirement:** CVII `[ALL-VERIFY-ALL] PASS: 7/7` exit 0; 8,232 additional structurally verified port-0 TTK diagrams, plus the unchanged prior February CII–CIV results. The user's latest Spark output confirms this CVII gate. CVI model inference and CV source provenance must also remain SHA-valid.

**Seven weeks:** Gulf inland May/August/November 2010; Southwest borderland February/May/August/November 2010. Only these two development regions are selectable. Each week consists of 168 consecutive hours, six unchanged model checkpoints and 7 TTK diagram streams (GT + six predictions). One historical snapshot at hour 0 and one at hour 84 per week must match exact XCVIII and XCIX data.

## Method & integrity

- Original **CIII corrected GUDHI** driver SHA: `59518de2eed0c5ff24e00a5e75c9889226bd1d456f252d6be7449f72afbce632`.
- Original **CIV spatial-locality** driver SHA: `6381c75d84d10e812bed3c93d53bc6d03544cd8bf5f53f515cbf2bd952cec7d5`.
- Original **CVII TTK source** driver SHA: `4f06a60878a98c639931cc048c190a219347dff90b9d4749334d59f4802ae`.
- The controller SHA-checks all three originals before executing. It loads **week-bound copies in memory** and changes only region, UTC/WTK metadata, diagram paths, output paths, and the XCVIII/XCIX hour-0/84 sample IDs (2/3 through 14/15). It does not write back to original scripts or change GUDHI decoders, optimizer, radius grid or tolerance.
- Original Phase6 corrected metrics `dB`, `W2inf`, `W22` and the original Phase5E spatial radii **4, 8, 16, 32, 64, 96, 128, infinity** WTK grid cells (not kilometres), zero-pair tolerance `1e-10`, top-left **160×160 scalar wind-speed crop** remain unchanged. No full-500 topology claim.
- **One complete six-arm hour** is sealed atomically in each independent numeric/locality staging directory. Resume checks stage digests and source hashes. Completed results are immutable, and verify mode recomputes all values using the **same** trusted algorithm, not an independently implemented numerical method.
- **Each week:** 1,008 GUDHI comparisons/4 output SHA files; 8,064 locality aggregate + 16,128 H0/H1 rows/5 output SHA files, plus 42 daily rows. All 1,008 infinite-radius locality values must match the sealed week's corrected W22; 96 aggregate and 192 dimension-level locality anchors must reproduce XCIX's two historical fields.
- **Seven weeks:** 7,056 model-to-GT corrected PD comparisons; 56,448 aggregate locality rows; 112,896 dimension-level rows. Eight-week summary contains **48** equally weighted region-week-by-model rows (8 blocks ×6 arms) and means across 8 blocks, not 1,344 independent hourly replicates.

## 1. Install

```bash
cd ~/PhIRE
unzip Phase7_CVIII_CIX_CX_2010_Development_Numerics_20261021.zip -d phase7_cviii_update
( cd phase7_cviii_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_cviii_update/scripts/*.py
PATCH=phase7_cviii_update/patches/phase7_notes_CVII_to_CX.patch
git apply --check "$PATCH" && git apply "$PATCH"
```

If `git apply --check` fails, **stop**; do not append notes twice. Do not edit the frozen source scripts.

## 2. Self-test and Gulf-May corrected-GUDHI smoke

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
SCRIPT=phase7_cviii_update/scripts/phase7_cviii_cix_2010_dev_topology.py
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python \
 "$SCRIPT" --mode selftest --repo-root "$PWD"
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python \
 "$SCRIPT" --mode preflight-numeric --repo-root "$PWD" \
 --region gulf_inland --week-start 2010-05-01T00:00:00 \
 2>&1 | tee logs/phase7_cviii_numeric_preflight_20261021.log
echo "Preflight exit: $?"
```

Require self-test + numerical `[PREFLIGHT] PASS` and exit 0. Preflight verifies *real CVII source*, XCVIII/GUDHI anchor hashes, and that no immutable numeric output exists.

```bash
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python \
 "$SCRIPT" --mode numeric --repo-root "$PWD" \
 --region gulf_inland --week-start 2010-05-01T00:00:00 --max-new-hours 2 \
 2>&1 | tee logs/phase7_cviii_numeric_smoke_20261021.log
echo "Numeric smoke exit: $?"
```

Require `[PARTIAL-GATE] PASS: 2/168` exit 0. This checks six models for real numerical hours 0 and 1, including an exact XCVIII hour-0 regression at May's historical seasonal sample ID 2. If the gate fails, **stop**, preserve staging and log, and request investigation.

## 3. Finish/verify Gulf May numerics (required before locality smoke)

```bash
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python \
 "$SCRIPT" --mode numeric --repo-root "$PWD" \
 --region gulf_inland --week-start 2010-05-01T00:00:00 \
 2>&1 | tee logs/phase7_cviii_gulf_may_numeric_20261021.log
echo "Numeric completion exit: $?"
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python \
 "$SCRIPT" --mode verify-numeric --repo-root "$PWD" \
 --region gulf_inland --week-start 2010-05-01T00:00:00 \
 2>&1 | tee logs/phase7_cviii_gulf_may_verify_20261021.log
echo "Numeric verify exit: $?"
```

Both commands must exit 0; required `[COMPUTE-GATE] PASS` and `[VERIFY-GATE] PASS`. The SHA ledger has four checked files.

## 4. Gulf-May locality smoke (must pass before unattended seven-week batch)

```bash
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python \
 "$SCRIPT" --mode preflight-locality --repo-root "$PWD" \
 --region gulf_inland --week-start 2010-05-01T00:00:00 \
 2>&1 | tee logs/phase7_cix_locality_preflight_20261021.log
echo "Locality preflight exit: $?"
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python \
 "$SCRIPT" --mode locality --repo-root "$PWD" \
 --region gulf_inland --week-start 2010-05-01T00:00:00 --max-new-hours 2 \
 2>&1 | tee logs/phase7_cix_locality_smoke_20261021.log
echo "Locality smoke exit: $?"
```

Require `[ANCHOR] PASS` hour 0/84 in preflight and `[PARTIAL-GATE] PASS: 2/168` in locality smoke, with exit 0 at each. Every numeric/locality computation is inside `gudhi-audit`, **not** `/usr/bin/python3` or Docker.

## 5. Single fail-fast unattended completion command

Only after both **real** numeric and locality smoke tests pass, launch this within tmux:

```bash
cd ~/PhIRE
bash phase7_cviii_update/scripts/run_after_smoke.sh "$PWD" \
  2>&1 | tee logs/phase7_cviii_cix_full_pipeline_20261021.log
echo "Pipeline exit: ${PIPESTATUS[0]}"
```

The shell runner has `set -Eeuo pipefail` and executes exactly:

1. `numeric-all` (resumes or verifies all seven numeric weeks),
2. `verify-numeric-all` (recompute all 7,056 scores),
3. `locality-all` (resumes or verifies all seven locality weeks),
4. `verify-locality-all` (recompute all 56,448+112,896 rows),
5. `summary` (cross-stage SHA + GUDHI infinite-radius checks and 48-block table),
6. `inventory`.

Target gates:

```text
[ALL-NUMERIC-ALL] PASS: 7/7 sealed cviii weeks
[ALL-VERIFY-NUMERIC-ALL] PASS: 7/7 sealed cviii weeks
[ALL-LOCALITY-ALL] PASS: 7/7 sealed cix weeks
[ALL-VERIFY-LOCALITY-ALL] PASS: 7/7 sealed cix weeks
[SUMMARY-GATE] PASS: eight equal-weighted 2010 development region-weeks, 48 model-block rows
[2010-DEV-GATE] PASS
```

Stop if any command exits nonzero or any gate is missing. **These gates are expected, not yet observed**. The local synthetic selftest and summary fixture do not constitute real GUDHI/locality evidence.

## 6. Read science-ready outputs

```bash
OUT=ttk_runs_fixed/topology_finetuning/phase7_cx_dev2010_eightweek_summary_20261021
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/summary_report.md"
head -12 "$OUT/per_region_week.csv"
```

Detailed immutable outputs per new week:
`ttk_runs_fixed/topology_finetuning/phase7_cviii_dev_{region}_{YYYYMMDD}_corrected_pd168_20261021/` and
`phase7_cix_dev_{region}_{YYYYMMDD}_locality168_20261021/`.

**Remaining after this:** separate frozen 2012 replication, predeclared external locked-region evaluation *only after methodological gates and 2012 audit are satisfied*, potential full-domain or displaced/split-merge targeted experiments, statistical uncertainty analysis, plots and writing. No access to held-out model scores in this package.
