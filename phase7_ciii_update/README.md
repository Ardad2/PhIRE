# Phase 7 Part CIII — corrected GUDHI full-week development numerics

Scope: the already verified 168 consecutive UTC hours in `gulf_inland`, 2010-Feb-01..07, and the six unchanged original PhIRE checkpoints. This is **one development region-week**, not independent hourly tests, an external geographic benchmark, or the XCI full protocol.

## Frozen inputs and validation

- CII 1,176 canonical VTI/port-0 VTU pairs, all 2,354 SHA256 entries, all numerical VTI readbacks and structural VTUs, 14 bitwise XCVII anchor pairs. CII script pinned SHA `f0e5ae86874c7e4d853320ce0bdc45b2a4fbfec3efc9b7ce6c7630edbdc39f2d`.
- XCVIII 112 decoded PDs, 96 corrected GUDHI comparisons, its four checksums, and historical XCIII/Phase6 GUDHI regressions. XCVIII script pinned SHA `5f2522cf5fb170138634bae21643dc32f9404f7ed9efad226a467fc5c0db9ff4`.
- XCI frozen crop: scalar speed in the **top-left 160×160**, NOT the entire 500×500. No new conversion, TTK, TF, WTK/HSDS reads, checkpoints or locality calculations.
- Mandatory numerical anchors: full-week hours 0 and 84 ↔ XCVIII gulf-inland indices 0 and 1, all six models, distances **and feature counts** and source diagram SHA; 12 combinations must reproduce.
- Reuses unchanged Phase6 decoder, GUDHI bottleneck, W2inf and W22 distance functions. The independent verifier recomputes using the *same* algorithm, not a separate implementation.

## Install (flat ZIP contents)

```bash
cd ~/PhIRE
unzip Phase7_CIII_Development_Fullweek_Corrected_PD_20261019.zip -d phase7_ciii_update
( cd phase7_ciii_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_ciii_update/scripts/*.py
PATCH=phase7_ciii_update/patches/phase7_notes_CII_to_CIII.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part ' dataset_generation_and_repair_notes.md | tail -3
```

## Selftest and source preflight

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
SCRIPT=phase7_ciii_update/scripts/phase7_ciii_dev_fullweek_numeric_pd.py
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$SCRIPT" --mode selftest --repo-root "$PWD"
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$SCRIPT" --mode preflight --repo-root "$PWD" \
  2>&1 | tee logs/phase7_ciii_preflight_20261019.log
echo "Preflight exit: $?"
```

**STOP** on any failed selftest, SHA, source/verifier gate, original Phase6 regression, 2010 boundary or preflight. Source preflight is **read-only**, though it may take some minutes since it verifies historical/112/1176 numerical/structural source chains.

## First three hours (smoke test)

```bash
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$SCRIPT" \
  --mode compute --repo-root "$PWD" --max-new-hours 3 \
  2>&1 | tee logs/phase7_ciii_first3_20261019.log
echo "Smoke exit: $?"
```

Require `[PARTIAL-GATE] PASS: 3/168` with exit 0. Hour 0 must pass the original XCVIII numerical anchor. The stage stores one atomically written, SHA-sealed six-model result per hour and rejects unexpected/mismatched contents on resume.

## Resume remaining hours

```bash
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$SCRIPT" \
  --mode compute --repo-root "$PWD" \
  2>&1 | tee logs/phase7_ciii_compute_20261019.log
echo "Compute exit: $?"
```

Prefer `tmux`. Require `[RESUME] PASS: 3/168`, 165 more `[HOUR-NUMERIC]` records, and `[COMPUTE-GATE] PASS` at exit 0. New immutable output:
`ttk_runs_fixed/topology_finetuning/phase7_ciii_dev_gulf_inland_feb2010_corrected_pd168_20261019/`.

## Independently verify

```bash
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$SCRIPT" \
  --mode verify --repo-root "$PWD" \
  2>&1 | tee logs/phase7_ciii_verify_20261019.log
echo "Verify exit: $?"
OUT=ttk_runs_fixed/topology_finetuning/phase7_ciii_dev_gulf_inland_feb2010_corrected_pd168_20261019
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/numeric_report.md"
```

Require `[VERIFY-GATE] PASS`, exit 0, **four checksum OK entries**. Output includes 1,008 per-hour score rows, 42 day-by-model summary rows, immutable manifest/report and SHA256 ledger. Hour 0/84 and historical 2007 regressions are enforced again. No topology interpretation before verification.

### Caveats

The archived 16-field experiment had two hours from each of eight region-weeks; this full-week experiment has 168 time-correlated hours from *one* development region-week. Descriptive hourly or day means are not independent geographic replicates. Do not select models based on locked-test outputs (none are touched). The original PhIRE pretrained model's geographic/temporal training exposure is unknown. Phase5E spatial locality remains a separate future gate.
