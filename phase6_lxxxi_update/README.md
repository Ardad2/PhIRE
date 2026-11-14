# Part LXXXI — Paired PD audit closeout and Phase 5E locality readiness

**Observed:** On Spark, the Part LXXX paired preflight, report, and manifest checks passed. The LXXXI workpack has been prepared and tested *locally*, but it has **not** been executed on Spark. This is a read-only readiness/lineage inventory; it does **not** run the sparse locality solver, change source/PD files, or close F1 training provenance.

## 1. Install and record Part LXXXI

Transfer the ZIP to the repository root on Spark:

```bash
cd ~/PhIRE
unzip Phase6_LXXXI_Locality_Readiness_20261014.zip -d phase6_lxxxi_update
( cd phase6_lxxxi_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase6_lxxxi_update/scripts/phase6_lxxxi_locality_readiness.py
PATCH=phase6_lxxxi_update/patches/phase6_notes_LXXX_to_LXXXI.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part LX' dataset_generation_and_repair_notes.md | tail -4
```

If `git apply --check` fails, **stop** and examine the local differences. This patch assumes your existing `dataset_generation_and_repair_notes.md` ends with the Part LXXX entry applied from the preceding workpack. Do not use `git reset --hard`, force the patch, or replace the master notes.

## 2. Read-only readiness gate

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
/usr/bin/python3 phase6_lxxxi_update/scripts/phase6_lxxxi_locality_readiness.py \
  --mode preflight \
  2>&1 | tee logs/phase6_lxxxi_readiness_20261014.log
printf 'LXXXI readiness exit: %s\n' "$?"
```

Expected: `[SOURCE] PASS` for frozen 840-row LXXIX CSV, `[PAIRED] PASS` for the LXXX generated manifest, canonical `[PHASE5E] PASS` (GT/CNN/F1, 168 each), `[NEW] PASS` (grad/CV/PERS, 168 each PD/VTI), `[PILOT] PASS`, and `[LXXXI] PASS ... read-only`. `[F1-LINEAGE] NOT CLOSED` is deliberate: finding old arrays/checkpoints does not prove training data/loss config.

The list of `[SOURCE-CANDIDATE]` lines identifies likely Phase 5E locality Python/shell files already on Spark with content hashes and possible CLI flags. It is **a discovery report**, not validation that one candidate is the canonical all-168 solver. If needed, rerun only the source discovery:

```bash
/usr/bin/python3 phase6_lxxxi_update/scripts/phase6_lxxxi_locality_readiness.py \
  --mode sources
```

## 3. Next stage, after examining the readiness report

Port/reuse the *existing* Phase 5E sparse-savings spatial matching implementation for `grad`, `cv`, and `pers`; retain frozen CNN/F1 reference outputs and compare five-arm sample-by-sample effects. Match the validated grid `4,8,16,32,64,96,128,infinity`, D0/D1 conventions, point coordinate/orientation mapping, and GT/SR persistence-mass coverage definitions. First use predeclared samples `0,24,48,69,96,120,144,167` and verify that the Phase 5E CNN/F1 outputs and the unconstrained corrected W22 agree. Then run all 168 fields, under a *new*, isolated output root.

**Caution:** existing GUDHI distance integrity and matching of the birth/death pairs do not by themselves validate the feature coordinates used for locality. The read-only gate is intentionally not a replacement for a pilot cross-check of critical point locations, C-order orientation and spatial admissibility.
