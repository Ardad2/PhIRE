# Phase 7 / XCIV — Canonical spatial-locality pilot (development only)

Prepared after live Spark **XCIII PASS** (24 corrected GUDHI comparisons, five historical Phase6 sample-0 regressions, all three output hashes). This workpack has **not run on real Spark data**.

## Scope and rules

* Use only four XC/XCII development fields, all seven sources (GT + CNN/GAN/gradient/CV/PERS/F1), **top-left 160×160** scalar-speed support.
* Run the original **unaltered** `spatial_pd/phase5e_sparse_locality_solver.py`, pinned by SHA256 `47c73decca4ba151fd8b250cae2911b4c2307471fe9c881fc42bfc968e211c3a` and the archived Phase6 LXXXIII result manifest.
* Explicitly validate the original VTK `read_features` birth/death critical coordinates against all 28 **numerically corrected** XCIII-source diagrams, including H0/H1 point sets and finite coordinate bounds. **Do not try to repair/skip failing data or replace numerical persistence points with plot positions** if the VTK XML reader cannot decode them.
* Frozen locality radii `[4,8,16,32,64,96,128,inf]` in **native grid cells**; compare infinite-radius solver W22 to all 24 independently calculated corrected XCIII GUDHI W22 values. Compare diagonal cost identity and radius monotonicity.
* Output 192 aggregate rows and 384 dimension rows, GT and SR persistence-mass coverage, original metric decompositions and descriptive four-field summary. Not a model-selection/generalization study. No locked test data, HSDS, TensorFlow, training or VTI extraction.
* Outputs are staged and published once to `ttk_runs_fixed/topology_finetuning/phase7_xciv_dev_locality_20261017`; no overwrites, no mutation of past runs or XCI.
* If source, VTK coordinate-readback, GUDHI infinity or SHA gates fail, stop and share log. Do not install libraries globally to bypass errors.

## Step 1: install and review notes patch

```bash
cd ~/PhIRE
unzip Phase7_XCIV_Development_Spatial_Locality_20261017.zip -d phase7_xciv_update
( cd phase7_xciv_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_xciv_update/scripts/*.py
PATCH=phase7_xciv_update/patches/phase7_notes_XCIII_to_XCIV.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part ' dataset_generation_and_repair_notes.md | tail -3
```

Expected final notes heading: `# Part XCIV`. Archive is **flat**, so extract exactly once with `-d phase7_xciv_update`. Do not apply the patch if master notes no longer end at XCIII or have local conflicting changes. Inspect and reconcile differences instead of forcing.

## Step 2: inexpensive self-test and read-only coordinate compatibility preflight

Use the same Spark micromamba `gudhi-audit` environment that ran XCIII.

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
SCRIPT=phase7_xciv_update/scripts/phase7_xciv_dev_locality.py
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$SCRIPT" \
  --mode selftest --repo-root "$PWD"
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$SCRIPT" \
  --mode preflight --repo-root "$PWD" \
  2>&1 | tee logs/phase7_xciv_preflight_20261017.log
echo "Preflight exit: $?"
```

**Must** see `[SOLVER] PASS`, four `[COORDINATES] PASS` messages, `[PREFLIGHT] PASS`, exit `0`. The preflight loads no external-test model outputs and writes no experimental artifacts. It also verifies the Phase6 LXXXIII file checksums, original source hashes and the XCIII numeric manifest.

**Important:** The TTK Docker XML reader failed during XCII. The working `gudhi-audit` environment may or may not recover the spatial VTK attributes correctly. If a `[COORDINATES]` check fails, do **not** run compute. We need a separately audited numeric coordinate reader; do not silently use the positionless XCIII diagram arrays.

## Step 3: only after preflight passes, compute locality

The sparse solver can take some time; use tmux if desired.

```bash
cd ~/PhIRE
set -o pipefail
SCRIPT=phase7_xciv_update/scripts/phase7_xciv_dev_locality.py
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$SCRIPT" \
  --mode compute --repo-root "$PWD" \
  2>&1 | tee logs/phase7_xciv_compute_20261017.log
echo "Compute exit: $?"
```

Require 24 `[LOCALITY]` messages, 24 GUDHI/Phase5E infinity-radius matches, `[LOCALITY-GATE] PASS` and exit 0. If the stage fails, it will not publish a completed directory.

## Step 4: independent readback and full numeric recomputation with same original solver

```bash
cd ~/PhIRE
set -o pipefail
SCRIPT=phase7_xciv_update/scripts/phase7_xciv_dev_locality.py
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$SCRIPT" \
  --mode verify --repo-root "$PWD" \
  2>&1 | tee logs/phase7_xciv_verify_20261017.log
echo "Verify exit: $?"
OUT=ttk_runs_fixed/topology_finetuning/phase7_xciv_dev_locality_20261017
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/locality_report.md"
```

Require `[VERIFY-GATE] PASS`, exit 0, four `OK` checksum records. Verification recomputes the 192 locality assignments with the **same** trusted algorithm and compares each row. This is a same-implementation reproducibility audit, not an independent solution algorithm.

**Send back the preflight output before a failed compute; for successful runs share `locality_report.md` and exit codes.** No conclusion about the held-out Upper Midwest/Pacific inland fields should be made from this four-development-field pilot.
