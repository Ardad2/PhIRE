# Phase 6 — Part LXXXIII, full 168-sample spatial locality (prepared)

**Status:** LXXXII pilot completed on Spark with `Pilot exit: 0`, `Verify exit: 0`, all four output hashes `OK`; an accidental second attempt to run the pilot refused overwrite as designed. The **new LXXXIII full-168 sweep has NOT been run on Spark**.

This package runs the **unchanged original Phase 5E sparse-savings solver** using the installed, already validated LXXXII pilot wrapper as its trusted interface. It reuses the original 168 GT, CNN, F1 diagrams and the three new 168-sample arms; no diagram regeneration, model training or modification of Phase 5E source files occurs.

The same `gudhi-audit` environment as LXXXII is required for `--mode full`; preflight and verification use only standard-library Python, except verify also reads the existing LXXXII script without importing vtk. All 1,008 GT/SR artifact hashes are matched to LXXIX before a run. The full solver checks **840** LXXIX `W22` infinity-radius cases, **2,688** original CNN/F1 Phase 5E spatial references, plus all eight saved pilot samples, including dimensional spatial metrics. Every completed five-model sample is checkpointed atomically to resume interrupted work. The output directory is new and never overwrites frozen artifacts.

## 1. Transfer and install on Spark

```bash
cd ~/PhIRE
unzip Phase6_LXXXIII_All168_Locality_20261014.zip -d phase6_lxxxiii_update
( cd phase6_lxxxiii_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase6_lxxxiii_update/scripts/*.py
PATCH=phase6_lxxxiii_update/patches/phase6_notes_LXXXII_to_LXXXIII.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part LX' dataset_generation_and_repair_notes.md | tail -4
```

Patch should add Part LXXXIII after the already-applied LXXXII heading. If `git apply --check` fails, **stop** and inspect differences. Never use force, reject patches, or replace the master notes.

## 2. Read-only full-source preflight

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
/usr/bin/python3 phase6_lxxxiii_update/scripts/phase6_lxxxiii_all168_locality.py \
  --mode preflight 2>&1 | tee logs/phase6_lxxxiii_preflight_20261014.log
echo "Preflight exit: $?"
```

Require `[REFERENCE] PASS`, `[SOLVER] PASS`, `[PREFLIGHT] PASS`, exit status 0. This preflight checks **all** 1,008 file hashes and the saved pilot's checksums, but performs no spatial optimization.

## 3. Full 168-sample spatial-locality run

Only after the preflight passes, run on Spark in tmux. This invokes original `read_features`, `SparsePrepared.solve` and `gudhi_w22` for all five models, both homology dimensions and all eight radii:

```bash
cd ~/PhIRE
tmux new -s phase6_lxxxiii_spatial
```

Inside tmux:

```bash
cd ~/PhIRE
set -o pipefail
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase6_lxxxiii_update/scripts/phase6_lxxxiii_all168_locality.py \
  --mode full 2>&1 | tee logs/phase6_lxxxiii_full_20261014.log
echo "Full spatial exit: $?"
```

Expected if successful:

```text
[HIST-REGRESSION] PASS: 2688 CNN/F1 sample×radius comparisons
[INFINITY-GATE] PASS: 840 LXXIX exact-sample W22 comparisons
[PILOT-CONTINUITY] PASS: 320 pilot aggregate comparisons plus dimensional checks
[FULL-GATE] PASS: 13440 dimension rows, 6720 aggregate rows, isolated output with checksums
Full spatial exit: 0
```

If interrupted, **repeat exactly the same `--mode full` command**. Completed samples are read back from JSON checkpoints and validated. Changed source/script hashes block a resume. If the final output has already been finalized, the driver intentionally refuses overwriting and tells you to use `--mode verify`.

## 4. Verify outputs without recomputing matching

After a successful full run:

```bash
cd ~/PhIRE
set -o pipefail
/usr/bin/python3 phase6_lxxxiii_update/scripts/phase6_lxxxiii_all168_locality.py \
  --mode verify 2>&1 | tee logs/phase6_lxxxiii_verify_20261014.log
echo "Verify exit: $?"
OUT=ttk_runs_fixed/topology_finetuning/phase6_lxxxiii_spatial_all168_20261014
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/summary.md"
```

Expected `[VERIFY-GATE] PASS` and status 0, plus seven `OK` checksum lines. The verification checks input/source continuity, every JSON checkpoint, all result row keys, checksum integrity, original CNN/F1 Phase 5E regressions, LXXIX infinity values, the saved LXXXII pilot and macro summary recomputation. It does not recompute every sparse assignment independently.

## 5. Interpret with correct scope

The `summary.md` contains mean constrained W22 at radii 4,16,32,infinity, GT persistence-mass coverage at radius 16, and paired F1–PERS comparisons across all radii. Additional CSVs retain all dimension-level and aggregate per-field outputs, macro tables, paired costs/GT coverage and factorial effects at every radius. Lower constrained W22 is better; GT coverage is a distinct higher-is-better persistence-mass measure. Infinity W22 is the corrected Euclidean-ground `W_{2,2}` **not** the historical TTK distance nor `W_{2,∞}`.

The LXXXII eight-sample ranking reversal is preliminary until this all-168 result completes. Do not make claims about geographic km without calibration. The archived F1 training/optimizer/seed lineage remains an outstanding verification; no new Phase 6 merge-tree distances or training-seed replication are supplied.
