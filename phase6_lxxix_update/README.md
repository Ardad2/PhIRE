# Part LXXIX: all-168 corrected GUDHI persistence-distance sweep

**Status: prepared, not run against Spark data.** Builds on the LXXVIII successful full/verify runs, which completed 504 VTU files for grad, cv and pers.

The script does not launch TTK or modify GT, CNN/F1, model outputs or the LXXVII extraction artifacts. It imports the exact corrected GUDHI parser/distance logic used in the passing three-arm sample-0 pilot. New outputs are written only under:

`ttk_runs_fixed/topology_finetuning/phase6_lxxix_corrected_pd_gudhi_20261014/`

## 1. Transfer ZIP and install documentation update

```bash
cd ~/PhIRE
unzip Phase6_LXXIX_Corrected_PD_FullSweep_20261014.zip -d phase6_lxxix_update
( cd phase6_lxxix_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase6_lxxix_update/scripts/*.py
PATCH=phase6_lxxix_update/patches/phase6_notes_LXXVIII_to_LXXIX.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part LX' dataset_generation_and_repair_notes.md | tail -4
```

The Git patch requires that Part LXXVIII already exists. Do not use `git apply --reject`, overwrite the notes, or reset unrelated work if the check fails.

## 2. Preflight: hash every canonical input + recompute GUDHI sample 0 (read-only)

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase6_lxxix_update/scripts/phase6_corrected_pd_full_sweep.py --mode preflight \
  2>&1 | tee logs/phase6_lxxix_preflight_20261014.log
printf 'LXXIX preflight exit: %s\n' "$?"
```

Require `[SOURCE] PASS` (1,008 input VTUs), `[PILOT] PASS`, `[S0] PASS` on grad/cv/pers, `[PREFLIGHT] PASS` and exit 0. The 1,008 inputs include 168 GT and 840 SR files; the 504 new-arm files are a subset. This stage writes no output except the `tee` console log.

## 3. Full sweep in tmux (only after preflight PASS)

```bash
cd ~/PhIRE
tmux new -s phase6_lxxix_gudhi
```

Inside tmux:

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase6_lxxix_update/scripts/phase6_corrected_pd_full_sweep.py --mode sweep \
  2>&1 | tee logs/phase6_lxxix_sweep_20261014.log
printf 'LXXIX sweep exit: %s\n' "$?"
```

The driver checks all input hashes against its own immutable manifest; it writes a CSV atomically after each whole five-arm sample. Rerunning the same sweep command resumes from completed samples and refuses input-hash or script changes. If it stops, inspect the log and do not delete or overwrite the saved artifacts. Output contains `per_sample_corrected_pd.csv` (840 rows), `manifest.json`, `summary.json`, `summary.md`, and `results_sha256.txt`.

## 4. Independent post-run integrity check (read-only)

After the sweep completes successfully:

```bash
cd ~/PhIRE
set -o pipefail
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase6_lxxix_update/scripts/phase6_corrected_pd_full_sweep.py --mode verify \
  2>&1 | tee logs/phase6_lxxix_verify_20261014.log
printf 'LXXIX verification exit: %s\n' "$?"
```

Require `[AUDIT] PASS`, `[VERIFY-GATE] PASS` and exit 0. Verification checks input-file hashes, unique method/sample keys, finite distances, numerical inequalities, sample-0 GUDHI regressions and frozen CSV checksum. It does **not** recompute all 840 distances independently, so describe it as an integrity check, not a second numerical implementation.

To display the summary after those checks:

```bash
cat ttk_runs_fixed/topology_finetuning/phase6_lxxix_corrected_pd_gudhi_20261014/summary.md
```

## Scope limitations and scientific caveats

- Corrected metrics: `dB` (L-infinity bottleneck), `W2inf` (q=2, L-infinity ground), and `W22` (q=2, Euclidean ground), all dimension-combined as in the already validated pilot.
- This script does not calculate merge-tree distances or Phase5E locality-constrained matching.
- Full 2×2 F1 checkpoint provenance is to be confirmed before publication claims; Phase5E's canonical F1 PDs are reused as documented, not regenerated.
- Mean/median and signed factorial effects are descriptive on 168 successive hourly benchmark fields; no hypothesis-test p-values or independent-run error bars are generated.
- If a numerical parse or GUDHI calculation fails on any sample, the sweep stops without declaring success. Do not silently drop that sample.
