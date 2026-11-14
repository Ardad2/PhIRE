# Phase 7 XCIII — corrected GUDHI persistence distances (development only)

**Prepared, not executed on live Spark data.** Follows verified XCII VTI and VTU extraction (28/28) with structural checks. Corrects the remaining numerical-validation gap by using the exact trusted Phase 6 LXXVII parser and GUDHI functions. **No locality scores are computed in this workpack.**

## Safety and output

- Inputs: frozen XCI protocol, original XCII `pd/dev_*` TTK outputs and checksums; old Phase 6 pilot, LXXIX CSV and canonical decoder.
- Scope: 4 specified **development** fields (Gulf inland / Southwest borderland), GT + 6 frozen models, 160×160 top-left scalar-speed crop only.
- Numerics: canonical corrected Phase 6 `parse`, `distance`; metrics `dB`, `W2inf`, `W22`; uses both persistence dimensions under the historical combination rule.
- Regression: original *five* Phase 6 sample-0 CNN/grad/CV/PERS/F1 distances rechecked against SHA-pinned LXXIX archive before new computation. The original GAN has no matching LXXIX regression and is scored only on new development samples.
- Never reads model predictions from locked regions or writes into historical directories. No TTK, inference, training or HSDS. The only output is `ttk_runs_fixed/topology_finetuning/phase7_xciii_dev_corrected_pd_20261016/`, published atomically, never overwritten.
- Verification repeats all 24 GUDHI distance computations using the *same* validated implementation. This is numeric reproducibility, not independence of distance algorithms.

## 1. Install package in PhIRE root

```bash
cd ~/PhIRE
unzip Phase7_XCIII_Development_Corrected_PD_20261016.zip -d phase7_xciii_update
( cd phase7_xciii_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_xciii_update/scripts/*.py
PATCH=phase7_xciii_update/patches/phase7_notes_XCII_to_XCIII.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part ' dataset_generation_and_repair_notes.md | tail -3
```

**Expected last heading: Part XCIII.** If the patch fails, stop. Do not force/overwrite. The archive is intentionally **flat** and should be unpacked only once to the named directory.

## 2. Self-test + preflight (read-only)

Use the same micromamba environment that passed Phase 6 LXXIX.

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python \
  phase7_xciii_update/scripts/phase7_xciii_dev_corrected_pd.py \
  --mode selftest --repo-root "$PWD"
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python \
  phase7_xciii_update/scripts/phase7_xciii_dev_corrected_pd.py \
  --mode preflight --repo-root "$PWD" \
  2>&1 | tee logs/phase7_xciii_preflight_20261016.log
printf 'Preflight exit: %s\n' "$?"
```

Require `[SELFTEST] PASS`, five `[LEGACY-REGRESSION] PASS` and `[PREFLIGHT] PASS` with exit 0. **Stop** on source/parser mismatch, rather than replacing the numerical implementation.

## 3. Compute 24 development-only distances

```bash
cd ~/PhIRE
set -o pipefail
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python \
  phase7_xciii_update/scripts/phase7_xciii_dev_corrected_pd.py \
  --mode compute --repo-root "$PWD" \
  2>&1 | tee logs/phase7_xciii_compute_20261016.log
printf 'Compute exit: %s\n' "$?"
```

Require 24 `[PD-NUMERIC]` records, `[COMPUTE-GATE] PASS`, exit 0. Depending on matching cost, `tmux` is appropriate. No `--overwrite` switch exists.

## 4. Verify, hashes and report

```bash
cd ~/PhIRE
set -o pipefail
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python \
  phase7_xciii_update/scripts/phase7_xciii_dev_corrected_pd.py \
  --mode verify --repo-root "$PWD" \
  2>&1 | tee logs/phase7_xciii_verify_20261016.log
printf 'Verify exit: %s\n' "$?"
OUT=ttk_runs_fixed/topology_finetuning/phase7_xciii_dev_corrected_pd_20261016
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/numeric_report.md"
```

Require `[VERIFY-GATE] PASS` and **three** output checksum OK records. Then share the report and exit codes. **Do not treat four development fields as a generalization study** or run external-region inference yet. The next separate stage is validated Phase5E sparse locality matching after a critical-point-coordinate reader audit.
