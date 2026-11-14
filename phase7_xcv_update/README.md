# Phase 7 — Part XCV: 16-field 2010 development source expansion

The XCIV local development pilot is verified. This isolated next stage retrieves **only 12 new full-resolution WTK fields** for the two development regions, reuses four already-screened original NPZ previews, and checks all 16 against the frozen LXXXVII screening and XCI evaluation protocol.

Scope is strictly **data extraction**. No six-model inference, scores, new checkpoint selection, TTK, PD, locality, locked external-test model access or 2012 replication data.

The times are not invented after observing model performance: four *already frozen* 2010 seasonal seven-day windows, at UTC offset `0` and `84` h. The 16 fields are a **seasonally distributed engineering pilot**, not a replacement for the primary XCI full-hour evaluation (168 h per region-week).

## Install in ~/PhIRE

```bash
cd ~/PhIRE
unzip Phase7_XCV_Development_Seasonal16_Data_20261018.zip -d phase7_xcv_update
( cd phase7_xcv_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_xcv_update/scripts/*.py
PATCH=phase7_xcv_update/patches/phase7_notes_XCIV_to_XCV.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part ' dataset_generation_and_repair_notes.md | tail -3
```

If `git apply --check` fails, stop. Do not force/overwrite or reapply prior patches. The expected last notes heading is `# Part XCV`.

## Offline self-test and preflight

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
SCRIPT=phase7_xcv_update/scripts/phase7_xcv_dev_seasonal16.py
python3 "$SCRIPT" --mode selftest --repo-root "$PWD"
python3 "$SCRIPT" --mode preflight --repo-root "$PWD" \
  2>&1 | tee logs/phase7_xcv_preflight_20261018.log
echo "Preflight exit: $?"
```

Require `[SELFTEST] PASS`, `[SOURCES] PASS`, `[SCOPE] PASS`, `[PREFLIGHT] PASS`, exit 0. Checks pre-existing four NPZ numerical values and all frozen source digests. No HSDS requests or outputs.

## Actual WTK extraction (only after preflight success)

Use the same Python environment that successfully imported `h5pyd`/NumPy for LXXXVII and the same read-only HSDS endpoint. Consider `tmux` for network stability.

```bash
cd ~/PhIRE
export HS_ENDPOINT="https://developer.nlr.gov/api/hsds"
export HSDS_ENDPOINT="$HS_ENDPOINT"
set -o pipefail
SCRIPT=phase7_xcv_update/scripts/phase7_xcv_dev_seasonal16.py
python3 "$SCRIPT" --mode extract --repo-root "$PWD" \
  2>&1 | tee logs/phase7_xcv_extract_20261018.log
echo "Extraction exit: $?"
```

Expect 16 `[FIELD]` entries (`origin=cached` four times, `origin=HSDS` twelve times), `[EXTRACT-GATE] PASS` and exit 0. The code reuses `phase7_lxxxvii_update/scripts/phase7_wtk_wind_screen.py` for parsing and native wind conversion, refusing to run if its source hash changed.

## Independent local verification

```bash
cd ~/PhIRE
set -o pipefail
SCRIPT=phase7_xcv_update/scripts/phase7_xcv_dev_seasonal16.py
python3 "$SCRIPT" --mode verify --repo-root "$PWD" \
  2>&1 | tee logs/phase7_xcv_verify_20261018.log
echo "Verify exit: $?"
OUT=ttk_runs_fixed/topology_finetuning/phase7_xcv_development_seasonal16_20261018
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/seasonal16_report.md"
```

Require `[VERIFY-GATE] PASS`, exit 0, and **18 SHA256 OK** entries (16 NPZ + two reports). This second stage reads files locally and does **not** reconnect to WTK or start TensorFlow.

## Boundaries and checks

- Both regions and all four seasonal UTC week starts are part of the frozen XCI protocol. Offset 84 is from the original LXXXVII predeclared 64-field screening.
- Reusing four previously captured source previews does not create new model predictions or overwrite XC. All extracted fields remain separate from original and XCII/XCIV output trees.
- No script updates the research notes: the bundled patch adds a new Part XCV, leaving Parts XCIV and earlier unchanged.
- Outputs are written to a hidden staging folder and atomically published at successful completion. The script refuses to overwrite any completed directory.
- HSDS success and external pretraining novelty are **not** implied by local fixture tests. The first real command must run on Spark.
