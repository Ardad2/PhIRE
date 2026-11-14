# Phase 7 CVIII/CIX/CX — additive integration hotfix (2026-10-21)

## Why

The installed controller expects `CVII_output/run_config.json`, but CVII
publishes its configuration under `CVII_output/extraction_manifest.json`,
key `run_config`. The hotfix changes this single source-binding expression in
memory after checking the unchanged original controller's SHA256. It adds
region/week and 168-hour binding checks. It does not edit the original code,
frozen numerical methods, original results, or existing staging directories.

The original unattended bash runner also used bare `micromamba`; it was not in
PATH of the child shell. This replacement runner requires the absolute
executable path supplied by `PHASE7_MICROMAMBA`.

## Install — in ~/PhIRE

```bash
unzip Phase7_CVIII_CIX_CX_Binding_Hotfix_20261021.zip -d phase7_cviii_hotfix_update
( cd phase7_cviii_hotfix_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_cviii_hotfix_update/scripts/*.py
# Do NOT reapply the original CVII->CX research-notes patch.
```

## Find binary (interactive Bash terminal)

```bash
export PHASE7_MICROMAMBA="$(type -P micromamba)"
if [ ! -x "$PHASE7_MICROMAMBA" ]; then
  type -a micromamba
  echo 'STOP: could not resolve executable; share output before continuing'
else
  "$PHASE7_MICROMAMBA" --version
fi
```

If the executable is not discoverable with `type -P`, locate the true binary
rather than guessing an absolute path. Do not change the installed environment.

## Gulf May smoke and verification (all after installing)

```bash
cd ~/PhIRE
set -o pipefail
HF=phase7_cviii_hotfix_update/scripts/phase7_cviii_cix_hotfix.py
run() { PYTHONNOUSERSITE=1 "$PHASE7_MICROMAMBA" run -n gudhi-audit python "$HF" --repo-root "$PWD" "$@"; }
run --mode selftest
run --mode preflight-numeric --region gulf_inland --week-start 2010-05-01T00:00:00
run --mode numeric --region gulf_inland --week-start 2010-05-01T00:00:00 --max-new-hours 2
```

Require `selftest`, preflight and `[PARTIAL-GATE] PASS` with exit zero.
Complete and independently verify Gulf-May numerics:

```bash
run --mode numeric --region gulf_inland --week-start 2010-05-01T00:00:00
run --mode verify-numeric --region gulf_inland --week-start 2010-05-01T00:00:00
run --mode preflight-locality --region gulf_inland --week-start 2010-05-01T00:00:00
run --mode locality --region gulf_inland --week-start 2010-05-01T00:00:00 --max-new-hours 2
```

Require the corrected GUDHI verification, XCIX anchor regressions, and the
locality `[PARTIAL-GATE] PASS: 2/168`. A nonzero exit blocks batch continuation.

## Full pipeline (after both numerical AND locality smoke gates pass)

```bash
cd ~/PhIRE
set -o pipefail
bash phase7_cviii_hotfix_update/scripts/run_after_smoke_hotfix.sh "$PWD" \
  2>&1 | tee logs/phase7_cviii_cix_hotfix_full_20261021.log
echo "Pipeline exit: ${PIPESTATUS[0]}"
```

Require `[ALL-NUMERIC-ALL]`, `[ALL-VERIFY-NUMERIC-ALL]`,
`[ALL-LOCALITY-ALL]`, `[ALL-VERIFY-LOCALITY-ALL]`,
`[SUMMARY-GATE]`, `[2010-DEV-GATE]` with PASS and exit 0.

## Integrity

- Original controller SHA256: `027fcea53c67a0123390a8082ede293f4bee0da0ac93383aa3a43612547df79f`.
- Original CIII/CIV/CVII pins remain checked at runtime by original controller.
- Only CVII metadata-binding read is adapted; frozen GUDHI, Phase5E, radii,
  topology crop and input/holdout boundaries are unchanged.
- Original controller itself remains the provenance SHA in any downstream
  `adapted_driver_sha256` field; hotfix logs an additional SHA for the patched
  in-memory version. Keep that log with the research record.
- Pre-existing numerical/locality outputs are not altered; immutable-output
  and SHA256 checks remain active.
- This is NOT proof of real Spark success. Fail fast on the first mismatch.
