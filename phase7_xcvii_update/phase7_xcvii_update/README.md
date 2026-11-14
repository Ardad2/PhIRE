# Phase 7 XCVII — 16-field development canonical TTK PD extraction

**Prepared 2026-10-18. No real TTK extraction has been run by this package author.**

The previous live XCVI run already produced **96 complete 500×500 model predictions**, passed six old-XC zero-drift regressions and eleven file hashes. This package only converts frozen XCVI GT/SR arrays to canonical 160×160 speed crops and creates their TTK port-0 persistence diagrams.

## Scope, provenance and safeguards

- `16` development fields, exact saved XCVI order (Gulf inland indices 0–7, Southwest borderland 8–15), four 2010 seasonal weeks, two predeclared hours per week (0 and 84).
- Seven sources (`gt`, `cnn`, `gan`, `grad`, `cv`, `pers`, `f1`), exactly `112` VTIs and `112` PD VTUs.
- Old XCII fields 0,1,2,3 correspond exactly to XCVI indices `0,4,8,12`. **28 old VTIs and 28 old PD VTUs are copied bit-for-bit**, with hash and numerical input checks; only 84 new pairs are generated.
- The existing `scripts/convert_phire_to_vti.py` and `ttkPersistenceDiagramCmd -t 1 -i ... -a wind_speed -o ...` are unchanged. Input + orientation readback invokes the pinned original `phase7_xcii_xml_compat_20261016` parser.
- Structural XML checks on PD VTUs **do not decode/verify numerical persistence data**. XCIII-style corrected numeric decoding/GUDHI and XCIV locality comparison are follow-up stages.
- Source identity, XCI frozen protocol, original four field values, complete source SHA256SUMS and zero locked-region scope are mandatory. No overwrite, no training or new inference. Writes happen atomically in an isolated staging directory.

## Install on Spark (ZIP contains FLAT paths)

```bash
cd ~/PhIRE
unzip Phase7_XCVII_Development_Seasonal16_Canonical_PD_20261018.zip -d phase7_xcvii_update
( cd phase7_xcvii_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_xcvii_update/scripts/*.py
PATCH=phase7_xcvii_update/patches/phase7_notes_XCVI_to_XCVII.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part ' dataset_generation_and_repair_notes.md | tail -3
```

If the notes are not exactly at Part XCVI, **stop**; never force-apply or reapply historical patches.

## Run (Docker `phire-ttk:latest` as installed on Spark)

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
SCRIPT=phase7_xcvii_update/scripts/phase7_xcvii_dev_seasonal16_pd.py

docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work:ro" -w /work phire-ttk:latest \
  python3 "$SCRIPT" --mode selftest --repo-root /work

docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work:ro" -w /work phire-ttk:latest \
  python3 "$SCRIPT" --mode preflight --repo-root /work \
  2>&1 | tee logs/phase7_xcvii_preflight_20261018.log
printf 'Preflight exit: %s\n' "$?"
```

Only if preflight exits zero, perform the real TTK extraction. Use `tmux` as needed; this runs 84 single-thread TTK commands.

```bash
cd ~/PhIRE
set -o pipefail
SCRIPT=phase7_xcvii_update/scripts/phase7_xcvii_dev_seasonal16_pd.py

docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" -w /work phire-ttk:latest \
  python3 "$SCRIPT" --mode extract --repo-root /work \
  2>&1 | tee logs/phase7_xcvii_extract_20261018.log
printf 'Extraction exit: %s\n' "$?"
```

Expect `112` `[PD]` lines, `28` copied XCII pairs + `84` newly computed PDs and `[EXTRACT-GATE] PASS`.

Verify the published output read-only:

```bash
cd ~/PhIRE
set -o pipefail
SCRIPT=phase7_xcvii_update/scripts/phase7_xcvii_dev_seasonal16_pd.py

docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work:ro" -w /work phire-ttk:latest \
  python3 "$SCRIPT" --mode verify --repo-root /work \
  2>&1 | tee logs/phase7_xcvii_verify_20261018.log
printf 'Verify exit: %s\n' "$?"

OUT=ttk_runs_fixed/topology_finetuning/phase7_xcvii_dev_seasonal16_pd_20261018
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/extraction_report.md"
```

Require `[VERIFY-GATE] PASS`, exit `0`, and `226/226` checksums; no scores published by XCVII.

## After XCVII

Use the corrected original XCIII Phase 6 GUDHI numerical decoder and its archived regression tests on all 112 diagrams. Check that the original 24 XCIII per-field GT/model distances are identical for the four reused fields, then calculate 96 model-to-GT numeric comparisons for 16 fields. Only after that gate should the spatial endpoint/locality solver be extended (96×8 radii). Do not inspect locked Upper Midwest/Pacific inland model outputs during engineering choices.
