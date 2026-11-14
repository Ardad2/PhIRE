# Phase 7 — XCII VTK XML readback compatibility hotfix (October 16, 2026)

**Prepared and tested locally, not yet validated against Spark's live XCII TTK outputs.**

## Diagnosis

The original Part XCII `selftest` fails using Docker's `vtkXMLImageDataReader`, and `extract` fails at the **first** persistence-diagram file when Docker's `vtkXMLUnstructuredGridReader` rejects an XML token. The original XCII `preflight` passed; `verify` failed only because extraction did not create its completed output directory. This is consistent with the earlier Phase 6 LXXVIII reader incompatibility, but it does **not** prove the current VTU numerical content is valid. If the standard-library XML parser rejects the current files, stop and report the failure.

The XCII driver at `phase7_xcii_update/scripts/phase7_xcii_dev_canonical_pd.py` is pinned at SHA-256 `743c7642611b6f8281891ffece8b60a6b717a0047042d745daa75cd42caee53b`. The compatibility launcher will not proceed if its hash differs.

## Precisely what changes

Without modifying the original driver on disk, the launcher imports it and swaps in three functions **in memory**:

- `check_vti`: independently parses converter-generated **ASCII Float32** VTI using standard-library XML, reads every 160×160 speed value and compares exact row-major `(y,x)` values to source float32 speed. It checks WholeExtent, Piece Extent, array name, dtype, and rejects nonfinite, altered, or transposed arrays.
- `check_pd`: parses TTK port-0 VTU using standard-library XML and verifies positive geometry counts, required `Birth`, `Persistence`, `PairIdentifier`, `PairType` arrays, and base64 framing for an appended payload. **It does not decompress, interpret or numerically validate the diagram arrays**. This is the same distinction used in Phase 6 LXXVIII; corrected numerical GUDHI decoding is required next.
- `selftest`: exercises the ASCII VTI path and checks deliberate numerical/orientation/XML tampering in place of the failing VTK-reader-dependent self-test.

An additional `compat_launcher_sha256` key is included in the original driver's output provenance dictionary. Its original script SHA, XCI/ XC source gates, original VTI converter, TTK command/options, four-field restricted scope, source hashes, report/checksum checks, atomic staging behavior, and collision rejection remain unchanged. No model predictions or original research artifacts are changed.

## Transfer + install (flat ZIP contents)

Transfer `Phase7_XCII_VTK_XML_Compat_20261016.zip` to `~/PhIRE`; then:

```bash
cd ~/PhIRE
unzip Phase7_XCII_VTK_XML_Compat_20261016.zip -d phase7_xcii_xml_compat_20261016
( cd phase7_xcii_xml_compat_20261016 && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_xcii_xml_compat_20261016/scripts/phase7_xcii_vtk_xml_compat.py
sha256sum phase7_xcii_update/scripts/phase7_xcii_dev_canonical_pd.py
```

Do not reapply any Part XCII notes patch. This is a compatibility hotfix, not a new experiment definition.

## 1. Self-test

```bash
cd ~/PhIRE
set -o pipefail
HOTFIX=phase7_xcii_xml_compat_20261016/scripts/phase7_xcii_vtk_xml_compat.py
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work:ro" -w /work phire-ttk:latest \
  python3 "$HOTFIX" --mode selftest --repo-root /work \
  2>&1 | tee logs/phase7_xcii_xml_compat_selftest_20261016.log
printf 'Selftest exit: %s\n' "$?"
```

Require `[COMPAT-SELFTEST] PASS` and exit 0. The standalone synthetic self-test does not need real XC/XCI arrays.

## 2. Read-only preflight

```bash
cd ~/PhIRE
set -o pipefail
HOTFIX=phase7_xcii_xml_compat_20261016/scripts/phase7_xcii_vtk_xml_compat.py
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work:ro" -w /work phire-ttk:latest \
  python3 "$HOTFIX" --mode preflight --repo-root /work \
  2>&1 | tee logs/phase7_xcii_xml_compat_preflight_20261016.log
printf 'Preflight exit: %s\n' "$?"
```

Require `[SOURCE] PASS`, `[PREFLIGHT] PASS` and exit 0. If the *completed* XCII output directory unexpectedly already exists, STOP and investigate, do not overwrite it.

## 3. Actual four-field, 28-diagram extraction

```bash
cd ~/PhIRE
set -o pipefail
HOTFIX=phase7_xcii_xml_compat_20261016/scripts/phase7_xcii_vtk_xml_compat.py
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" -w /work phire-ttk:latest \
  python3 "$HOTFIX" --mode extract --repo-root /work \
  2>&1 | tee logs/phase7_xcii_xml_compat_extract_20261016.log
printf 'Extract exit: %s\n' "$?"
```

Require 28 `[PD]` records, 28 `[PD-XML] PASS` records and `[EXTRACT-GATE] PASS` with exit 0. This step executes the unchanged TTK CLI once per source image (GT and six models × four development-only fields). If standard-library XML or exact VTI value comparison fails, stop. The failure cleanup removes temporary *staging* files; no previously completed experiment directory is deleted.

## 4. Read-only verification

```bash
cd ~/PhIRE
set -o pipefail
HOTFIX=phase7_xcii_xml_compat_20261016/scripts/phase7_xcii_vtk_xml_compat.py
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work:ro" -w /work phire-ttk:latest \
  python3 "$HOTFIX" --mode verify --repo-root /work \
  2>&1 | tee logs/phase7_xcii_xml_compat_verify_20261016.log
printf 'Verify exit: %s\n' "$?"

OUT=ttk_runs_fixed/topology_finetuning/phase7_xcii_dev_canonical_pd_20261015
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/extraction_report.md"
```

Require `[VERIFY-GATE] PASS` and exit 0, with **58** checksums OK: 28 VTIs, 28 VTUs, extraction manifest and report. Verification also checks the recorded *compat launcher SHA* and the original upstream source hashes.

## Next scientific gate

**Do not treat these structural VTU checks as numeric persistence validation or publish model rankings.** Run the canonical corrected Phase 6 LXXIX GUDHI reader on all 28 outputs in the separate `gudhi-audit` environment, then calculate corrected `dB`, `W2inf`, and `W22` only after that decoder and its historical regression gates pass. Phase 5E spatial-locality remains a separate following validation step. Do not use locked Upper Midwest/Pacific fields in this pilot.
