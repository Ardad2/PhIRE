# Phase 6 — Part LXXVIII: TTK VTU reader-compatibility hotfix

**Prepared locally, not run against Spark artifacts.** The original LXXVII driver remains unmodified. The compatibility launcher imports it and replaces only its `verify_pd()` VTK-reader call **in memory** with a strict XML/VTU-header check. The original source-array validation, sample selection, output paths, manifest SHA checks, TTK extraction, and pilot marker requirements remain unchanged.

## Evidence supplied from Spark

* LXXVII preflight passed for all three new trained arms and the 168 canonical Phase 5E GT diagrams.
* Gradient-only SR sample-0 VTI and VTU regenerated successfully. Both are **byte-identical** to their LXXIII counterparts: VTI SHA-256 `26e2af52cccaca6120ea4bc009a55215ebbae716bd9eaef44d91a4e2408e2030`; PD VTU SHA-256 `79b2b685968dad0387be8b5460881d3d641272807cd08e1e155efe7b6eabc00b`, 140,589 bytes.
* `ttkPersistenceDiagramCmd` completed, but `vtkXMLUnstructuredGridReader()` inside the Docker environment refused that pre-existing output (`Error parsing XML in stream at line 29, column 13`). The prior LXXIII pilot checked file presence, not this specific reader.
* The existing independently audited canonical `read_pd()` parser/GUDHI stage read the gradient-only PD, reproduced `dB=2.58057403564`, `W2inf=15.6249480697`, and `W22=18.6702768797`, then stopped because CV and PERS diagrams had not been generated. That error is downstream of the pilot stop, not a separate CV or PERS failure.

## Important validation boundary

`[PD-XML] PASS` asserts a well-formed VTK UnstructuredGrid XML envelope, plausible positive point/cell counts, presence of `Birth`, `Persistence`, `PairIdentifier`, `PairType` arrays, and the expected appended base64 framing (if present). It **does not** claim to decode or validate the compressed numerical payload. Successful execution of the separate existing LXXVII canonical `read_pd()` / GUDHI pilot gate is **required** before scaling. After full extraction, perform a complete canonical parser/schema validation and corrected-GUDHI distance sweep for all 504 new diagrams before scientific interpretation. The existing Phase 5E spatial-coordinate schema also needs its own validation before locality matching.

If the XML-level probe itself fails on either old or new gradient VTU, **stop**. Do not silently accept failed data.

## 1. Transfer, integrity check, optional notes update

```bash
cd ~/PhIRE
unzip Phase6_LXXVIII_PD_XML_Reader_Hotfix_20261014.zip -d phase6_lxxviii_hotfix_20261014
( cd phase6_lxxviii_hotfix_20261014 && sha256sum -c SHA256SUMS )
python3 -m py_compile phase6_lxxviii_hotfix_20261014/scripts/*.py

# Optional documentation patch, only once, and only when notes end at LXXVII:
PATCH=phase6_lxxviii_hotfix_20261014/patches/phase6_notes_LXXVII_to_LXXVIII.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part LX' dataset_generation_and_repair_notes.md | tail -4
```

## 2. Read-only probe — DO THIS FIRST

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work:ro" -w /work phire-ttk:latest \
  python3 phase6_lxxviii_hotfix_20261014/scripts/phase6_lxxvii_pd_xml_compat.py \
  --repo-root /work --mode probe \
  2>&1 | tee logs/phase6_lxxviii_probe_20261014.log
printf 'Probe exit: %s\n' "$?"
```

Require `[PROBE] PASS` and exit `0`. This verifies both old and new gradient VTUs with the XML-level checker, and their VTI/VTU SHA-256 equality. If the probe fails, do not run the pilot; share its log for diagnosis.

## 3. Resume sample-0 pilot for all three SR arms

```bash
cd ~/PhIRE
set -o pipefail
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" -w /work phire-ttk:latest \
  python3 phase6_lxxviii_hotfix_20261014/scripts/phase6_lxxvii_pd_xml_compat.py \
  --repo-root /work --mode pilot --arm all \
  2>&1 | tee logs/phase6_lxxviii_pilot_20261014.log
printf 'Pilot exit: %s\n' "$?"
```

The existing gradient SR VTI/PD remains in place and must pass validation; CV/PERS sample-0 VTI and PD will be written into their original separate LXXVII output directories. The original `phase6_lxxvii_update/scripts/phase6_canonical_newarms_pd.py` is neither overwritten nor modified, so its hash in `input_manifest.json` remains valid.

## 4. Run the original independent canonical/GUDHI pilot gate

```bash
cd ~/PhIRE
set -o pipefail
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase6_lxxvii_update/scripts/phase6_pilot_pd_gudhi_gate.py \
  2>&1 | tee logs/phase6_lxxviii_pilot_gudhi_20261014.log
printf 'GUDHI pilot exit: %s\n' "$?"
```

Require `[GRAD-CONTINUITY] PASS`, `[PILOT-GATE] PASS`, and exit `0`. It must produce the original `pilot_verified.json` tied to the three actual sample-0 SR PD hashes. **Stop and share outputs here before launching the full 168-per-arm extraction.**

## 5. Only after the preceding gates pass: full extraction and inventory

Use `tmux` for the write run. Within the session:

```bash
cd ~/PhIRE
set -o pipefail
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" -w /work phire-ttk:latest \
  python3 phase6_lxxviii_hotfix_20261014/scripts/phase6_lxxvii_pd_xml_compat.py \
  --repo-root /work --mode full --arm all \
  2>&1 | tee logs/phase6_lxxviii_full_20261014.log
printf 'Full exit: %s\n' "$?"
```

To verify using the same XML compatibility check in a read-only mount:

```bash
cd ~/PhIRE
set -o pipefail
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work:ro" -w /work phire-ttk:latest \
  python3 phase6_lxxviii_hotfix_20261014/scripts/phase6_lxxvii_pd_xml_compat.py \
  --repo-root /work --mode verify --arm all \
  2>&1 | tee logs/phase6_lxxviii_verify_20261014.log
printf 'Verify exit: %s\n' "$?"
```

The target remains 504 SR VTIs and 504 SR PD port-0 VTUs, with the 168 GT files reused read-only. XML-level checking and an inventory count alone are **not** a substitute for the subsequent 504-case canonical numeric/schema validation.
