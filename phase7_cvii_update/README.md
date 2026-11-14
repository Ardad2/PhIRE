# Phase 7 Part CVII — Batch canonical TTK persistence extraction for all seven remaining 2010 development weeks

## Boundaries

This ZIP requires **seven independently verified CVI inference weeks** first. The last shared CVI excerpt only showed one sealed week, not `verify-all` success. Stop if that gate fails. The original February Gulf CII output is already verified and must remain untouched.

CVI script SHA: `3cf69b42c57a9ce16b6d37ca97285b8b54ce3696cc93d0ea6141a124a491b647`.
Original CII script SHA: `f0e5ae86874c7e4d853320ce0bdc45b2a4fbfec3efc9b7ce6c7630edbdc39f2d`.

Each of seven weeks: 168h × 7 GT/model arms = 1176 canonical VTI+VTU pairs. Fourteen original XCVII pairs are bitwise reused (hours 0 and 84, **region-specific** XCVI indices); the other 1162 pairs are freshly extracted using original CII's unchanged command. Seven weeks: 8232 pairs = 98 reused + 8134 new. Separate sealed per-week outputs contain 2354 SHA256 entries and 1176 real TTK diagrams. No 2012 data; no locked external geographic evaluation. CII only validates VTI scalars and VTU structure; corrected numerical persistence decoding and spatial-locality matching are later gates.

## 0. CVI independent verification MUST pass

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
/usr/bin/python3 phase7_cvi_update/scripts/phase7_cvi_dev2010_sevenweek_inference.py \
  --mode verify-all --repo-root "$PWD" \
  2>&1 | tee logs/phase7_cvi_verify_all_20261020.log
echo "CVI verify-all exit: $?"
```

Require `[ALL-VERIFY-ALL] PASS: 7/7`, exit 0.

## 1. Install ZIP (flat layout) and notes patch

```bash
cd ~/PhIRE
unzip Phase7_CVII_Development_Seven_2010_Week_Canonical_PD_20261020.zip -d phase7_cvii_update
( cd phase7_cvii_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_cvii_update/scripts/*.py
PATCH=phase7_cvii_update/patches/phase7_notes_CVI_to_CVII.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part ' dataset_generation_and_repair_notes.md | tail -3
```

Stop if the archive/checksum/patch checks fail. If Part CVII is already applied, do not apply twice.

## 2. Offline container self-test and one-week preflight

```bash
cd ~/PhIRE
set -o pipefail
SCRIPT=phase7_cvii_update/scripts/phase7_cvii_dev2010_sevenweek_pd.py
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work:ro" -w /work phire-ttk:latest \
  python3 "$SCRIPT" --mode selftest --repo-root /work

docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work:ro" -w /work phire-ttk:latest \
  python3 "$SCRIPT" --mode preflight --repo-root /work \
  --region gulf_inland --week-start 2010-05-01T00:00:00 \
  2>&1 | tee logs/phase7_cvii_gulf_may_preflight_20261020.log
echo "CVII preflight exit: $?"
```

Expect original CI/CVI/XCVII/CII source hashes, XCVII h0/h84 anchors and `[PREFLIGHT] PASS`, exit 0.

## 3. One-week 13-pair real-TTK smoke test

```bash
cd ~/PhIRE
set -o pipefail
SCRIPT=phase7_cvii_update/scripts/phase7_cvii_dev2010_sevenweek_pd.py
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" -w /work phire-ttk:latest \
  python3 "$SCRIPT" --mode extract --repo-root /work \
  --region gulf_inland --week-start 2010-05-01T00:00:00 --max-new-pairs 13 \
  2>&1 | tee logs/phase7_cvii_gulf_may_first13_20261020.log
echo "CVII smoke exit: $?"
```

Expect `[PARTIAL-GATE] PASS: 13/1176`, exit 0. Must see seven bitwise XCVII anchor pairs at h0 and six **NEW_TTK** pairs at h1. Do not continue after any error.

## 4. Resume all seven TTK weeks in one serial batch (recommend tmux)

```bash
cd ~/PhIRE
set -o pipefail
SCRIPT=phase7_cvii_update/scripts/phase7_cvii_dev2010_sevenweek_pd.py
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" -w /work phire-ttk:latest \
  python3 "$SCRIPT" --mode extract-all --repo-root /work \
  2>&1 | tee logs/phase7_cvii_extract_all_20261020.log
echo "CVII extract-all exit: $?"
```

Require seven `[EXTRACT-GATE] PASS` and `[ALL-EXTRACT-ALL] PASS: 7/7`, exit 0. Existing staged diagram pairs are rechecked for source/hash/VTU structure and reused; already sealed weeks are verified and skipped. **Never launch two runs against the same output concurrently.**

## 5. Independently verify seven new weeks

```bash
cd ~/PhIRE
set -o pipefail
SCRIPT=phase7_cvii_update/scripts/phase7_cvii_dev2010_sevenweek_pd.py
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work:ro" -w /work phire-ttk:latest \
  python3 "$SCRIPT" --mode verify-all --repo-root /work \
  2>&1 | tee logs/phase7_cvii_verify_all_20261020.log
echo "CVII verify-all exit: $?"

docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work:ro" -w /work phire-ttk:latest \
  python3 "$SCRIPT" --mode inventory --repo-root /work
```

Require `[ALL-VERIFY-ALL] PASS: 7/7` and exit 0. The per-week CII verifier validates all 1176 GT/model input scalars, 1176 VTU structures, bitwise historical anchors and all 2354 SHA256 file entries. Independent *numerical PD decoder* and GUDHI distances **not performed** by this workpack.

### Resource/progress guidance

Each new TTK pair executes `ttkPersistenceDiagramCmd -t 1` with the frozen top-left 160×160 scalar-speed crop. There are 8134 new command invocations. That stage is likely slower than the already-completed six-model inference, but no reliable wall clock has been established for all seven weeks. If interrupted, reissue the **same command**; don't delete `.staging` directories. Stop immediately if hashes, anchor parity or structure validation fail. If an individual week is sealed, it's safe to proceed to a separately gated numerical stage for that week after its verification—do not imply the other six are complete.

The scientific eight-block aggregation must weight each 2010 development region-week equally; the 168 hours and 7 days are temporally correlated. Keep 2012 replication and locked external evaluations separate.
