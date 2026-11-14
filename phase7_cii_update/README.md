# Phase 7 — Part CII: 168-hour canonical persistence diagrams

**Prepared October 18, 2026. Real TTK extraction is NOT yet executed by the workpack author.**

## Verified Spark prerequisite

Part CI has independently verified all **1,008** full500 predictions for the frozen Gulf inland 2010-02-01..07 UTC week. All 13 output checksums passed and original two-hour XCVI prediction anchors reproduce exactly. The earlier XCVII canonical PD experiment also verified its 112 files, including GT and six model diagrams for this week's hours 0 and 84.

## Scope and immutable scientific choices

- **Development only**, frozen XCI: `gulf_inland`, WTK grid rectangle `(205,705,1471,1971)`, indices 27048–27215, hourly offsets 0..167. Locked `upper_midwest` and `pacific_inland` untouched.
- Same seven source order: `gt, cnn, gan, grad, cv, pers, f1`. Same scalar wind-speed crop at top-left `(x0=0,y0=0)`, **160×160**. VTK x=columns, y=rows.
- Original canonical VTI Float32 conversion and `ttkPersistenceDiagramCmd -t 1`; no changes to checkpoint weights, inference or thresholding.
- 168×7 = **1176** canonical pairs, including **14 reused bit-for-bit XCVII pairs** (168-hour positions 0 and 84 correspond to XCVII indices 0 and 1) and **1162 new single-threaded TTK runs**.
- **Only VTI numeric readback and PD VTU structure are validated here.** VTU compressed numerical persistence pairs need a separate corrected GUDHI decoder/regression gate before any topology result can be reported.

## Installation (flat ZIP)

Copy `Phase7_CII_Development_Fullweek_Canonical_PD_20261018.zip` to `~/PhIRE`, then:

```bash
cd ~/PhIRE
unzip Phase7_CII_Development_Fullweek_Canonical_PD_20261018.zip -d phase7_cii_update
( cd phase7_cii_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_cii_update/scripts/*.py
PATCH=phase7_cii_update/patches/phase7_notes_CI_to_CII.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part ' dataset_generation_and_repair_notes.md | tail -3
```

The final heading must be **Part CII**. Do not force a failed patch or edit historical entries.

## Self-test and offline, read-only preflight

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
SCRIPT=phase7_cii_update/scripts/phase7_cii_dev_fullweek_pd.py

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work:ro" -w /work phire-ttk:latest \
  python3 "$SCRIPT" --mode selftest --repo-root /work

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work:ro" -w /work phire-ttk:latest \
  python3 "$SCRIPT" --mode preflight --repo-root /work \
  2>&1 | tee logs/phase7_cii_preflight_20261018.log
printf 'Preflight exit: %s\n' "$?"
```

Require `[SELFTEST] PASS`, `[SOURCES] PASS`, `[ANCHOR] PASS`, `[PREFLIGHT] PASS`, exit **0**. Source preflight verifies 13 CI and 226 XCVII checksums, frozen source code hashes and geographic scope; no TensorFlow/HSDS/TTK calls, no output writes. Disk free safety floor 8 GiB; actual TTK runtime not predicted.

## 13-pair production smoke batch

This first call makes 7 **reused** hour-zero pairs and 6 **new** hour-one TTK pairs, then stops safely:

```bash
cd ~/PhIRE
set -o pipefail
SCRIPT=phase7_cii_update/scripts/phase7_cii_dev_fullweek_pd.py

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" -w /work phire-ttk:latest \
  python3 "$SCRIPT" --mode extract --repo-root /work --max-new-pairs 13 \
  2>&1 | tee logs/phase7_cii_first13_20261018.log
printf 'Smoke exit: %s\n' "$?"
```

Require 13 `[PD]` logs, `[PARTIAL-GATE] PASS: 13/1176`, exit **0**. If any TTK/VTU or provenance check fails, stop and share the log. A created pair directory is sealed atomically. If an interrupted TTK leaves an uncommitted `.cii_tmp_...` directory, the next attempt **fails closed**, rather than silently deleting/misclassifying that work.

## Resume to the complete set

Use a `tmux` session for this longer job. **Do not run a second extract concurrently.**

```bash
cd ~/PhIRE
set -o pipefail
SCRIPT=phase7_cii_update/scripts/phase7_cii_dev_fullweek_pd.py

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" -w /work phire-ttk:latest \
  python3 "$SCRIPT" --mode extract --repo-root /work \
  2>&1 | tee logs/phase7_cii_extract_20261018.log
printf 'Extract exit: %s\n' "$?"
```

Require `[RESUME] PASS: 13/1176`, 1176 total `[PD]` records across both runs, then `[EXTRACT-GATE] PASS` and exit 0. The original 14 XCVII pairs are copied bit-for-bit, 1162 new TTK runs complete, and output publication is atomic and overwrite-protected.

## Read-only independent verification

```bash
cd ~/PhIRE
set -o pipefail
SCRIPT=phase7_cii_update/scripts/phase7_cii_dev_fullweek_pd.py

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work:ro" -w /work phire-ttk:latest \
  python3 "$SCRIPT" --mode verify --repo-root /work \
  2>&1 | tee logs/phase7_cii_verify_20261018.log
printf 'Verify exit: %s\n' "$?"

OUT=ttk_runs_fixed/topology_finetuning/phase7_cii_dev_gulf_inland_feb2010_pd168_20261018
( cd "$OUT" && sha256sum -c SHA256SUMS ) \
  > logs/phase7_cii_sha256_20261018.log 2>&1
printf 'Checksum exit: %s\n' "$?"
tail -6 logs/phase7_cii_sha256_20261018.log
cat "$OUT/extraction_report.md"
```

Require `[VERIFY-GATE] PASS`, exit 0, 2354 checksum entries all OK. The original staged pair folders are intentionally preserved; do not delete the staging directory while assessing the result.

## Evidence boundaries and next step

These are **1176 diagrams, not 1176 independent experiments**. The week contains 168 temporally correlated hours. After CII passes, run an additive *corrected numerical* GUDHI stage to compare 168×6 = 1008 model-to-GT global distances (and ensure the two old hour anchors agree with XCVIII/XCIII); only then extend the Phase5E locality optimizer and eventually other region-week blocks. No locked-test results or scientific claims are produced here. Pretrained PhIRE geographic exposure remains unknown.
