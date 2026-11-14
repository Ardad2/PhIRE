# Phase 7 Part XCIX — Spatial locality of 16 development fields

**Scope:** 16 existing development snapshots from Gulf inland and Southwest borderland, each already numerically evaluated in XCVIII. **No external locked-test inference or region access**. All six original model checkpoints, canonical crop, radii, XCI protocol and optimizer remain unchanged.

## Package layout

Flat ZIP: `README.md`, `xcix_append.md`, `scripts/phase7_xcix_dev_seasonal16_locality.py`, `patches/phase7_notes_XCVIII_to_XCIX.patch`, `SHA256SUMS`. Extract with `unzip ... -d phase7_xcix_update`, no duplicate nested directory.

## Install and document (once)

```bash
cd ~/PhIRE
unzip Phase7_XCIX_Development_Seasonal16_Spatial_Locality_20261018.zip -d phase7_xcix_update
( cd phase7_xcix_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_xcix_update/scripts/*.py
PATCH=phase7_xcix_update/patches/phase7_notes_XCVIII_to_XCIX.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part ' dataset_generation_and_repair_notes.md | tail -3
```

## Tests and preflight

Use the existing `gudhi-audit` micromamba environment. The original solver imports VTK, SciPy and GUDHI. Source hashes enforce immutability.

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
SCRIPT=phase7_xcix_update/scripts/phase7_xcix_dev_seasonal16_locality.py
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$SCRIPT" --mode selftest --repo-root "$PWD"
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$SCRIPT" --mode preflight --repo-root "$PWD" \
    2>&1 | tee logs/phase7_xcix_preflight_20261018.log
echo "Preflight exit: $?"
```

**Gate:** `[SELFTEST] PASS`, 16 `[COORDINATES] PASS`, `[PREFLIGHT] PASS`, and exit 0. The preflight checks 112 diagram feature inventories; no experiment output is written. Stop if anything fails.

## Compute (may be costly; tmux recommended)

```bash
cd ~/PhIRE
set -o pipefail
SCRIPT=phase7_xcix_update/scripts/phase7_xcix_dev_seasonal16_locality.py
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$SCRIPT" --mode compute --repo-root "$PWD" \
    2>&1 | tee logs/phase7_xcix_compute_20261018.log
echo "Compute exit: $?"
```

**Gate:** 96 `[LOCALITY]` checks plus `[OLD-4] PASS`, `[LOCALITY-GATE] PASS`, exit 0. All 96 infinity-radius Wasserstein distances must reproduce the corrected XCVIII scores, and original four-field XCIV output must reproduce **all** 192 aggregate and 384 dimension rows.

## Independently verify (recomputes all matching)

```bash
cd ~/PhIRE
set -o pipefail
SCRIPT=phase7_xcix_update/scripts/phase7_xcix_dev_seasonal16_locality.py
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$SCRIPT" --mode verify --repo-root "$PWD" \
    2>&1 | tee logs/phase7_xcix_verify_20261018.log
echo "Verify exit: $?"
OUT=ttk_runs_fixed/topology_finetuning/phase7_xcix_dev_seasonal16_locality_20261018
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/locality_report.md"
```

**Gate:** `[VERIFY-GATE] PASS`, exit 0, all five output SHA256 entries `OK`. No experiment output overwrites. This verifier repeats the **same validated optimizer**, not an independent algorithm.

## Caveats

The radii are WTK grid cells, not km. Persistence-mass coverage is *not* placement accuracy. Only two hours (0 and 84) per region-week were sampled; these do not establish robust 168-hour seasonal or geographic generalization. The four original fields map to indices 0, 4, 8, 12. Original pretrained PhIRE geographic membership remains unknown. The original TopoInVis archive remains unmodified.
