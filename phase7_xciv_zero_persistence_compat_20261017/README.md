# Phase 7 XCIV — Zero-persistence compatibility launcher (October 17, 2026)

**Purpose.** Repair a documented representation mismatch between the position-bearing original Phase5E reader and corrected canonical XCIII persistence parser, **without altering the source XCIV script, solver, diagram files, frozen XCI protocol, or any model**.

Observed read-only diagnostic on Spark, `dev_gt_s0_speed_p160_x0_y0_pd_port_0.vtu`:

| Dimension | Canonical count | Original Phase5E retained | Canonical pairs with abs(persistence) <= 1e-10 | Retained pairs |
|---|---:|---:|---:|---|
| H0 | 1452 | 1434 | 18 | Exactly equal, max absolute difference 0 |
| H1 | 1466 | 1453 | 13 | Exactly equal, max absolute difference 0 |

The Phase5E solver's `read_features()` already discards pairs whose `abs(persistence) <= 1e-10`; the positionless canonical parser preserves those zero/near-zero diagram points. Such points lie on/very near the diagonal and do not create meaningful matching cost; strict numerical *equality* still must be demonstrated via the frozen XCIII W22 infinite-radius gate.

## Safety rules

- Requires the **unmodified** original XCIV driver SHA-256 `71168242dbac580936250122385315c153999e4d5ef27e83b4a7ad6c93da63ad`.
- Uses the frozen XCII, XCIII, XCI, checkpoint, source and Phase6 regression checks in original XCIV unchanged.
- Calls the exact original original Phase5E `read_features` to read 28 sources. For each of their 56 H0/H1 diagrams, canonical zero/near-zero pairs are removed *only from the comparison target*, with original solver tolerance. The removed count must **exactly** match the original read_features skipped-zero count. All remaining numerical values and original spatial endpoint finiteness/bounds checks are executed by original XCIV `pairs()`.
- The original `SparsePrepared` sparse solver and GUDHI infinity-radius identity remain unchanged; no changes to radii, 160×160 crop, four development fields, output CSV schemas or averaging.
- Result writes are staged and atomically published at original XCIV output path `ttk_runs_fixed/topology_finetuning/phase7_xciv_dev_locality_20261017`. Existing output path is a fatal collision, never overwritten. Manifest includes compatibility launcher/source SHA-256 and excluded-pair counts for 28 diagrams, covered by SHA256SUMS.
- Also normalizes H0/H1 count dictionary keys to strings to avoid JSON integer-key normalization errors in original XCIV verification; counts themselves are unchanged.
- The prior XCIV preflight/compute failed **before publishing**, so the expected output directory should not exist. If it exists, **stop** and inspect rather than deleting/overwriting.
- A PASS is local numerical/implementation validation for four development fields only, not held-out generalization evidence. The locked Upper Midwest/Pacific inland regions remain inaccessible.
- No notes patch: Part XCIV is already recorded. Record the hotfix resolution in next dated research entry after success.

## Install (package is flat)

```bash
cd ~/PhIRE
unzip Phase7_XCIV_Zero_Persistence_Compat_20261017.zip -d phase7_xciv_zero_persistence_compat_20261017
( cd phase7_xciv_zero_persistence_compat_20261017 && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_xciv_zero_persistence_compat_20261017/scripts/*.py
```

## Self-test and preflight (no outputs, no locked-region inputs)

```bash
cd ~/PhIRE
mkdir -p logs; set -o pipefail
COMPAT=phase7_xciv_zero_persistence_compat_20261017/scripts/phase7_xciv_zero_compat.py
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$COMPAT" --mode selftest --repo-root "$PWD"
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$COMPAT" --mode preflight --repo-root "$PWD" \
  2>&1 | tee logs/phase7_xciv_zero_compat_preflight_20261017.log
echo "Preflight exit: $?"
```

Require four `[COORDINATES] PASS`, `[PREFLIGHT] PASS` and exit `0`, as well as original source/solver/regression checks. **No computation unless the preflight passes.**

## Compute and verify (if previous step passed)

```bash
cd ~/PhIRE; set -o pipefail
COMPAT=phase7_xciv_zero_persistence_compat_20261017/scripts/phase7_xciv_zero_compat.py
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$COMPAT" --mode compute --repo-root "$PWD" \
  2>&1 | tee logs/phase7_xciv_zero_compat_compute_20261017.log
echo "Compute exit: $?"
```

Require 24 `[LOCALITY]`, `[LOCALITY-GATE] PASS`, exit `0`. Then:

```bash
cd ~/PhIRE; set -o pipefail
COMPAT=phase7_xciv_zero_persistence_compat_20261017/scripts/phase7_xciv_zero_compat.py
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$COMPAT" --mode verify --repo-root "$PWD" \
  2>&1 | tee logs/phase7_xciv_zero_compat_verify_20261017.log
echo "Verify exit: $?"
OUT=ttk_runs_fixed/topology_finetuning/phase7_xciv_dev_locality_20261017
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/locality_report.md"
```

Require `[VERIFY-GATE] PASS`, `[COMPAT-VERIFY] PASS`, exit `0`, and four SHA-256 entries `OK`.

## Testing scope of package itself

Local: Python compilation, package SHA-256 verification, identity-pair self-test, synthetic 28-diagram integration (192 aggregate/384 dimensional results), save/recompute/verify, and negative tests for tampered report, nonzero pair drift, and zero-count mismatch. **No real Spark inputs were available in the package test environment**, so live preflight, actual GUDHI infinity checks, and live model outcomes remain pending.
