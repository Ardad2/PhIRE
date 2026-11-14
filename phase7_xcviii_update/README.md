# Phase 7 XCVIII — Development-only Seasonal16 Corrected Persistence Distances

**Prepared workpack, not a claim of completed numerical evaluation on Spark.**

Scope: exact 112 verified XCVII diagrams (16 2010 development fields; GT + six archived checkpoints). Reuse SHA-pinned XCIII/Phase6 GUDHI functions; 96 per-field `dB`,`W2inf`,`W22` results; 48 region-week descriptive block rows. Original XCIII 4-field scores must match on 24 comparisons. No external-test access, no TTK, no TF, no WTK, no retraining.

**Packaging fix:** This ZIP has FLAT members (`README.md`, `scripts/`, `patches/`, `SHA256SUMS`). When extracted with `unzip ... -d phase7_xcviii_update` there must be one directory level only. If `unzip` reports an existing member, stop rather than blindly replacing it.

## Step 1 — Install and update master notes

```bash
cd ~/PhIRE
unzip Phase7_XCVIII_Development_Seasonal16_Corrected_PD_20261018.zip -d phase7_xcviii_update
( cd phase7_xcviii_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_xcviii_update/scripts/*.py
PATCH=phase7_xcviii_update/patches/phase7_notes_XCVII_to_XCVIII.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part ' dataset_generation_and_repair_notes.md | tail -3
```

Expect last heading Part XCVIII. If `git apply --check` fails, check the actual note status; do not use force or repeat older patches.

## Step 2 — Selftest/preflight (no output writes)

```bash
cd ~/PhIRE; mkdir -p logs; set -o pipefail
SCRIPT=phase7_xcviii_update/scripts/phase7_xcviii_dev_seasonal16_numeric_pd.py
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$SCRIPT" \
  --mode selftest --repo-root "$PWD"
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$SCRIPT" \
  --mode preflight --repo-root "$PWD" \
  2>&1 | tee logs/phase7_xcviii_preflight_20261018.log
printf 'Preflight exit: %s\n' "$?"
```

Require `[SELFTEST] PASS`, five `[LEGACY-REGRESSION] PASS`, `[SOURCE] PASS`, `[OLD-4] PASS`, `[DEV-SCOPE] PASS`, `[PREFLIGHT] PASS`, exit 0. The old XCIII verifier also prints `[VERIFY-GATE] PASS` during source checks.

## Step 3 — Compute the real 96 corrected distances

```bash
cd ~/PhIRE; set -o pipefail
SCRIPT=phase7_xcviii_update/scripts/phase7_xcviii_dev_seasonal16_numeric_pd.py
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$SCRIPT" \
  --mode compute --repo-root "$PWD" \
  2>&1 | tee logs/phase7_xcviii_compute_20261018.log
printf 'Compute exit: %s\n' "$?"
```

Require 96 `[PD-NUMERIC]` and `[COMPUTE-GATE] PASS`, exit 0. Use tmux if this takes a while. Abort if any four-field regression differs. No output directory will be published on failure.

## Step 4 — Independently reexecute 96 distances and verify

```bash
cd ~/PhIRE; set -o pipefail
SCRIPT=phase7_xcviii_update/scripts/phase7_xcviii_dev_seasonal16_numeric_pd.py
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$SCRIPT" \
  --mode verify --repo-root "$PWD" \
  2>&1 | tee logs/phase7_xcviii_verify_20261018.log
printf 'Verify exit: %s\n' "$?"
OUT=ttk_runs_fixed/topology_finetuning/phase7_xcviii_dev_seasonal16_corrected_pd_20261018
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/numeric_report.md"
```

Require `[VERIFY-GATE] PASS`, exit 0, and **4/4 SHA256** entries. Verifier recomputes the distances using the same trusted numeric implementation, not an independent second algorithm.

## Caveats

- The `W2inf`/`W22` labels denote the archived definitions; no new metric definitions are introduced.
- 16 fields = two predetermined snapshots from each of four weeks in each of two development regions. Eight region-week groups summarize two samples apiece. This does not yield eight independent seasonal experiments or support external generalization claims.
- The primary crop is at `(x0,y0)=(0,0)` with shape 160×160. Spatial locality, distributed tiles, full 500×500 topology and MT remain separate future gates.
- Never access locked Upper Midwest/Pacific inland model predictions/scores until the frozen XCI external-evaluation prerequisites are complete. The original PhIRE pretraining geography remains unknown.
