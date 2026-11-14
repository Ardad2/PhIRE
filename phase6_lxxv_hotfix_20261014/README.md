# Phase 6 Part LXXV — GT-parity diagnostic hotfix (2026-10-14)

## Why this hotfix exists

The original Part LXXV script erroneously required the historical CNN GT `.vti` under `ttk_runs_fixed/cnn/`. The archived historical **GT PD VTU** is present and was already validated by the LXXIV GUDHI regression, but the corresponding original VTI is not necessarily stored there. As a result, all three Part LXXV diagnostic modes stopped before comparing arrays or PDs.

This hotfix does **not** change the original LXXV notes patch (already applied), candidate checkpoints, arrays, TTK outputs, or corrected-GUDHI JSON. It changes only the diagnostics' path expectations:

- Required: historical CNN GT PD `.vtu`, new gradient-only GT PD `.vtu`, new GT `.vti`, and both source `dataGT.npy` and `idx.npy` files.
- Optional: historical CNN GT `.vti`. If absent, `--mode inventory`, `--mode arrays-vti` and `--mode pd` continue; the old VTI comparison is explicitly skipped.
- Optional: `--historical-vti /known/exact/path.vti` if independently located and verified.

Source-derived context: project notes Part XLI documented mixed historical orientation conventions (historical CNN/UV speed fields transpose canonical C-order representations on tested sample 69). Hence a sample-0 PD difference should not automatically be attributed to TTK threads or to different GT data. This hotfix does **not** prove that exact cause for sample 0.

## Install on Spark

Transfer `Phase6_LXXV_GT_Parity_Hotfix_20261014.zip` to `~/PhIRE`, then:

```bash
cd ~/PhIRE
unzip Phase6_LXXV_GT_Parity_Hotfix_20261014.zip -d phase6_lxxv_hotfix_20261014
python3 -m py_compile phase6_lxxv_hotfix_20261014/scripts/phase6_s0_gt_parity_v2.py
```

**Do not reapply the original LXXV Git patch.** Your notes already have Part LXXV.

## Run (non-mutating to experimental artifacts)

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail

/usr/bin/python3 phase6_lxxv_hotfix_20261014/scripts/phase6_s0_gt_parity_v2.py \
  --mode inventory 2>&1 | tee logs/phase6_s0_gt_parity_v2_inventory_20261014.log
printf 'Inventory exit: %s\n' "$?"

docker run --rm -v "$PWD:/work:ro" -w /work phire-ttk:latest \
  python3 phase6_lxxv_hotfix_20261014/scripts/phase6_s0_gt_parity_v2.py \
  --repo-root /work --mode arrays-vti \
  2>&1 | tee logs/phase6_s0_gt_parity_v2_fields_20261014.log
printf 'Fields exit: %s\n' "$?"

PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python \
  phase6_lxxv_hotfix_20261014/scripts/phase6_s0_gt_parity_v2.py \
  --mode pd 2>&1 | tee logs/phase6_s0_gt_parity_v2_pd_20261014.log
printf 'PD exit: %s\n' "$?"
```

## Interpretation

- `[GT FIELD] PASS`: historical and new source sample-0 vectors and all benchmark indices match, and the new VTI agrees with the canonical speed calculation within `atol=1e-5`. This does *not* assert equality to an unlocated original historical VTI.
- `[GT DIAGRAM] MATCH`: the historical and new finite GT PD birth/death multisets match exactly.
- `[GT DIAGRAM] DIFFER`: their multisets differ; review reported D0/D1 counts, zero-persistence pairs, and unmatched examples. This outcome still returns status 0, because it is an observed diagnostic difference rather than a runtime failure.
- `[GT FIELD] FAIL` and nonzero exit: stop; array or new-VTI parity is not established.

Never treat a historical-CNN-vs-new-model corrected-distance comparison as strictly controlled solely because historical GUDHI regression passed. For final matched multi-model topology comparisons, prefer canonical C-order regeneration of CNN, F1 and all three new Phase 6 arms under one set of extraction settings, with frozen originals left unchanged. Phase 5E locality also requires matched orientation and critical-vertex coordinate conventions.

## Validation of this hotfix in the chat environment

- Python bytecode compilation: PASS.
- Synthetic file-tree test without historical VTI: inventory PASS, arrays/VTI PASS, GT PD difference reported without failure.
- Deliberate source-array mismatch test: correctly fails the field parity gate.
- These tests do not execute against Spark's actual data or reproduce TTK extraction.
