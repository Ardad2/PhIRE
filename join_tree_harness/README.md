# Join-tree compatibility harness

This bundle tests whether the colleague toolkit's existing hierarchical
`_build_join_tree_graph()` can operate sensibly on regular 2D scalar-grid
adjacency.

It does **not** claim independent TTK `MergeTreeDistance` validation.

## Phase 1 — synthetic semantic test

Copy the two Python files to Spark, then run:

```bash
cd ~/PhIRE
mkdir -p join_tree_harness

PYTHONNOUSERSITE=1 python3 join_tree_harness/phase1_synthetic_join_tree.py \
  --repo ~/PhIRE/third_party/tda-toolkit-mapper \
  --out ~/PhIRE/join_tree_harness/phase1_synthetic \
  2>&1 | tee ~/PhIRE/join_tree_harness/phase1_synthetic.log
```

Then:

```bash
cat ~/PhIRE/join_tree_harness/phase1_synthetic/summary.txt
ls -lh ~/PhIRE/join_tree_harness/phase1_synthetic/
```

The key diagnostics are:
- current 2D pair-graph component count
- hierarchical 4-neighbor component count
- whether the hierarchical 4-neighbor graph is a connected tree
- 4-neighbor versus 8-neighbor sensitivity

## Phase 2 — sample 69, only if Phase 1 looks sensible

Use authoritative array paths on Spark. Example:

```bash
PYTHONNOUSERSITE=1 python3 join_tree_harness/phase2_real_field.py \
  --repo ~/PhIRE/third_party/tda-toolkit-mapper \
  --sample 69 \
  --input GT=data_out_fixed/wind_mrhr_cnn/dataGT.npy \
  --input CNN=data_out_fixed/wind_mrhr_cnn/dataSR.npy \
  --out ~/PhIRE/join_tree_harness/phase2_sample69 \
  2>&1 | tee ~/PhIRE/join_tree_harness/phase2_sample69.log
```

Add reconstruction-only and F1 with extra `--input LABEL=PATH` arguments once
their authoritative array paths are confirmed.

Phase 2 is still structural only. If it looks promising, the next stage is to
align regular-grid connectivity with TTK's triangulation and compare critical
values / merge order before attempting any tree-distance parity.
