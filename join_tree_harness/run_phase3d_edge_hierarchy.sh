#!/usr/bin/env bash
set -euo pipefail

PHIRE="${PHIRE:-$HOME/PhIRE}"
JT_PY="${JT_PY:-$HOME/micromamba/envs/join-tree-audit/bin/python}"
OUT="${1:-$PHIRE/join_tree_harness/phase3d_edge_overlap}"
EXTRACT="$OUT/ttk_tree_extract"
COMPARE="$OUT/compare"
mkdir -p "$EXTRACT" "$COMPARE"

echo "===== STEP 1: EXTRACT TTK NODE MAP + ARC TABLES ====="
PYTHONNOUSERSITE=1 /usr/bin/python3 \
  "$PHIRE/join_tree_harness/phase3d_extract_ttk_tree.py" \
  --phire "$PHIRE" \
  --out "$EXTRACT" \
  2>&1 | tee "$OUT/step1_extract.log"

echo
echo "===== STEP 2: COMPARE EDGE / HIERARCHY STRUCTURE ====="
PYTHONNOUSERSITE=1 "$JT_PY" \
  "$PHIRE/join_tree_harness/phase3d_compare_edges.py" \
  --repo "$PHIRE/third_party/tda-toolkit-mapper" \
  --sample 69 \
  --crop 160 \
  --cnn-root "$PHIRE/data_out_fixed/wind_mrhr_cnn" \
  --uv-root "$PHIRE/data_out/wind_finetune_candidateUV_expanded2688" \
  --f1-root "$PHIRE/data_out/wind_finetune_candidateF_grad_E2_low_expanded2688" \
  --ttk-tree-extract "$EXTRACT" \
  --out "$COMPARE" \
  2>&1 | tee "$OUT/step2_compare.log"

echo
echo "===== FREEZE WHOLE PHASE 3D ====="
find "$OUT" -type f ! -name phase3d_all_sha256.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$OUT/phase3d_all_sha256.txt"
cat "$OUT/phase3d_all_sha256.txt"
