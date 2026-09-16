#!/usr/bin/env bash
set -euo pipefail
PHIRE="${PHIRE:-$HOME/PhIRE}"
JT_PY="${JT_PY:-$HOME/micromamba/envs/join-tree-audit/bin/python}"
OUT="${1:-$PHIRE/join_tree_harness/phase3e_exact_vti}"
SCAL="$OUT/exact_scalars"
CMP="$OUT/compare"
mkdir -p "$SCAL" "$CMP"

echo "===== STEP 1: EXTRACT EXACT VTI SCALARS ====="
PYTHONNOUSERSITE=1 /usr/bin/python3 \
  "$PHIRE/join_tree_harness/phase3e_extract_exact_vti_scalars.py" \
  --phire "$PHIRE" \
  --out "$SCAL" \
  2>&1 | tee "$OUT/step1_extract.log"

echo
echo "===== STEP 2: REBUILD + COMPARE USING EXACT TTK INPUT SCALARS ====="
PYTHONNOUSERSITE=1 "$JT_PY" \
  "$PHIRE/join_tree_harness/phase3e_compare_exact_vti.py" \
  --repo "$PHIRE/third_party/tda-toolkit-mapper" \
  --exact-scalars "$SCAL" \
  --ttk-tree-extract "$PHIRE/join_tree_harness/phase3d_edge_overlap/ttk_tree_extract" \
  --out "$CMP" \
  2>&1 | tee "$OUT/step2_compare.log"

echo
echo "===== FREEZE WHOLE PHASE 3E ====="
find "$OUT" -type f ! -name phase3e_all_sha256.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$OUT/phase3e_all_sha256.txt"
cat "$OUT/phase3e_all_sha256.txt"
