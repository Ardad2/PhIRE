#!/usr/bin/env bash
set -euo pipefail

PHIRE="${PHIRE:-$HOME/PhIRE}"
JT_PY="${JT_PY:-$HOME/micromamba/envs/join-tree-audit/bin/python}"
OUT="${1:-$PHIRE/join_tree_harness/phase4b_full168}"
EXTRACT="$OUT/extract"
COMPARE="$OUT/compare"
mkdir -p "$EXTRACT" "$COMPARE"

echo "===== PHASE 4B / STEP 1: EXTRACT ALL 1008 EXACT TTK INPUTS ====="
PYTHONNOUSERSITE=1 /usr/bin/python3 \
  "$PHIRE/join_tree_harness/phase4b_extract_full168.py" \
  --phire "$PHIRE" \
  --out "$EXTRACT" \
  --start 0 \
  --stop 168 \
  2>&1 | tee "$OUT/step1_extract.log"

echo
echo "===== PHASE 4B / STEP 2: FULL 6_anti CONSTRUCTION-PARITY SWEEP ====="
PYTHONNOUSERSITE=1 "$JT_PY" \
  "$PHIRE/join_tree_harness/phase4b_compare_full168.py" \
  --repo "$PHIRE/third_party/tda-toolkit-mapper" \
  --extract "$EXTRACT" \
  --out "$COMPARE" \
  2>&1 | tee "$OUT/step2_compare.log"

echo
echo "===== FREEZE WHOLE PHASE 4B ====="
find "$OUT" -type f ! -name phase4b_all_sha256.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$OUT/phase4b_all_sha256.txt"
cat "$OUT/phase4b_all_sha256.txt"
