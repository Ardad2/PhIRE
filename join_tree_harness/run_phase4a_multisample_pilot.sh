#!/usr/bin/env bash
set -euo pipefail

PHIRE="${PHIRE:-$HOME/PhIRE}"
JT_PY="${JT_PY:-$HOME/micromamba/envs/join-tree-audit/bin/python}"
OUT="${1:-$PHIRE/join_tree_harness/phase4a_multisample_pilot}"
SAMPLES="${SAMPLES:-0,24,48,69,96,120,144,167}"
EXTRACT="$OUT/extract"
COMPARE="$OUT/compare"
mkdir -p "$EXTRACT" "$COMPARE"

echo "===== PHASE 4A / STEP 1: EXTRACT EXACT TTK INPUTS ====="
PYTHONNOUSERSITE=1 /usr/bin/python3 \
  "$PHIRE/join_tree_harness/phase4a_extract_pilot.py" \
  --phire "$PHIRE" \
  --out "$EXTRACT" \
  --samples "$SAMPLES" \
  2>&1 | tee "$OUT/step1_extract.log"

echo
echo "===== PHASE 4A / STEP 2: BUILD 6_anti + TEST PARITY ====="
PYTHONNOUSERSITE=1 "$JT_PY" \
  "$PHIRE/join_tree_harness/phase4a_compare_pilot.py" \
  --repo "$PHIRE/third_party/tda-toolkit-mapper" \
  --extract "$EXTRACT" \
  --out "$COMPARE" \
  2>&1 | tee "$OUT/step2_compare.log"

echo
echo "===== FREEZE WHOLE PHASE 4A ====="
find "$OUT" -type f ! -name phase4a_all_sha256.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$OUT/phase4a_all_sha256.txt"
cat "$OUT/phase4a_all_sha256.txt"
