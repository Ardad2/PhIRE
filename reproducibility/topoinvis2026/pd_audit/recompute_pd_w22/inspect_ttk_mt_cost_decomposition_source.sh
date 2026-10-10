#!/usr/bin/env bash
set -euo pipefail

DIST="/usr/local/include/ttk/base/MergeTreeDistance.h"
BASE="/usr/local/include/ttk/base/MergeTreeBase.h"

for f in "$DIST" "$BASE"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 2
  fi
done

show_matches() {
  local file="$1"
  local regex="$2"
  local before="${3:-20}"
  local after="${4:-80}"

  echo
  echo "================================================================================"
  echo "FILE: $file"
  echo "PATTERN: $regex"
  echo "================================================================================"

  grep -n -A"$after" -B"$before" -E "$regex" "$file" \
    | head -1600 || true
}

echo "================================================================================"
echo "MERGE-TREE COST-DECOMPOSITION SOURCE PREFLIGHT"
echo "================================================================================"
echo "DIST=$DIST"
echo "BASE=$BASE"

show_matches \
  "$DIST" \
  'class[[:space:]]+MergeTreeDistance|public:|protected:|private:' \
  8 35

show_matches \
  "$BASE" \
  'relabelCost\(|relabelCostOnly\(|deleteCost\(|insertCost\(|computeDistance\(' \
  30 120

show_matches \
  "$DIST" \
  'treeBackTable|forestBackTable|outputMatching|computeMatching|matching' \
  35 140

show_matches \
  "$DIST" \
  'costMatrix|assignment|Assignment|auction|Hungarian|exhaustive' \
  35 140

show_matches \
  "$DIST" \
  'treeTable|forestTable|computeTree|computeForest|dynamic|equation|Equation' \
  35 140

show_matches \
  "$DIST" \
  'distanceSquaredRoot_|normalizedWasserstein_|branchDecomposition_|execute\(' \
  45 180

show_matches \
  "$BASE" \
  'distanceSquaredRoot_|normalizedWasserstein_|wassersteinPower_|keepSubtree_|useMinMaxPair_|persistenceThreshold_' \
  15 70

echo
echo "================================================================================"
echo "DONE"
echo "================================================================================"
