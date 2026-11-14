#!/usr/bin/env bash
# Phase 7: ordered, fail-fast, development-only numerical pipeline.
set -Eeuo pipefail
cd "${1:-$HOME/PhIRE}"
SCRIPT=phase7_cviii_update/scripts/phase7_cviii_cix_2010_dev_topology.py
mkdir -p logs
[[ -f "$SCRIPT" ]] || { echo "[STOP] Missing installed controller: $SCRIPT" >&2; exit 1; }
run() {
  local mode="$1" log="logs/phase7_cviii_${1}_20261021.log"
  echo "[PIPELINE] Starting $mode at $(date -u +%FT%TZ)"
  PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python "$SCRIPT" \
    --mode "$mode" --repo-root "$PWD" 2>&1 | tee "$log"
  echo "[PIPELINE] Completed $mode at $(date -u +%FT%TZ)"
}
run numeric-all
run verify-numeric-all
run locality-all
run verify-locality-all
run summary
run inventory
echo '[2010-DEV-GATE] PASS: seven numerically and spatially verified region-weeks + immutable equal-weighted eight-week summary; 2012/locked external remain pending'
