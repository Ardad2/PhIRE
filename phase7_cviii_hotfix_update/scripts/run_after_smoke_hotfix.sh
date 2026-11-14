#!/usr/bin/env bash
# Additive fail-fast continuation for CVIII/CIX/CX. Original scripts untouched.
set -Eeuo pipefail
cd "${1:-$HOME/PhIRE}"
SCRIPT=phase7_cviii_hotfix_update/scripts/phase7_cviii_cix_hotfix.py
mkdir -p logs
[[ -f "$SCRIPT" ]] || { echo "[STOP] Missing installed hotfix: $SCRIPT" >&2; exit 1; }
[[ -n "${PHASE7_MICROMAMBA:-}" && -x "${PHASE7_MICROMAMBA:-}" ]] || {
  echo '[STOP] Set PHASE7_MICROMAMBA to the absolute, executable micromamba binary path' >&2
  exit 127
}
run() {
  local mode="$1" log="logs/phase7_cviii_hotfix_${1}_20261021.log"
  echo "[PIPELINE] Starting $mode at $(date -u +%FT%TZ)"
  PYTHONNOUSERSITE=1 "$PHASE7_MICROMAMBA" run -n gudhi-audit python "$SCRIPT" \
    --mode "$mode" --repo-root "$PWD" 2>&1 | tee "$log"
  # Exit status alone is not proof: require the named scientific gate in the log.
  local expected
  case "$mode" in
    selftest) expected='[SELFTEST] PASS:' ;;
    numeric-all) expected='[ALL-NUMERIC-ALL] PASS: 7/7' ;;
    verify-numeric-all) expected='[ALL-VERIFY-NUMERIC-ALL] PASS: 7/7' ;;
    locality-all) expected='[ALL-LOCALITY-ALL] PASS: 7/7' ;;
    verify-locality-all) expected='[ALL-VERIFY-LOCALITY-ALL] PASS: 7/7' ;;
    summary) expected='[SUMMARY-GATE] PASS:' ;;
    inventory) expected='[WEEK]' ;;
    *) echo "[STOP] Unknown mode $mode" >&2; exit 1 ;;
  esac
  if ! grep -Fq "$expected" "$log"; then
    echo "[STOP] Missing scientific gate $expected in $log" >&2
    exit 1
  fi
  if [[ "$mode" == inventory ]]; then
    [[ $(grep -Fc '[WEEK]' "$log") -eq 7 ]] || {
      echo '[STOP] Expected exactly seven inventory records' >&2; exit 1;
    }
  fi
  echo "[PIPELINE] Completed and gated $mode at $(date -u +%FT%TZ)"
}
run selftest
run numeric-all
run verify-numeric-all
run locality-all
run verify-locality-all
run summary
run inventory
echo '[2010-DEV-GATE] PASS: seven verified numerical and locality weeks + immutable equal-weight eight-week summary; no locked tests'
