#!/usr/bin/env bash
# =============================================================================
# prepare_release_artifacts.sh  --  TopoInVis 2026 / arXiv v1
#
# Run ONCE on Spark, from the repository root, BEFORE creating the release tag:
#
#     cd ~/PhIRE
#     bash reproducibility/topoinvis2026/prepare_release_artifacts.sh
#
# What it does (nothing is committed or pushed; nothing outside the repo and
# the staging directory is modified):
#   1. verifies the frozen corrected-PD audit archives / key CSV checksums;
#   2. copies the small, authoritative audit files (code, CSV tables, summaries,
#      environment records, checksum manifests) from the external audit
#      directory into reproducibility/topoinvis2026/pd_audit/, preserving the
#      original recompute_pd/ and recompute_pd_w22/ layout;
#   3. checks that the 2688-field fixed-pair constraints NPZ exists (it is
#      git-ignored and must be force-added);
#   4. regenerates every manuscript table and the loss-study figures from the
#      released per-field table and checks them against the paper's numbers;
#   5. stages large files as GitHub release assets in $ASSETS;
#   6. writes SHA256SUMS and prints the git commands to run next.
#
# Overrides: AUDIT, W22, ASSETS, MAXMB (per-file size cap for copies, default 25)
# =============================================================================
set -uo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "run inside the PhIRE repo"; exit 1; }
cd "$ROOT"

AUDIT="${AUDIT:-$HOME/phire_runtime_audit_20260809_221548}"
W22="${W22:-$AUDIT/recompute_pd_w22}"
DEST="reproducibility/topoinvis2026"
ASSETS="${ASSETS:-$HOME/topoinvis2026_release_assets}"
MAXMB="${MAXMB:-25}"
PY="${PY:-/usr/bin/python3}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

CONSTRAINTS_DIR="ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded2688_constraints"
CONSTRAINTS_NPZ="$CONSTRAINTS_DIR/ttk_pd_critical_pairs_gtvalues.npz"
BENCH_TFR="example_data_fixed/wind_MR-HR.tfrecord"

# Hashes recorded in dataset_generation_and_repair_notes.md
H_W22_SWEEP="71d6af21ba74729d8165c2ed3bf30dbf44a47159814434382ab6443aba7356f0"
H_W22_TAR="11f898411ecb67123488e4a871766289e7e79e30ace981975d46eeb5fcae6f7d"
H_GUDHI_TAR="dcbca5cc3f34fbe54faa809e1ca964c2662976bb624dc30dc0dbd503816ffc8d"

WARN=0
ok()   { printf '  [ok]   %s\n' "$*"; }
warn() { printf '  [WARN] %s\n' "$*"; WARN=$((WARN+1)); }
hdr()  { printf '\n== %s\n' "$*"; }
sha()  { sha256sum "$1" | awk '{print $1}'; }

# -----------------------------------------------------------------------------
hdr "1. Integrity of the frozen corrected-PD audit"
[ -d "$AUDIT" ] || { echo "AUDIT not found: $AUDIT"; exit 1; }
[ -f "$W22/w22_full_sweep.csv" ] || { echo "missing $W22/w22_full_sweep.csv"; exit 1; }

[ "$(sha "$W22/w22_full_sweep.csv")" = "$H_W22_SWEEP" ] \
  && ok "w22_full_sweep.csv matches the frozen hash" \
  || warn "w22_full_sweep.csv hash differs from the frozen value"

for pair in "w22_distance_audit_20260908.tar.gz:$H_W22_TAR" "gudhi_distance_audit_20260907.tar.gz:$H_GUDHI_TAR"; do
  f="${pair%%:*}"; h="${pair##*:}"
  if [ -f "$AUDIT/$f" ]; then
    [ "$(sha "$AUDIT/$f")" = "$h" ] && ok "$f matches the frozen hash" || warn "$f hash differs"
  else
    warn "$AUDIT/$f not found (archive will not be staged)"
  fi
done

# Re-check every per-artifact manifest that exists in the audit directories.
for m in "$AUDIT"/recompute_pd/*sha256*.txt "$W22"/*sha256*.txt; do
  [ -f "$m" ] || continue
  if sha256sum -c --quiet "$m" >/dev/null 2>&1; then ok "manifest $(basename "$m"): all OK"
  else warn "manifest $(basename "$m"): some entries failed (files may have moved since freezing)"; fi
done

[ -f "$W22/corrected_pd_mt/corrected_pd_mt_joined.csv" ] \
  && ok "corrected_pd_mt/ outputs present" \
  || { echo "missing $W22/corrected_pd_mt/ -- run scripts/analyze_corrected_pd_mt_tradeoff.py first"; exit 1; }

# -----------------------------------------------------------------------------
hdr "2. Copy small authoritative audit files into $DEST/pd_audit/"
copy_small () {   # copy_small SRC_DIR DST_DIR  (top-level text artifacts only)
  local src="$1" dst="$2"; mkdir -p "$dst"
  find "$src" -maxdepth 1 -type f \( -name '*.py' -o -name '*.csv' -o -name '*.txt' \
       -o -name '*.yml' -o -name '*.yaml' -o -name '*.md' -o -name '*.json' -o -name '*.sh' \) \
       -print0 | while IFS= read -r -d '' f; do
    local mb=$(( $(stat -c %s "$f") / 1048576 ))
    if [ "$mb" -gt "$MAXMB" ]; then
      printf '  [skip] %s (%d MB > %d MB; stage as a release asset if needed)\n' "$f" "$mb" "$MAXMB"
    else
      cp -p "$f" "$dst/"
    fi
  done
  printf '  copied %s -> %s (%s files)\n' "$src" "$dst" "$(find "$dst" -maxdepth 1 -type f | wc -l)"
}
copy_small "$AUDIT/recompute_pd"          "$DEST/pd_audit/recompute_pd"
copy_small "$W22"                         "$DEST/pd_audit/recompute_pd_w22"
copy_small "$W22/corrected_pd_mt"         "$DEST/pd_audit/recompute_pd_w22/corrected_pd_mt"

for f in canonical_pd_pilot.py; do
  [ -f "$DEST/pd_audit/recompute_pd/$f" ] && ok "PD diagram parser $f included" || warn "$f not found in $AUDIT/recompute_pd"
done
for f in w22_full_sweep.py w22_full_sweep.csv gudhi_w22_crosscheck_full.py select_near_tie_visual_cases.py; do
  [ -f "$DEST/pd_audit/recompute_pd_w22/$f" ] && ok "$f included" || warn "$f not found in $W22"
done
[ -f "$DEST/pd_audit/recompute_pd/gudhi_crosscheck_full.py" ] && ok "gudhi_crosscheck_full.py included" \
  || warn "gudhi_crosscheck_full.py not found in $AUDIT/recompute_pd"

# -----------------------------------------------------------------------------
hdr "3. Fixed-pair constraints used by every fixed-pair configuration"
if [ -f "$CONSTRAINTS_NPZ" ]; then
  ok "$CONSTRAINTS_NPZ ($(du -h "$CONSTRAINTS_NPZ" | cut -f1)) -- git-ignored (*.npz): force-add it"
else
  warn "$CONSTRAINTS_NPZ missing; regenerate with the command in docs/topoinvis_2026_reproducibility.md"
fi

# -----------------------------------------------------------------------------
hdr "4. Regenerate manuscript tables/figures from the released per-field table"
JOINED="$DEST/pd_audit/recompute_pd_w22/corrected_pd_mt/corrected_pd_mt_joined.csv"
mkdir -p "$DEST/manuscript_outputs"
if "$PY" "$DEST/manuscript_tools/make_tables.py" --joined "$JOINED" \
      --out "$DEST/manuscript_outputs/tables" | tee "$DEST/manuscript_outputs/make_tables.log"; then
  ok "every mean, median, rank, win count, factorial effect and Pareto front matches the paper"
else
  warn "table verification FAILED -- see $DEST/manuscript_outputs/make_tables.log (do not tag yet)"
fi
"$PY" "$DEST/manuscript_tools/make_figures.py" --joined "$JOINED" --out "$DEST/manuscript_outputs/figures" \
  >/dev/null 2>&1 && ok "loss-study figures regenerated" || warn "make_figures.py failed (matplotlib?)"

# -----------------------------------------------------------------------------
hdr "5. Stage release assets in $ASSETS (not committed)"
mkdir -p "$ASSETS"
for f in "$AUDIT/w22_distance_audit_20260908.tar.gz" "$AUDIT/gudhi_distance_audit_20260907.tar.gz"; do
  [ -f "$f" ] && cp -p "$f" "$ASSETS/"
done
if [ -f "$BENCH_TFR" ]; then
  cp -p "$BENCH_TFR" "$ASSETS/topoinvis2026_benchmark_wind_MR-HR.tfrecord"
else
  warn "$BENCH_TFR not found (benchmark asset not staged)"
fi
( cd "$ASSETS" && sha256sum * | grep -v SHA256SUMS > SHA256SUMS )
du -h "$ASSETS"/* | sed 's/^/  /'
echo "  GitHub limit: 2 GiB per asset."

# -----------------------------------------------------------------------------
hdr "6. Checksums and next steps"
( cd "$DEST" && git ls-files -z -- . | python3 -c '
import hashlib
from pathlib import Path
import sys

paths = sorted(set(
    p.decode("utf-8")
    for p in sys.stdin.buffer.read().split(b"\0")
    if p
))
for name in paths:
    if name == "SHA256SUMS":
        continue
    path = Path(name)
    if not path.is_file():
        raise SystemExit(f"Missing tracked file: {name}")
    print(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  ./{name}")
' > SHA256SUMS )
ok "wrote $DEST/SHA256SUMS ($(wc -l < "$DEST/SHA256SUMS") files)"
du -sh "$DEST" | sed 's/^/  size: /'

cat <<EOF

Review, then commit and tag:

  git add docs/topoinvis_2026_reproducibility.md $DEST
  git add -f $CONSTRAINTS_NPZ
  git status --short | head -50
  git commit -m "Add TopoInVis 2026 arXiv-v1 reproducibility package"
  git push origin master

Then publish the draft release with tag topoinvis-2026-arxiv-v1 targeting that
commit, and upload every file in $ASSETS (including SHA256SUMS) as assets.

Warnings: $WARN
EOF

if (( WARN > 0 )); then
  echo "[error] Release preparation has $WARN warning(s). Review and correct before tagging." >&2
  exit 1
fi
