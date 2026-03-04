#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Defaults
# -----------------------------
NUM_SAMPLES=25
START_IDX=0
PATCH=160
X0=0
Y0=0
THREADS=20
METHODS=("gan" "cnn")
OUT_ROOT="ttk_runs"

DO_BUILD=1
DO_INSTALL=1
DO_INFER=1
DO_TOPO=1
DO_PHYSICS=1
DO_PLOTS=1

DO_GIT_PUSH=0
GIT_MSG="Update combined topology+physics results"

# -----------------------------
# Helpers
# -----------------------------
die() { echo "[ERR] $*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

timestamp() { date +"%Y%m%d-%H%M%S"; }

retry() {
  # retry N cmd...
  local n="$1"; shift
  local i=1
  until "$@"; do
    if [[ $i -ge $n ]]; then
      return 1
    fi
    echo "[WARN] command failed (attempt $i/$n): $*"
    i=$((i+1))
    sleep 3
  done
}

repo_root() {
  # Resolve repo root from this script location
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  (cd "$here/.." && pwd)
}

latest_wind_outdir() {
  # newest data_out/wind-* directory by mtime
  ls -dt data_out/wind-* 2>/dev/null | head -n 1
}

ensure_run_scripts() {
  # Ensure run_paired scripts exist at repo root; restore from archive if not.
  if [[ ! -f run_paired_wind_mr_hr.py ]]; then
    if [[ -f archive/phase_a/run_paired_wind_mr_hr.py ]]; then
      cp archive/phase_a/run_paired_wind_mr_hr.py .
      echo "[OK] Restored run_paired_wind_mr_hr.py from archive/phase_a/"
    else
      die "Missing run_paired_wind_mr_hr.py and archive copy not found."
    fi
  fi

  if [[ ! -f run_paired_wind_mr_hr_cnn.py ]]; then
    if [[ -f archive/phase_a/run_paired_wind_mr_hr_cnn.py ]]; then
      cp archive/phase_a/run_paired_wind_mr_hr_cnn.py .
      echo "[OK] Restored run_paired_wind_mr_hr_cnn.py from archive/phase_a/"
    else
      die "Missing run_paired_wind_mr_hr_cnn.py and archive copy not found."
    fi
  fi
}

ensure_tf1_compat() {
  # Auto-patch TF2→TF1 compat if needed (safe, makes timestamped backups).
  local need_patch=0
  grep -q "import tensorflow.compat.v1 as tf" PhIREGANs.py || need_patch=1
  grep -q "import tensorflow.compat.v1 as tf" utils.py || need_patch=1
  grep -q "import tensorflow.compat.v1 as tf" sr_network.py || need_patch=1

  if [[ $need_patch -eq 0 ]]; then
    echo "[OK] TF1 compat already present."
    return 0
  fi

  echo "[WARN] TF1 compat not detected. Auto-patching PhIREGANs.py / utils.py / sr_network.py ..."
  local ts; ts="$(timestamp)"
  cp PhIREGANs.py "PhIREGANs.py.bak_${ts}"
  cp utils.py "utils.py.bak_${ts}"
  cp sr_network.py "sr_network.py.bak_${ts}"

  python3 - <<'PY'
from pathlib import Path
files = ["PhIREGANs.py", "sr_network.py", "utils.py"]
for fn in files:
    p = Path(fn)
    text = p.read_text()

    # TF1 compat import
    if "import tensorflow.compat.v1 as tf" not in text:
        text = text.replace(
            "import tensorflow as tf",
            "import tensorflow.compat.v1 as tf\n"
            "tf.disable_v2_behavior()"
        )

    # contrib initializer -> glorot
    text = text.replace(
        "tf.contrib.layers.xavier_initializer()",
        "tf.glorot_uniform_initializer()"
    )

    # numpy tostring -> tobytes
    text = text.replace(".tostring()", ".tobytes()")

    p.write_text(text)
    print("Patched", fn)
PY
  echo "[OK] TF1 compat patch applied."
}

copy_outputs_to_canonical() {
  local src="$1"
  local dst="$2"
  [[ -d "$src" ]] || die "Missing source outdir: $src"

  rm -rf "$dst"
  mkdir -p "$dst"

  # Copy expected files
  for f in dataGT.npy dataIN.npy dataSR.npy idx.npy; do
    [[ -f "$src/$f" ]] || die "Missing $src/$f"
    cp "$src/$f" "$dst/"
  done
}

verify_counts() {
  python3 - <<'PY'
import numpy as np
for m in ["gan","cnn"]:
    gt = np.load(f"data_out/wind_mrhr_{m}/dataGT.npy", mmap_mode="r")
    inp = np.load(f"data_out/wind_mrhr_{m}/dataIN.npy", mmap_mode="r")
    sr  = np.load(f"data_out/wind_mrhr_{m}/dataSR.npy", mmap_mode="r")
    print(f"{m}: GT={gt.shape} IN={inp.shape} SR={sr.shape}")
PY
}

verify_gan_cnn_differ() {
  python3 - <<'PY'
import numpy as np
gan = np.load("data_out/wind_mrhr_gan/dataSR.npy", mmap_mode="r")
cnn = np.load("data_out/wind_mrhr_cnn/dataSR.npy", mmap_mode="r")
diff = float(np.abs(gan[:5]-cnn[:5]).mean())
mx = float(np.abs(gan[:5]-cnn[:5]).max())
print("mean|GAN-CNN| (first 5):", diff)
print("max |GAN-CNN| (first 5):", mx)
if diff == 0.0 and mx == 0.0:
    raise SystemExit("ERROR: GAN and CNN SR outputs are identical. Fix output plumbing/checkpoints.")
PY
}

summarize_merged() {
  python3 - <<'PY'
import pandas as pd
df = pd.read_csv("ttk_runs/combined/psnr_topology_physics_merged.csv")
print(df["method"].value_counts())
print(df.groupby("method")[["psnr","pd_distance","mt_distance"]].mean())
PY
}

usage() {
  cat <<USAGE
Usage:
  bash scripts/run_full_experiment.sh [options]

Options:
  --num N            Number of samples to analyze (default: ${NUM_SAMPLES})
  --start I          Start sample index (default: ${START_IDX})
  --patch P          Patch size for scalar crop (default: ${PATCH})
  --x0 X             Patch x0 (default: ${X0})
  --y0 Y             Patch y0 (default: ${Y0})
  --threads T        Threads for TTK (default: ${THREADS})
  --out-root DIR     Output root for topology runs (default: ${OUT_ROOT})

  --skip-build       Skip TFRecord build
  --skip-install     Skip installing TFRecords into example_data/
  --skip-infer       Skip GAN/CNN inference step
  --skip-topo        Skip topology pipeline
  --skip-physics     Skip analysis_compare.py
  --skip-plots       Skip plot_topology_physics_analysis.py

  --git-push         Stage+commit+push ONLY ttk_runs/combined outputs
  --git-msg "MSG"    Commit message for --git-push

Examples:
  bash scripts/run_full_experiment.sh
  bash scripts/run_full_experiment.sh --num 30 --start 0 --patch 160 --x0 0 --y0 0
  bash scripts/run_full_experiment.sh --skip-build --skip-install --num 25 --git-push
USAGE
}

# -----------------------------
# Arg parsing
# -----------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num) NUM_SAMPLES="$2"; shift 2;;
    --start) START_IDX="$2"; shift 2;;
    --patch) PATCH="$2"; shift 2;;
    --x0) X0="$2"; shift 2;;
    --y0) Y0="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;
    --out-root) OUT_ROOT="$2"; shift 2;;

    --skip-build) DO_BUILD=0; shift;;
    --skip-install) DO_INSTALL=0; shift;;
    --skip-infer) DO_INFER=0; shift;;
    --skip-topo) DO_TOPO=0; shift;;
    --skip-physics) DO_PHYSICS=0; shift;;
    --skip-plots) DO_PLOTS=0; shift;;

    --git-push) DO_GIT_PUSH=1; shift;;
    --git-msg) GIT_MSG="$2"; shift 2;;

    -h|--help) usage; exit 0;;
    *) die "Unknown arg: $1 (use --help)";;
  esac
done

# -----------------------------
# Main
# -----------------------------
ROOT="$(repo_root)"
cd "$ROOT"

have python3 || die "python3 not found"
have docker || die "docker not found"

ensure_run_scripts
ensure_tf1_compat

END_IDX=$((START_IDX + NUM_SAMPLES - 1))
SAMPLES_LIST="$(seq -s ' ' "$START_IDX" "$END_IDX")"

echo "============================================================"
echo "[run] root: $ROOT"
echo "[run] samples: $START_IDX..$END_IDX (N=$NUM_SAMPLES)"
echo "[run] patch: ${PATCH} x0=${X0} y0=${Y0}"
echo "[run] threads: $THREADS"
echo "[run] out-root: $OUT_ROOT"
echo "============================================================"

# 1) Build TFRecords (WTK -> TFRecords)
if [[ $DO_BUILD -eq 1 ]]; then
  echo "[step] build TFRecords: build_example_data_extension_500.py"
  retry 3 python3 build_example_data_extension_500.py || die "TFRecord build failed"
else
  echo "[skip] build TFRecords"
fi

# 2) Install TFRecords into example_data/
if [[ $DO_INSTALL -eq 1 ]]; then
  echo "[step] install TFRecords into example_data/"
  mkdir -p example_data_backup
  mv example_data/wind_*.tfrecord example_data_backup/ 2>/dev/null || true
  cp example_data_extension_500/wind_*.tfrecord example_data/
  ls -lh example_data/wind_*.tfrecord
else
  echo "[skip] install TFRecords"
fi

# 3) Inference (TFRecords -> data_out/wind-*)
if [[ $DO_INFER -eq 1 ]]; then
  echo "[step] paired inference: GAN"
  BEFORE="$(latest_wind_outdir || true)"
  python3 run_paired_wind_mr_hr.py || die "GAN paired inference failed"
  GAN_OUT="$(latest_wind_outdir)"
  echo "[info] BEFORE=$BEFORE"
  echo "[info] GAN_OUT=$GAN_OUT"

  echo "[step] paired inference: CNN"
  BEFORE="$(latest_wind_outdir || true)"
  python3 run_paired_wind_mr_hr_cnn.py || die "CNN paired inference failed"
  CNN_OUT="$(latest_wind_outdir)"
  echo "[info] BEFORE=$BEFORE"
  echo "[info] CNN_OUT=$CNN_OUT"

  echo "[step] copy outputs to canonical folders"
  copy_outputs_to_canonical "$GAN_OUT" "data_out/wind_mrhr_gan"
  copy_outputs_to_canonical "$CNN_OUT" "data_out/wind_mrhr_cnn"

  echo "[check] shapes"
  verify_counts
  echo "[check] GAN vs CNN differ"
  verify_gan_cnn_differ
else
  echo "[skip] inference"
fi

# 4) Topology pipeline (NPY -> VTI -> PD/MT -> distances -> combined)
if [[ $DO_TOPO -eq 1 ]]; then
  echo "[step] topology pipeline"
  rm -rf vtk_inputs "$OUT_ROOT"
  mkdir -p vtk_inputs "$OUT_ROOT"

  bash scripts/run_topology_pipeline.sh \
    --methods "${METHODS[@]}" \
    --samples $SAMPLES_LIST \
    --scalar speed \
    --patch "$PATCH" --x0 "$X0" --y0 "$Y0" \
    --threads "$THREADS" \
    --out-root "$OUT_ROOT" \
    --debug \
    --cleanup-per-method

  [[ -f "$OUT_ROOT/combined/combined_pairwise_results.csv" ]] || die "Missing combined_pairwise_results.csv"
else
  echo "[skip] topology pipeline"
fi

# 5) Merge physics + topology
if [[ $DO_PHYSICS -eq 1 ]]; then
  echo "[step] analysis_compare.py (physics merge)"
  python3 scripts/analysis_compare.py \
    --topology-csv "$OUT_ROOT/combined/combined_pairwise_results.csv" \
    --method-dir gan=data_out/wind_mrhr_gan \
    --method-dir cnn=data_out/wind_mrhr_cnn \
    --out-csv "$OUT_ROOT/combined/psnr_topology_physics_merged.csv" \
    --out-report "$OUT_ROOT/combined/psnr_topology_physics_report.txt"
else
  echo "[skip] physics merge"
fi

# 6) Plots
if [[ $DO_PLOTS -eq 1 ]]; then
  echo "[step] plot_topology_physics_analysis.py"
  python3 scripts/plot_topology_physics_analysis.py \
    --merged-csv "$OUT_ROOT/combined/psnr_topology_physics_merged.csv" \
    --delta-csv "$OUT_ROOT/combined/psnr_topology_physics_delta.csv" \
    --outdir "$OUT_ROOT/combined/figs_topology_physics"
else
  echo "[skip] plots"
fi

# 7) Summary
if [[ -f "$OUT_ROOT/combined/psnr_topology_physics_merged.csv" ]]; then
  echo "============================================================"
  echo "[summary] merged counts + means"
  summarize_merged
  echo "============================================================"
fi

# 8) Optional git push (only combined outputs)
if [[ $DO_GIT_PUSH -eq 1 ]]; then
  echo "[step] git push combined outputs only"
  git restore --staged . 2>/dev/null || true

  git add -f \
    "$OUT_ROOT/combined/"*.csv \
    "$OUT_ROOT/combined/"*.txt \
    "$OUT_ROOT/combined/"*.png \
    "$OUT_ROOT/combined/figs_topology_physics"

  echo "[info] staged:"
  git diff --cached --name-only | sed -n '1,120p'

  if [[ -z "$(git diff --cached --name-only)" ]]; then
    echo "[info] nothing staged; skipping commit/push"
  else
    git commit -m "${GIT_MSG} (N=${NUM_SAMPLES})"
    git push
  fi
else
  echo "[skip] git push"
fi

echo "[done] full experiment complete"
