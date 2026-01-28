#!/usr/bin/env bash
set -euo pipefail

ARCH="$(uname -m)"
DEFAULT_PLATFORM="linux/amd64"
TF_PLATFORM="${TF_PLATFORM:-$DEFAULT_PLATFORM}"
TOPO_PLATFORM="${TOPO_PLATFORM:-$DEFAULT_PLATFORM}"

echo "Host arch: ${ARCH}"
echo "TF_PLATFORM: ${TF_PLATFORM}"
echo "TOPO_PLATFORM: ${TOPO_PLATFORM}"

echo "== Phase B: SR generation (TF1.15.5 container) =="
docker run -it --rm \
  --platform "${TF_PLATFORM}" \
  -u "$(id -u)":"$(id -g)" \
  -e HOME=/work \
  -e PIP_CACHE_DIR=/work/.cache/pip \
  -v "$PWD":/work \
  -w /work \
  tensorflow/tensorflow:1.15.5-py3 \
  bash -lc '
    set -e
    python -V
    python -c "import tensorflow as tf; print(tf.__version__)"
    python run_paired_wind_mr_hr.py
    python run_paired_wind_mr_hr_cnn.py
  '

echo "== Phase B: Topology container build (Gudhi) =="
docker buildx build \
  --platform "${TOPO_PLATFORM}" \
  -f Dockerfile.topo \
  -t phire-topo:phaseb \
  --load \
  .

echo "== Phase B: Persistence eval + plots (topology container) =="
rm -rf figs_phase_b
docker run -it --rm \
  --platform "${TOPO_PLATFORM}" \
  -u "$(id -u)":"$(id -g)" \
  -e HOME=/work \
  -v "$PWD":/work \
  -w /work \
  phire-topo:phaseb \
  bash -lc '
    set -e
    python -V
    python -c "import gudhi; print(\"gudhi:\", gudhi.__version__)"

    # Always recreate output dir inside container (prevents savefig failures)
    mkdir -p figs_phase_b

    python phase_b_persistence_eval.py \
      --gan_dir data_out/wind_mrhr_gan \
      --cnn_dir data_out/wind_mrhr_cnn \
      --out_csv phase_b_persistence_results.csv \
      --fields speed \
      --min_pers 1e-2 \
      --max_pts 120 \
      --patch 160 \
      --stride 160 \
      --w2_audit_every 3

    # Recreate again just in case anything nuked it
    mkdir -p figs_phase_b
    python plot_phase_b_summary.py

    echo "DONE. Key outputs:"
    ls -1 phase_b_persistence_results.csv figs_phase_b/*.png figs_phase_b/*.csv
  '

