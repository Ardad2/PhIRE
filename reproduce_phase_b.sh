#!/usr/bin/env bash
set -euo pipefail

# Default: if running on Apple Silicon, force amd64 for both TF1.15 + gudhi wheels.
#TF 1.15.5 images are effectively amd64-only and gudhi==3.9.0 isn't avaliable as a pip install for linux/arm64.
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
    python phase_b_persistence_eval.py \
      --gan_dir data_out/wind_mrhr_gan \
      --cnn_dir data_out/wind_mrhr_cnn \
      --out_csv phase_b_persistence_results.csv
    python plot_phase_b_summary.py
    echo "DONE. Key outputs:"
    ls -1 phase_b_persistence_results.csv figs_phase_b/*.png
  '
