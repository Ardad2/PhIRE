#!/usr/bin/env bash
set -euo pipefail

TF_PLATFORM="${TF_PLATFORM:-linux/amd64}"
TOPO_PLATFORM="${TOPO_PLATFORM:-linux/amd64}"

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

    # MR->HR paired outputs
    python run_paired_wind_mr_hr.py      # GAN
    python run_paired_wind_mr_hr_cnn.py  # CNN
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
