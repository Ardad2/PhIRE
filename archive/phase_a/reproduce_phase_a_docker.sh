#!/usr/bin/env bash
set -euo pipefail

IMAGE="tensorflow/tensorflow:1.15.5-py3"
WORKDIR="/work"

docker run --rm \
  --platform linux/amd64 \
  -u "$(id -u)":"$(id -g)" \
  -e HOME="${WORKDIR}" \
  -e PYTHONUSERBASE="${WORKDIR}/.local" \
  -e PIP_CACHE_DIR="${WORKDIR}/.cache/pip" \
  -v "$PWD":"${WORKDIR}" \
  -w "${WORKDIR}" \
  "${IMAGE}" \
  bash -lc '
    set -euo pipefail

    echo "Python:"; python -V
    echo "TensorFlow:"; python -c "import tensorflow as tf; print(tf.__version__)"

    # Install the few deps Phase A needs (user-site, no permission issues)
    if [ -f requirements-docker.txt ]; then
      python -m pip install --user --no-cache-dir -r requirements-docker.txt
    else
      python -m pip install --user --no-cache-dir pandas==1.1.5 scipy==1.5.4 matplotlib==3.3.4
    fi

    # Clean + deterministic outputs (so reruns are consistent)
    rm -rf data_out/wind_mrhr_gan data_out/wind_mrhr_cnn figs_quick chi_bar.png
    rm -f euler_curve_speed_mrhr.png euler_curve_absdiff_speed_mrhr.png topo_euler_results_mrhr.csv psnr_mse.txt

    mkdir -p data_out

    # ---- Run GAN MR->HR ----
    python run_paired_wind_mr_hr.py
    if [ ! -d data_out/wind_mrhr_gan ]; then
      latest=$(ls -dt data_out/wind-* 2>/dev/null | head -n 1)
      [ -n "${latest}" ] && mv "${latest}" data_out/wind_mrhr_gan
    fi

    # ---- Run CNN MR->HR ----
    python run_paired_wind_mr_hr_cnn.py
    if [ ! -d data_out/wind_mrhr_cnn ]; then
      latest=$(ls -dt data_out/wind-* 2>/dev/null | head -n 1)
      [ -n "${latest}" ] && mv "${latest}" data_out/wind_mrhr_cnn
    fi

    # Figures + topology metrics
    python quick_figs.py
    python topo_euler_eval_container.py
    python plot_euler_curves_mrhr.py
    python plot_metric_summary.py

    # Pixel metrics (log to file)
    python topology_psnr_compare.py | tee psnr_mse.txt

    echo ""
    echo "DONE. Key outputs:"
    ls -1 \
      figs_quick/*.png \
      topo_euler_results_mrhr.csv \
      euler_curve_speed_mrhr.png \
      euler_curve_absdiff_speed_mrhr.png \
      chi_bar.png \
      psnr_mse.txt
  '
