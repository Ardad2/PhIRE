#!/usr/bin/env bash
set -euo pipefail

docker run -it --rm \
  --platform linux/amd64 \
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

    # Install any python deps your scripts need (safe to rerun)
    python -m pip install --upgrade pip
    if [ -f requirements.txt ]; then python -m pip install -r requirements.txt; fi

    # Run Phase A in the correct order
    python run_paired_wind_mr_hr.py
    python run_paired_wind_mr_hr_cnn.py
    python quick_figs.py
    python topo_euler_eval_container.py
    python plot_euler_curves_mrhr.py
    python plot_metric_summary.py

    echo "DONE. Key outputs:"
    ls -1 figs_quick/*.png topo_euler_results_mrhr.csv euler_curve_* psnr_bar.png
  '
