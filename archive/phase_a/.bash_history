python -c "import tensorflow as tf; print(tf.__version__)"
python run_paired_wind_lr_mr.py
python - <<'PY'
import tensorflow as tf
from PhIREGANs import PhIREGANs

p = PhIREGANs(data_type='wind', mu_sig=[[0,0],[1,1]])  # dummy
ds = tf.data.TFRecordDataset('example_data/wind_LR-MR.tfrecord')
ds = ds.map(p._parse_train_).batch(1)

it = ds.make_one_shot_iterator()
idx, lr, hr = it.get_next()

with tf.Session() as sess:
    try:
        i, a, b = sess.run([idx, lr, hr])
        print("SUCCESS: HR exists in this TFRecord")
        print("idx:", i, "lr:", a.shape, "hr:", b.shape)
    except Exception as e:
        print("FAILED: no HR in this TFRecord (or schema mismatch)")
        print(type(e).__name__, str(e)[:300])
PY

clear
find data_out -maxdepth 2 -type f | sort
python - <<'PY'
import numpy as np, glob, os

run = sorted(glob.glob("data_out/wind-*/"))[-1]
print("Using run folder:", run)

x = np.load(os.path.join(run, "dataIN.npy"))
y = np.load(os.path.join(run, "dataGT.npy"))
p = np.load(os.path.join(run, "dataSR.npy"))
idx = np.load(os.path.join(run, "idx.npy"))

print("idx:", idx.shape, idx[:10])
print("IN :", x.shape, x.dtype, "min/max", float(x.min()), float(x.max()))
print("GT :", y.shape, y.dtype, "min/max", float(y.min()), float(y.max()))
print("SR :", p.shape, p.dtype, "min/max", float(p.min()), float(p.max()))
PY

python run_paired_wind_mr_hr.py
clear
python run_paired_wind_mr_hr_cnn.py
ls -lt data_out | head
find data_out/wind-20260115-152200 -maxdepth 1 -type f -print
mv data_out/wind-20260115-151022 data_out/wind_mrhr_gan
mv data_out/wind-20260115-152200 data_out/wind_mrhr_cnn
ls -l data_out
export HOME=/work
python -m pip install --user --no-cache-dir scipy pandas
python topo_euler_eval_container.py
ls -l data_out/wind_mrhr_gan
ls -l data_out/wind_mrhr_cnn
clear
python topology_psnr_compare.py
python make_quick_figs.py
ls -lh figs_quick
python quick_figs.py
ls -lh figs_quick
python plot_metric_summary.py
ls -lh psnr_bar.png chiL2_bar.png
python plot_euler_curves_mrhr.py
ls -l *.png
clear
exit
cd /work
ls
python -V
python -c "import tensorflow as tf; print(tf.__version__)"
python run_paired_wind_mr_hr.py
python run_paired_wind_mr_hr_cnn.py
python quick_figs.py
python topo_euler_eval_container.py
python plot_euler_curves_mrhr.py
python plot_metric_summary.py
ls -1 figs_quick/*.png topo_euler_results_mrhr.csv euler_curve_* psnr_bar.png
exit
