from pathlib import Path
import numpy as np
import tensorflow as tf
import utils

outdir = Path("example_data_fixed")
outdir.mkdir(parents=True, exist_ok=True)

# Build a tiny synthetic 2-channel HR stack with very different channels
N, H, W, C = 3, 100, 100, 2
hr = np.zeros((N, H, W, C), dtype=np.float64)

for i in range(N):
    hr[i, :, :, 0] = i + 1.0
    hr[i, :, :, 1] = np.arange(H * W, dtype=np.float64).reshape(H, W) + 1000 * i

path = outdir / "synthetic_MR-HR_test.tfrecord"
utils.generate_TFRecords(str(path), hr, mode="train", K=5)

print("Wrote:", path)

# Read back and test whether LR channels are duplicated
exact = 0
rows = []
for row, raw in enumerate(tf.compat.v1.io.tf_record_iterator(str(path))):
    ex = tf.train.Example()
    ex.ParseFromString(raw)
    f = ex.features.feature

    h = int(f["h_LR"].int64_list.value[0])
    w = int(f["w_LR"].int64_list.value[0])
    c = int(f["c"].int64_list.value[0])

    lr = np.frombuffer(f["data_LR"].bytes_list.value[0], dtype=np.float64).reshape(h, w, c)

    if np.array_equal(lr[..., 0], lr[..., 1]):
        exact += 1
        rows.append(row)

print("total rows:", row + 1)
print("exact duplicated rows:", exact)
print("duplicated row ids:", rows)
