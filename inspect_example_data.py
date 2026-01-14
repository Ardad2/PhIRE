import os
import numpy as np

root = "example_data"

def try_load_numpy(p: str):
    try:
        arr = np.load(p, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            print("  -> npz keys:", list(arr.keys())[:20])
        else:
            print("  -> array shape:", arr.shape, "dtype:", arr.dtype, "min/max:", float(arr.min()), float(arr.max()))
    except Exception as e:
        print("  -> (not numpy-loadable)", type(e).__name__, str(e)[:120])

for dirpath, _, filenames in os.walk(root):
    for fn in sorted(filenames):
        p = os.path.join(dirpath, fn)
        print(p)
        if fn.endswith((".npy", ".npz")):
            try_load_numpy(p)
