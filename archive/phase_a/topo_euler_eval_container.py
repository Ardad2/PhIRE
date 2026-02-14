import os
import numpy as np
import pandas as pd
from scipy.ndimage import label

def wind_speed(uv):
    return np.sqrt(uv[...,0]**2 + uv[...,1]**2).astype(np.float32)

def betti0_and_holes(mask):
    # 4-connected components by default in scipy.ndimage.label for 2D
    _, n_comp = label(mask)
    _, n_bg = label(~mask)
    holes = max(n_bg - 1, 0)
    return n_comp, holes

def euler_curve(field, thresholds, sublevel=True):
    chi = []
    for t in thresholds:
        mask = (field <= t) if sublevel else (field >= t)
        b0, holes = betti0_and_holes(mask)
        chi.append(b0 - holes)
    return np.asarray(chi, dtype=np.float32)

def curve_dist(a, b):
    d = a - b
    return float(np.mean(np.abs(d))), float(np.sqrt(np.mean(d*d))), float(np.max(np.abs(d)))

def bicubic_to_target(x_in, target_hw):
    # simple separable bicubic via scipy zoom (order=3)
    from scipy.ndimage import zoom
    Ht, Wt = target_hw
    H, W, C = x_in.shape
    zH, zW = Ht / H, Wt / W
    out = zoom(x_in, (zH, zW, 1), order=3)
    return out.astype(np.float32)

def eval_run(run_dir, tag, n_thresh=50, sublevel=True):
    x = np.load(os.path.join(run_dir, "dataIN.npy"))
    y = np.load(os.path.join(run_dir, "dataGT.npy"))
    p = np.load(os.path.join(run_dir, "dataSR.npy"))
    idx = np.load(os.path.join(run_dir, "idx.npy"))

    y_sp = wind_speed(y)
    p_sp = wind_speed(p)

    target_hw = y.shape[1:3]
    b_sp = np.stack([wind_speed(bicubic_to_target(x[i], target_hw)[None,...])[0] for i in range(x.shape[0])], axis=0)

    rows = []
    for i in range(y_sp.shape[0]):
        gt = y_sp[i]; sr = p_sp[i]; bc = b_sp[i]
        tmin, tmax = float(gt.min()), float(gt.max())
        thresholds = np.linspace(tmin, tmax, n_thresh)

        chi_gt = euler_curve(gt, thresholds, sublevel=sublevel)
        chi_sr = euler_curve(sr, thresholds, sublevel=sublevel)
        chi_bc = euler_curve(bc, thresholds, sublevel=sublevel)

        sr_l1, sr_l2, sr_mx = curve_dist(chi_sr, chi_gt)
        bc_l1, bc_l2, bc_mx = curve_dist(chi_bc, chi_gt)

        rows.append({
            "run": tag,
            "sample_idx": int(idx[i]),
            "SR_chi_L1": sr_l1, "SR_chi_L2": sr_l2, "SR_chi_max": sr_mx,
            "BIC_chi_L1": bc_l1, "BIC_chi_L2": bc_l2, "BIC_chi_max": bc_mx,
            "tmin": tmin, "tmax": tmax
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    gan_dir = "data_out/wind_mrhr_gan"
    cnn_dir = "data_out/wind_mrhr_cnn"

    out = pd.concat([
        eval_run(gan_dir, "GAN_MRHR"),
        eval_run(cnn_dir, "CNN_MRHR"),
    ], ignore_index=True)

    out.to_csv("topo_euler_results_mrhr.csv", index=False)

    summary = out.groupby("run")[["SR_chi_L1","SR_chi_L2","SR_chi_max","BIC_chi_L1","BIC_chi_L2","BIC_chi_max"]].mean()
    print("\n=== Mean χ-curve distances vs GT (lower is better) ===")
    print(summary)
    print("\nWrote topo_euler_results_mrhr.csv")
