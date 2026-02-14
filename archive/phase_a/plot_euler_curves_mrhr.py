import os, numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

GAN_DIR = "data_out/wind_mrhr_gan"
CNN_DIR = "data_out/wind_mrhr_cnn"

def bicubic_upsample(x_in, scale):
    # x_in: (N,h,w,C)
    return ndi.zoom(x_in, zoom=(1, scale, scale, 1), order=3)

def euler_number(binary, conn=8):
    # Euler = (#components in foreground) - (#holes)
    if conn == 8:
        structure = np.ones((3,3), dtype=np.int8)
    else:  # 4-connectivity
        structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.int8)

    labeled_fg, n_fg = ndi.label(binary, structure=structure)

    inv = ~binary
    labeled_bg, n_bg = ndi.label(inv, structure=structure)

    # labels touching border are "outside background"
    border = np.unique(np.concatenate([
        labeled_bg[0, :], labeled_bg[-1, :],
        labeled_bg[:, 0], labeled_bg[:, -1]
    ]))
    border = border[border != 0]
    n_border = len(border)

    holes = n_bg - n_border
    return int(n_fg - holes)

def euler_curve(field2d, thresholds, mode="superlevel"):
    # mode superlevel: binary = field >= t
    # mode sublevel:   binary = field <= t
    chi = []
    for t in thresholds:
        if mode == "superlevel":
            b = field2d >= t
        else:
            b = field2d <= t
        chi.append(euler_number(b, conn=8))
    return np.array(chi, dtype=np.float32)

def load_run(run_dir):
    x_in = np.load(os.path.join(run_dir, "dataIN.npy"))   # (N,h,w,2)
    y_gt = np.load(os.path.join(run_dir, "dataGT.npy"))   # (N,H,W,2)
    y_sr = np.load(os.path.join(run_dir, "dataSR.npy"))   # (N,H,W,2)
    return x_in, y_gt, y_sr

def speed(u, v):
    return np.sqrt(u*u + v*v)

def main():
    x_gan, gt_gan, sr_gan = load_run(GAN_DIR)
    x_cnn, gt_cnn, sr_cnn = load_run(CNN_DIR)

    # sanity: GT should match
    assert gt_gan.shape == gt_cnn.shape, "GT shapes differ between runs"
    gt = gt_gan
    N, H, W, C = gt.shape

    # Bicubic baseline: upsample the INPUT (MR) to HR size
    scale = H // x_gan.shape[1]
    bic = bicubic_upsample(x_gan, scale=scale)

    # Choose scalar field to compute topology on:
    # (1) speed magnitude |v|
    gt_speed  = speed(gt[...,0], gt[...,1])
    gan_speed = speed(sr_gan[...,0], sr_gan[...,1])
    cnn_speed = speed(sr_cnn[...,0], sr_cnn[...,1])
    bic_speed = speed(bic[...,0], bic[...,1])

    # thresholds: use robust range from GT to avoid outliers
    lo = np.quantile(gt_speed, 0.01)
    hi = np.quantile(gt_speed, 0.99)
    thresholds = np.linspace(lo, hi, 80)

    # compute chi-curves per sample
    curves = {}
    for name, arr in [("GT", gt_speed), ("GAN", gan_speed), ("CNN", cnn_speed), ("BIC", bic_speed)]:
        chi_all = []
        for i in range(N):
            chi_all.append(euler_curve(arr[i], thresholds, mode="superlevel"))
        curves[name] = np.stack(chi_all, axis=0)  # (N, T)

    # plot mean +/- std
    plt.figure(figsize=(10,6))
    for name in ["GT","GAN","CNN","BIC"]:
        m = curves[name].mean(axis=0)
        s = curves[name].std(axis=0)
        plt.plot(thresholds, m, label=name)
        plt.fill_between(thresholds, m-s, m+s, alpha=0.15)

    plt.title("Euler characteristic curve (speed |v|) — MR→HR")
    plt.xlabel("Threshold")
    plt.ylabel("Euler characteristic χ")
    plt.legend()
    plt.tight_layout()
    plt.savefig("euler_curve_speed_mrhr.png", dpi=200)
    print("Wrote euler_curve_speed_mrhr.png")

    # Optional: plot absolute deviation from GT
    plt.figure(figsize=(10,6))
    gt_m = curves["GT"].mean(axis=0)
    for name in ["GAN","CNN","BIC"]:
        m = curves[name].mean(axis=0)
        plt.plot(thresholds, np.abs(m - gt_m), label=f"|{name}-GT|")
    plt.title("Mean |Δχ| vs threshold (speed |v|) — MR→HR")
    plt.xlabel("Threshold")
    plt.ylabel("|Δχ|")
    plt.legend()
    plt.tight_layout()
    plt.savefig("euler_curve_absdiff_speed_mrhr.png", dpi=200)
    print("Wrote euler_curve_absdiff_speed_mrhr.png")

if __name__ == "__main__":
    main()
