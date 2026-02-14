import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt

def load_run(run_dir):
    x = np.load(os.path.join(run_dir, "dataIN.npy"))   # (N,h,w,2)
    y = np.load(os.path.join(run_dir, "dataGT.npy"))   # (N,H,W,2)
    p = np.load(os.path.join(run_dir, "dataSR.npy"))   # (N,H,W,2)
    idx = np.load(os.path.join(run_dir, "idx.npy"))
    return idx, x, y, p

def speed(a):
    return np.sqrt(a[...,0]**2 + a[...,1]**2)

def save_triptych(figpath, GT, GAN, CNN, title, vmin=None, vmax=None):
    fig, axs = plt.subplots(1,3, figsize=(12,4))
    for ax, img, name in zip(axs, [GT, GAN, CNN], ["GT", "GAN", "CNN"]):
        im = ax.imshow(img, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title)
    fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(figpath, dpi=200)
    plt.close(fig)

def save_error_pair(figpath, err_gan, err_cnn, title):
    fig, axs = plt.subplots(1,2, figsize=(9,4))
    for ax, img, name in zip(axs, [err_gan, err_cnn], ["|GAN-GT|", "|CNN-GT|"]):
        im = ax.imshow(img)
        ax.set_title(name)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title)
    fig.colorbar(im, ax=axs, fraction=0.03, pad=0.03)
    fig.tight_layout()
    fig.savefig(figpath, dpi=200)
    plt.close(fig)

def main():
    # assumes you renamed these folders already
    gan_dir = "data_out/wind_mrhr_gan"
    cnn_dir = "data_out/wind_mrhr_cnn"

    outdir = "figs_quick"
    os.makedirs(outdir, exist_ok=True)

    idx_g, _, gt_g, sr_g = load_run(gan_dir)
    idx_c, _, gt_c, sr_c = load_run(cnn_dir)

    # sanity: assume same GT ordering
    assert np.all(idx_g == idx_c), "idx mismatch between GAN and CNN runs"
    assert gt_g.shape == gt_c.shape, "GT shapes differ"

    # pick one sample
    k = 0
    GT = gt_g[k]
    GAN = sr_g[k]
    CNN = sr_c[k]

    # components
    for ch, name in [(0,"u"), (1,"v")]:
        vmin = float(np.min([GT[...,ch].min(), GAN[...,ch].min(), CNN[...,ch].min()]))
        vmax = float(np.max([GT[...,ch].max(), GAN[...,ch].max(), CNN[...,ch].max()]))
        save_triptych(
            os.path.join(outdir, f"triptych_{name}.png"),
            GT[...,ch], GAN[...,ch], CNN[...,ch],
            title=f"Wind component {name} (sample {k})",
            vmin=vmin, vmax=vmax
        )
        save_error_pair(
            os.path.join(outdir, f"error_{name}.png"),
            np.abs(GAN[...,ch]-GT[...,ch]),
            np.abs(CNN[...,ch]-GT[...,ch]),
            title=f"Absolute error in {name} (sample {k})"
        )

    # speed magnitude
    GTs, GANs, CNNs = speed(GT), speed(GAN), speed(CNN)
    vmin = float(np.min([GTs.min(), GANs.min(), CNNs.min()]))
    vmax = float(np.max([GTs.max(), GANs.max(), CNNs.max()]))
    save_triptych(
        os.path.join(outdir, "triptych_speed.png"),
        GTs, GANs, CNNs,
        title=f"Speed magnitude |v| (sample {k})",
        vmin=vmin, vmax=vmax
    )
    save_error_pair(
        os.path.join(outdir, "error_speed.png"),
        np.abs(GANs-GTs),
        np.abs(CNNs-GTs),
        title=f"Absolute error in speed |v| (sample {k})"
    )

    print("Wrote figures to:", outdir)

if __name__ == "__main__":
    main()
