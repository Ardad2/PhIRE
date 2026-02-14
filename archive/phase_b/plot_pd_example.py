import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd

from skimage.transform import resize


def _to_uv_last(arr):
    a = np.asarray(arr)
    if a.ndim == 4:
        if a.shape[-1] == 2: return a
        if a.shape[1] == 2: return np.transpose(a, (0,2,3,1))
    raise ValueError(a.shape)

def magnitude(uv):
    return np.sqrt(uv[...,0]**2 + uv[...,1]**2)

def normalize_pair(gt, pred, eps=1e-12):
    mn = float(np.min(gt)); mx = float(np.max(gt))
    denom = (mx - mn) + eps
    gt_nc = (gt - mn) / denom
    pred_nc = (pred - mn) / denom
    gt_c = np.clip(gt_nc, 0, 1)
    pred_c = np.clip(pred_nc, 0, 1)
    return gt_c, pred_c

def pd_superlevel(field01):
    f = -np.asarray(field01, dtype=np.float64)
    cc = gd.CubicalComplex(top_dimensional_cells=f)
    pers = cc.persistence()
    d0, d1 = [], []
    for dim, (b,d) in pers:
        if not (np.isfinite(b) and np.isfinite(d)): 
            continue
        if dim == 0: d0.append((b,d))
        if dim == 1: d1.append((b,d))
    return np.array(d0), np.array(d1)

def bicubic_to(shape_hw, field2d):
    return resize(field2d, shape_hw, order=3, mode="reflect", anti_aliasing=True, preserve_range=True).astype(np.float64)

def scatter_pd(ax, diag, label):
    if diag.size == 0:
        return
    ax.scatter(diag[:,0], diag[:,1], s=10, alpha=0.8, label=label)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g. data_out/wind_mrhr_gan or data_out/wind_mrhr_cnn")
    ap.add_argument("--name", required=True, help="GAN_MRHR or CNN_MRHR")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--field", choices=["speed","u","v"], default="speed")
    ap.add_argument("--outdir", default="figs_phase_b")
    args = ap.parse_args()

    rdir = Path(args.run_dir)
    gt = _to_uv_last(np.load(rdir/"dataGT.npy"))[args.sample]
    sr = _to_uv_last(np.load(rdir/"dataSR.npy"))[args.sample]
    mr = _to_uv_last(np.load(rdir/"dataIN.npy"))[args.sample]

    if args.field == "speed":
        gt2 = magnitude(gt)
        sr2 = magnitude(sr)
        # bicubic: upsample u,v then magnitude
        bic_u = bicubic_to(gt2.shape, mr[...,0])
        bic_v = bicubic_to(gt2.shape, mr[...,1])
        bic2 = np.sqrt(bic_u*bic_u + bic_v*bic_v)
    elif args.field == "u":
        gt2 = gt[...,0]; sr2 = sr[...,0]
        bic2 = bicubic_to(gt2.shape, mr[...,0])
    else:
        gt2 = gt[...,1]; sr2 = sr[...,1]
        bic2 = bicubic_to(gt2.shape, mr[...,1])

    gt_c, sr_c = normalize_pair(gt2, sr2)
    _, bic_c = normalize_pair(gt2, bic2)

    gt0, gt1 = pd_superlevel(gt_c)
    sr0, sr1 = pd_superlevel(sr_c)
    bi0, bi1 = pd_superlevel(bic_c)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    for dim, (dgt, dsr, dbi) in [(0,(gt0,sr0,bi0)), (1,(gt1,sr1,bi1))]:
        fig, ax = plt.subplots(figsize=(6,5))
        scatter_pd(ax, dgt, "GT")
        scatter_pd(ax, dsr, "SR")
        scatter_pd(ax, dbi, "BICUBIC")

        # diagonal
        lo, hi = ax.get_xlim()
        lo2, hi2 = ax.get_ylim()
        lo = min(lo, lo2); hi = max(hi, hi2)
        ax.plot([lo,hi],[lo,hi], linewidth=1)

        ax.set_title(f"{args.name} sample {args.sample}: PD{dim} on {args.field} (superlevel)")
        ax.set_xlabel("birth"); ax.set_ylabel("death")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"pd_overlay_{args.name}_s{args.sample}_{args.field}_pd{dim}.png", dpi=200)
        plt.close(fig)

    print("[OK] wrote PD overlays to", outdir)

if __name__ == "__main__":
    main()
