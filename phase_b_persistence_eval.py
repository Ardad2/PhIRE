import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

import gudhi as gd
from gudhi.wasserstein import wasserstein_distance
from gudhi import bottleneck_distance


#Helpers

def _to_uv_last(arr: np.ndarray) -> np.ndarray:
    """
    Ensure array is (..., H, W, 2). Supports common layouts:
      - (N,H,W,2)
      - (N,2,H,W)
      - (H,W,2)
      - (2,H,W)
    """
    a = np.asarray(arr)
    if a.ndim == 4:
        # (N,H,W,2)
        if a.shape[-1] == 2:
            return a
        # (N,2,H,W)
        if a.shape[1] == 2:
            return np.transpose(a, (0, 2, 3, 1))
    if a.ndim == 3:
        # (H,W,2)
        if a.shape[-1] == 2:
            return a[None, ...]
        # (2,H,W)
        if a.shape[0] == 2:
            return np.transpose(a, (1, 2, 0))[None, ...]
    raise ValueError(f"Unsupported shape for wind field: {a.shape}")


def magnitude(uv: np.ndarray) -> np.ndarray:
    """uv: (N,H,W,2) -> mag: (N,H,W)"""
    u = uv[..., 0]
    v = uv[..., 1]
    return np.sqrt(u * u + v * v)

def normalize_pair(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-12):
    """
    Avoid "range artifacts" by normalizing both using GT min/max (pairwise normalization).
    Returns (gt_n, pred_n, mn, mx).
    """
    mn = float(np.min(gt))
    mx = float(np.max(gt))
    denom = (mx - mn) + eps
    gt_n = (gt - mn) / denom
    pred_n = (pred - mn) / denom
    # keep thresholds stable
    gt_n = np.clip(gt_n, 0.0, 1.0)
    pred_n = np.clip(pred_n, 0.0, 1.0)
    return gt_n, pred_n, mn, mx

#Convert MSE to PSNR
def psnr_from_mse(mse: float, data_range: float = 1.0, eps: float = 1e-12) -> float:
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse + eps)


def pd_from_field_superlevel(field01: np.ndarray):
    """
    Compute PDs for superlevel sets by
    computing sublevel persistence on (-field).

    field01: 2D array, assumed in [0,1].
    Returns dict {0: [(b,d),...], 1: [(b,d),...]} in the (-field) convention.
    Note: Distances are invariant to the global sign flip.
    """
    f = -np.asarray(field01, dtype=np.float64)
    cc = gd.CubicalComplex(top_dimensional_cells=f)
    pers = cc.persistence()

    diags = {0: [], 1: []}
    for dim, (b, d) in pers:
        if dim in (0, 1):
            # Keep finite pairs only
            if np.isfinite(b) and np.isfinite(d):
                diags[dim].append((float(b), float(d)))
    return diags


def _as_diag(d):
    a = np.asarray(d, dtype=np.float64)
    if a.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return a.reshape(-1, 2)

def prune_diag(diag, min_pers=1e-3, max_pts=300):
    a = np.asarray(diag, dtype=np.float64).reshape(-1, 2)
    if a.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    pers = np.abs(a[:, 1] - a[:, 0])
    a = a[pers >= min_pers]
    if a.shape[0] > max_pts:
        pers = np.abs(a[:, 1] - a[:, 0])
        idx = np.argsort(pers)[::-1][:max_pts]
        a = a[idx]
    return a



def distances(diag_a, diag_b):
    """
    Returns (w1, w2, bottleneck) for two diagrams (list of (b,d)).
    """

    A = prune_diag(diag_a, min_pers=1e-3, max_pts=300)
    B = prune_diag(diag_b, min_pers=1e-3, max_pts=300)

    # Gudhi expects list[(b,d)]
    w1 = float(wasserstein_distance(A, B, order=1, internal_p=2))
    w2 = float(wasserstein_distance(A, B, order=2, internal_p=2))
    bn = float(bottleneck_distance(A, B))
    return w1, w2, bn


def bicubic_resize_to(target_hw, src_hw, src_field):
    """
    Bicubic resize of a 2D field to match target H,W with scikit-image resize (order=3).
    """
    from skimage.transform import resize
    th, tw = target_hw
    sh, sw = src_hw
    if (th, tw) == (sh, sw):
        return src_field
    out = resize(
        src_field,
        (th, tw),
        order=3,            # bicubic resizing
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True
    )
    return out.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gan_dir", default="data_out/wind_mrhr_gan", help="run dir containing dataGT.npy/dataIN.npy/dataSR.npy for GAN")
    ap.add_argument("--cnn_dir", default="data_out/wind_mrhr_cnn", help="run dir containing dataGT.npy/dataIN.npy/dataSR.npy for CNN")
    ap.add_argument("--out_csv", default="phase_b_persistence_results.csv")
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all samples")
    args = ap.parse_args()

    run_dirs = {
        "GAN_MRHR": Path(args.gan_dir),
        "CNN_MRHR": Path(args.cnn_dir),
    }

    rows = []
    for run_name, rdir in run_dirs.items():
        gt_path = rdir / "dataGT.npy"
        sr_path = rdir / "dataSR.npy"
        in_path = rdir / "dataIN.npy"

        if not gt_path.exists() or not sr_path.exists() or not in_path.exists():
            raise FileNotFoundError(f"{run_name}: expected files not found in {rdir}")

        gt_uv = _to_uv_last(np.load(gt_path))
        sr_uv = _to_uv_last(np.load(sr_path))
        in_uv = _to_uv_last(np.load(in_path))

        gt_mag = magnitude(gt_uv)   # (N, Hh, Wh)
        sr_mag = magnitude(sr_uv)   # (N, Hh, Wh) (match GT)
        in_mag = magnitude(in_uv)   # (N, Hm, Wm) (MR input)

        n = gt_mag.shape[0]
        if args.max_samples and args.max_samples > 0:
            n = min(n, args.max_samples)

        for i in range(n):
            gt2 = gt_mag[i]
            sr2 = sr_mag[i]

            # Bicubic baseline: resize u and v separately, then magnitude
            mr_uv2 = in_uv[i]          # shape (Hm,Wm,2) because _to_uv_last made N,H,W,2
            mr_u = mr_uv2[..., 0]
            mr_v = mr_uv2[..., 1]

            bic_u = bicubic_resize_to(gt2.shape, mr_u.shape, mr_u)
            bic_v = bicubic_resize_to(gt2.shape, mr_v.shape, mr_v)

            bic2 = np.sqrt(bic_u * bic_u + bic_v * bic_v)


            # Normalize SR and BIC using GT min/max (pairwise)
            gt_n, sr_n, mn, mx = normalize_pair(gt2, sr2)
            _,  bic_n, _,  _  = normalize_pair(gt2, bic2)

            # Pixel metrics on normalized magnitude
            mse_sr = float(np.mean((sr_n - gt_n) ** 2))
            psnr_sr = float(psnr_from_mse(mse_sr, data_range=1.0))

            mse_bic = float(np.mean((bic_n - gt_n) ** 2))
            psnr_bic = float(psnr_from_mse(mse_bic, data_range=1.0))

            # PDs (superlevel via -field)
            pd_gt = pd_from_field_superlevel(gt_n)
            pd_sr = pd_from_field_superlevel(sr_n)
            pd_bi = pd_from_field_superlevel(bic_n)

            # Distances vs GT
            w1_0_sr, w2_0_sr, bn_0_sr = distances(pd_sr[0], pd_gt[0])
            w1_1_sr, w2_1_sr, bn_1_sr = distances(pd_sr[1], pd_gt[1])

            w1_0_bi, w2_0_bi, bn_0_bi = distances(pd_bi[0], pd_gt[0])
            w1_1_bi, w2_1_bi, bn_1_bi = distances(pd_bi[1], pd_gt[1])

            rows.append({
                "run": run_name,
                "sample": i,
                "gt_min": mn,
                "gt_max": mx,
                "gt_shape": f"{gt2.shape[0]}x{gt2.shape[1]}",
                "mr_shape": f"{mr2.shape[0]}x{mr2.shape[1]}",
                "mse_sr": mse_sr,
                "psnr_sr": psnr_sr,
                "mse_bic": mse_bic,
                "psnr_bic": psnr_bic,

                "w1_pd0_sr": w1_0_sr,
                "w2_pd0_sr": w2_0_sr,
                "bn_pd0_sr": bn_0_sr,
                "w1_pd1_sr": w1_1_sr,
                "w2_pd1_sr": w2_1_sr,
                "bn_pd1_sr": bn_1_sr,

                "w1_pd0_bic": w1_0_bi,
                "w2_pd0_bic": w2_0_bi,
                "bn_pd0_bic": bn_0_bi,
                "w1_pd1_bic": w1_1_bi,
                "w2_pd1_bic": w2_1_bi,
                "bn_pd1_bic": bn_1_bi,

                "n_pairs_pd0_gt": len(pd_gt[0]),
                "n_pairs_pd1_gt": len(pd_gt[1]),
            })

        print(f"[OK] {run_name}: processed {n} samples from {rdir}")

    df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    df.to_csv(out_path, index=False)
    print(f"[WROTE] {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    main()
