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
    Pairwise normalize using GT min/max. Returns:
      gt_nc, pred_nc  : normalized (NOT clipped) in principle around [0,1] but may exceed
      gt_c,  pred_c   : clipped to [0,1] (for PD)
      mn, mx
    """
    mn = float(np.min(gt))
    mx = float(np.max(gt))
    denom = (mx - mn) + eps
    gt_nc = (gt - mn) / denom
    pred_nc = (pred - mn) / denom

    # PD needs stable [0,1]
    gt_c = np.clip(gt_nc, 0.0, 1.0)
    pred_c = np.clip(pred_nc, 0.0, 1.0)
    return gt_nc, pred_nc, gt_c, pred_c, mn, mx

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



def distances(diag_a, diag_b, min_pers=1e-3, max_pts=300):
    """
    Returns (w1, w2, bottleneck) for two diagrams.
    """
    A = prune_diag(diag_a, min_pers=min_pers, max_pts=max_pts)
    B = prune_diag(diag_b, min_pers=min_pers, max_pts=max_pts)

    w1 = float(wasserstein_distance(A, B, order=1, internal_p=2))
    w2 = float(wasserstein_distance(A, B, order=2, internal_p=2))
    bn = float(bottleneck_distance(A, B))
    return w1, w2, bn, int(A.shape[0]), int(B.shape[0])


def bicubic_resize_to(target_hw, src_field):
    """
    Bicubic resize of a 2D field to match target H,W with scikit-image resize (order=3).
    """
    from skimage.transform import resize
    th, tw = target_hw
    out = resize(
        src_field,
        (th, tw),
        order=3,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True
    )
    return out.astype(np.float64)


def compute_block(gt2, sr2, bic2, min_pers, max_pts, prefix):
    """
    Compute PSNR (on unclipped normalized) + PD distances (on clipped).
    Returns dict with keys prefixed by `prefix`.
    """
    gt_nc, sr_nc, gt_c, sr_c, mn, mx = normalize_pair(gt2, sr2)
    _,  bic_nc, _,  bic_c, _,  _     = normalize_pair(gt2, bic2)

    mse_sr = float(np.mean((sr_nc - gt_nc) ** 2))
    psnr_sr = float(psnr_from_mse(mse_sr, data_range=1.0))
    mse_bic = float(np.mean((bic_nc - gt_nc) ** 2))
    psnr_bic = float(psnr_from_mse(mse_bic, data_range=1.0))

    pd_gt = pd_from_field_superlevel(gt_c)
    pd_sr = pd_from_field_superlevel(sr_c)
    pd_bi = pd_from_field_superlevel(bic_c)

    w1_0_sr, w2_0_sr, bn_0_sr, n0_sr, n0_gt = distances(pd_sr[0], pd_gt[0], min_pers, max_pts)
    w1_1_sr, w2_1_sr, bn_1_sr, n1_sr, n1_gt = distances(pd_sr[1], pd_gt[1], min_pers, max_pts)

    w1_0_bi, w2_0_bi, bn_0_bi, n0_bi, _ = distances(pd_bi[0], pd_gt[0], min_pers, max_pts)
    w1_1_bi, w2_1_bi, bn_1_bi, n1_bi, _ = distances(pd_bi[1], pd_gt[1], min_pers, max_pts)

    return {
        f"{prefix}_gt_min": mn,
        f"{prefix}_gt_max": mx,

        f"{prefix}_mse_sr": mse_sr,
        f"{prefix}_psnr_sr": psnr_sr,
        f"{prefix}_mse_bic": mse_bic,
        f"{prefix}_psnr_bic": psnr_bic,

        f"{prefix}_w1_pd0_sr": w1_0_sr,
        f"{prefix}_w2_pd0_sr": w2_0_sr,
        f"{prefix}_bn_pd0_sr": bn_0_sr,
        f"{prefix}_w1_pd1_sr": w1_1_sr,
        f"{prefix}_w2_pd1_sr": w2_1_sr,
        f"{prefix}_bn_pd1_sr": bn_1_sr,

        f"{prefix}_w1_pd0_bic": w1_0_bi,
        f"{prefix}_w2_pd0_bic": w2_0_bi,
        f"{prefix}_bn_pd0_bic": bn_0_bi,
        f"{prefix}_w1_pd1_bic": w1_1_bi,
        f"{prefix}_w2_pd1_bic": w2_1_bi,
        f"{prefix}_bn_pd1_bic": bn_1_bi,

        f"{prefix}_n_pd0_gt": len(pd_gt[0]),
        f"{prefix}_n_pd1_gt": len(pd_gt[1]),
        f"{prefix}_n_pd0_sr_pruned": n0_sr,
        f"{prefix}_n_pd1_sr_pruned": n1_sr,
        f"{prefix}_n_pd0_bic_pruned": n0_bi,
        f"{prefix}_n_pd1_bic_pruned": n1_bi,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gan_dir", default="data_out/wind_mrhr_gan")
    ap.add_argument("--cnn_dir", default="data_out/wind_mrhr_cnn")
    ap.add_argument("--out_csv", default="phase_b_persistence_results.csv")
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all samples")
    ap.add_argument("--min_pers", type=float, default=1e-3, help="prune: min persistence")
    ap.add_argument("--max_pts", type=int, default=300, help="prune: max points kept")
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

        # Scalars
        gt_speed = magnitude(gt_uv)
        sr_speed = magnitude(sr_uv)
        in_speed = magnitude(in_uv)

        gt_u = gt_uv[..., 0]
        gt_v = gt_uv[..., 1]
        sr_u = sr_uv[..., 0]
        sr_v = sr_uv[..., 1]

        n = gt_speed.shape[0]
        if args.max_samples and args.max_samples > 0:
            n = min(n, args.max_samples)

        for i in range(n):
            # GT
            gt2_speed = gt_speed[i]
            gt2_u = gt_u[i]
            gt2_v = gt_v[i]

            # SR
            sr2_speed = sr_speed[i]
            sr2_u = sr_u[i]
            sr2_v = sr_v[i]

            # MR input (for bicubic baseline)
            mr_uv2 = in_uv[i]     # (Hm,Wm,2)
            mr_u = mr_uv2[..., 0]
            mr_v = mr_uv2[..., 1]

            # Bicubic u,v then derived speed
            bic_u = bicubic_resize_to(gt2_speed.shape, mr_u)
            bic_v = bicubic_resize_to(gt2_speed.shape, mr_v)
            bic_speed = np.sqrt(bic_u * bic_u + bic_v * bic_v)

            row = {
                "run": run_name,
                "sample": i,
                "gt_shape": f"{gt2_speed.shape[0]}x{gt2_speed.shape[1]}",
                "mr_shape": f"{in_speed[i].shape[0]}x{in_speed[i].shape[1]}",
                "min_pers": args.min_pers,
                "max_pts": args.max_pts,
            }

            # speed / u / v blocks
            row.update(compute_block(gt2_speed, sr2_speed, bic_speed, args.min_pers, args.max_pts, "speed"))
            row.update(compute_block(gt2_u, sr2_u, bic_u, args.min_pers, args.max_pts, "u"))
            row.update(compute_block(gt2_v, sr2_v, bic_v, args.min_pers, args.max_pts, "v"))

            rows.append(row)

        print(f"[OK] {run_name}: processed {n} samples from {rdir}")

    df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    df.to_csv(out_path, index=False)
    print(f"[WROTE] {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
