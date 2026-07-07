#!/usr/bin/env python3
"""
Candidate D: PyTorch residual refiner with PD Wasserstein loss.

Implements a small residual CNN (RefinerNet) that post-processes the frozen
baseline CNN SR output to minimize:

    L_total = L_uv
            + lambda_speed * L_speed
            + lambda_grad  * L_grad
            + lambda_crit  * L_crit
            + lambda_pd    * L_PD

where L_PD = W_2 Wasserstein distance between persistence diagrams of scalar
wind speed (superlevel-set topology via torch_topological CubicalComplex).

Must be run inside .venv_candidateD_pd:
    .venv_candidateD_pd/bin/python scripts/run_candidateD_pd_refiner.py [opts]

GUDHI compatibility
-------------------
torch_topological 0.1.9 + gudhi 3.12.0 requires a one-line patch.
This script detects the incompatibility and applies the patch automatically
to the venv source. See docs/candidateD_pd_gradient_smoke.md for details.

Workflow
--------
1. Run --diagnostic-only --lambda-pd 0.0 --pd-crop-size 100
   Read the printed L_PD/L_uv ratio.
2. Choose lambda_pd from the printed recommendations.
3. Run with --lambda-pd <value> --epochs 3
"""

import argparse
import importlib
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Repo paths ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent

_DATA_IN  = REPO_ROOT / "data_out_fixed" / "wind_mrhr_cnn"
_OUT_DIR  = REPO_ROOT / "data_out"        / "wind_finetune_pilot_candidateD"
_MDL_DIR  = REPO_ROOT / "models_fixed"   / "topology_finetuning" / "wind_finetune_pilot_candidateD"
_LOG_PATH = REPO_ROOT / "logs"            / "wind_finetune_pilot_candidateD.log"
_RPT_PATH = REPO_ROOT / "docs"            / "candidateD_pd_refiner_notes.md"

_PROTECTED = [
    REPO_ROOT / "data_out_fixed" / "wind_mrhr_cnn",
    REPO_ROOT / "data_out_fixed" / "wind_mrhr_gan",
    REPO_ROOT / "data_out"       / "wind_finetune_pilot_candidateB",
    REPO_ROOT / "data_out"       / "wind_finetune_pilot_candidateC",
    REPO_ROOT / "models_fixed"   / "topology_finetuning" / "wind_finetune_pilot_candidateB",
    REPO_ROOT / "models_fixed"   / "topology_finetuning" / "wind_finetune_pilot_candidateC",
]

_EPS = 1e-8

# Normalization constants matching the TF1 PhIRE/Candidate B/C convention
# (PhIREGANs' mu_sig=[[0.7684, -0.4575], [5.02455, 5.9017]]). dataSR.npy /
# dataGT.npy are already physical (denormalized) [u, v] -- see
# PhIREGANs.test_paired(), which applies `mu_sig[1]*batch + mu_sig[0]`
# before saving. L_uv must be computed on NORMALIZED [u, v] to match
# sr_network.py's content_loss (computed on normalized x_HR/x_SR); the
# scalar-speed losses below intentionally keep operating on physical units.
_MU_UV    = (0.7684, -0.4575)
_SIGMA_UV = (5.02455, 5.9017)


# ── GUDHI compatibility shim ──────────────────────────────────────────────────

def _apply_gudhi_compat_patch() -> str:
    """
    Detect and fix torch_topological 0.1.9 + gudhi >= 3.10 incompatibility.

    In torch_topological 0.1.9, CubicalComplex._get_persistence() passes a
    torch.Tensor to gudhi.CubicalComplex(), but gudhi >= 3.10 requires a
    numpy array.  This function patches the venv source file if needed.

    Returns one of: "patched", "already_patched", "not_needed", "unknown"
    """
    try:
        import torch_topological.nn.cubical_complex as _mod
    except ImportError:
        raise ImportError(
            "torch_topological not found. "
            "Run: .venv_candidateD_pd/bin/pip install torch-topological"
        )

    src_path = Path(_mod.__file__)
    src = src_path.read_text()

    OLD = "top_dimensional_cells=x.flatten()"
    NEW = "top_dimensional_cells=x.detach().cpu().numpy().flatten()"

    if OLD in src:
        # Apply patch: also fix dimensions= to accept torch.Size
        patched = src.replace(OLD, NEW)
        if "dimensions=x.shape," in patched:
            patched = patched.replace("dimensions=x.shape,", "dimensions=list(x.shape),")
        src_path.write_text(patched)
        importlib.reload(_mod)
        import torch_topological.nn as _tt_nn
        _tt_nn.CubicalComplex = _mod.CubicalComplex
        return "patched"
    elif NEW in src:
        return "already_patched"
    elif "detach().cpu().numpy()" in src and "top_dimensional_cells" in src:
        # Venv was patched with an equivalent form (e.g. via a _cells variable).
        return "already_patched_variant"
    else:
        # Pattern unrecognized; _verify_pd_gradients() is the authoritative check.
        logging.info(
            "[compat] GUDHI call pattern not recognized. "
            "Proceeding — _verify_pd_gradients() is the authoritative check."
        )
        return "unrecognized"


def _verify_pd_gradients() -> None:
    """
    Verify that CubicalComplex + WassersteinDistance produces gradients.
    Aborts with a clear error if the environment is misconfigured.
    """
    from torch_topological.nn import CubicalComplex, WassersteinDistance

    rng = np.random.default_rng(seed=0)
    x_np = rng.random((16, 16)).astype(np.float32)
    y_np = rng.random((16, 16)).astype(np.float32)
    x = torch.tensor(x_np, requires_grad=True)
    y = torch.tensor(y_np)

    try:
        cc = CubicalComplex(superlevel=True)
        wl = WassersteinDistance(q=2)
        loss = wl(cc(x), cc(y))
        loss.backward()
    except Exception as e:
        raise RuntimeError(
            f"PD gradient verification FAILED: {e}\n"
            "Ensure you are running inside .venv_candidateD_pd with the GUDHI\n"
            "compatibility patch applied. Run the audit script first:\n"
            "  .venv_candidateD_pd/bin/python scripts/smoke_candidateD_pd_grad.py\n"
            "See docs/candidateD_pd_gradient_smoke.md for patch instructions."
        ) from e

    if x.grad is None or not (x.grad != 0).any():
        raise RuntimeError(
            "PD backward() produced no gradients on a test tensor.\n"
            "This indicates an environment issue. "
            "See docs/candidateD_pd_gradient_smoke.md."
        )
    nz = int((x.grad != 0).sum())
    logging.info("[verify] PD gradient check passed: %d/256 nonzero entries", nz)


# ── Model ─────────────────────────────────────────────────────────────────────

class RefinerNet(nn.Module):
    """
    Small residual CNN refiner for [u, v] wind fields.

    Initialized so that output ≈ input at t=0:
        refined = cnn_uv + residual_scale * body(cnn_uv)
    where body's last conv layer is zero-initialized.
    """

    def __init__(
        self,
        hidden: int = 32,
        kernel: int = 3,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        pad = kernel // 2
        self.body = nn.Sequential(
            nn.Conv2d(2, hidden, kernel, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel, padding=pad),
        )
        # Zero-initialize the final layer so initial output = input exactly
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_scale * self.body(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Loss functions ─────────────────────────────────────────────────────────────

def _speed_t(uv: torch.Tensor) -> torch.Tensor:
    """(B, 2, H, W) → (B, H, W) scalar speed."""
    return torch.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2 + _EPS)


def _grad_mag_t(s: torch.Tensor) -> torch.Tensor:
    """(B, H, W) → (B, H, W) gradient magnitude via central finite differences."""
    dy = F.pad(s[:, 1:, :] - s[:, :-1, :], (0, 0, 0, 1))
    dx = F.pad(s[:, :, 1:] - s[:, :, :-1], (0, 1, 0, 0))
    return torch.sqrt(dx * dx + dy * dy + _EPS)


def _normalize_uv(uv: torch.Tensor) -> torch.Tensor:
    """(B, 2, H, W) physical [u, v] -> normalized [u, v] (see _MU_UV/_SIGMA_UV)."""
    mu    = torch.tensor(_MU_UV,    dtype=uv.dtype, device=uv.device).view(1, 2, 1, 1)
    sigma = torch.tensor(_SIGMA_UV, dtype=uv.dtype, device=uv.device).view(1, 2, 1, 1)
    return (uv - mu) / sigma


def l_uv(sr: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """MSE on NORMALIZED [u, v] (matches sr_network.py's content_loss).
    sr/gt are physical [u, v] as loaded from dataSR.npy/dataGT.npy."""
    return F.mse_loss(_normalize_uv(sr), _normalize_uv(gt))


def l_speed(sr: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(_speed_t(sr), _speed_t(gt))


def l_grad(sr: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(_grad_mag_t(_speed_t(sr)), _grad_mag_t(_speed_t(gt)))


def l_crit(
    sr: torch.Tensor,
    gt: torch.Tensor,
    pool: int = 3,
    high_z: float = 1.0,
) -> torch.Tensor:
    """
    Critical-value proxy: MSE at GT superlevel-set local maxima.

    Matches the Candidate C convention: local maxima detected by max-pool
    with pool×pool kernel; adaptive threshold = mean + high_z * std of GT speed.
    """
    spd_sr = _speed_t(sr)  # (B, H, W)
    spd_gt = _speed_t(gt)

    s4 = spd_gt.unsqueeze(1)  # (B, 1, H, W)
    local_max = F.max_pool2d(s4, kernel_size=pool, stride=1, padding=pool // 2)
    is_local_max = s4 >= local_max - _EPS  # (B, 1, H, W)

    threshold = spd_gt.mean() + high_z * spd_gt.std()
    high_mask = is_local_max.squeeze(1) & (spd_gt >= threshold)  # (B, H, W)

    if not high_mask.any():
        return torch.zeros((), device=sr.device, requires_grad=True)

    return F.mse_loss(spd_sr[high_mask], spd_gt[high_mask])


def l_pd(
    sr: torch.Tensor,
    gt: torch.Tensor,
    cubical,
    w_loss_fn,
    crop: int = 100,
) -> torch.Tensor:
    """
    PD Wasserstein-2 loss on GT-normalized scalar speed.

    Uses superlevel-set persistence (CubicalComplex(superlevel=True)) on a
    spatial crop of size crop×crop, with speed normalized to [0, 1] using
    the GT range.  Gradients flow to sr via sparse indexing at critical cells.

    Operates per-sample (returns mean over batch).
    """
    spd_sr = _speed_t(sr)   # (B, H, W)
    spd_gt = _speed_t(gt)

    total: torch.Tensor | None = None
    n_ok = 0

    for b in range(spd_sr.shape[0]):
        H_c = min(crop, spd_sr.shape[1])
        W_c = min(crop, spd_sr.shape[2])
        sr_c = spd_sr[b, :H_c, :W_c]
        gt_c = spd_gt[b, :H_c, :W_c]

        gt_min = gt_c.detach().min()
        gt_max = gt_c.detach().max()
        denom = (gt_max - gt_min).clamp(min=_EPS)

        sr_n = ((sr_c - gt_min) / denom).clamp(0.0, 1.0)
        gt_n = ((gt_c - gt_min) / denom).clamp(0.0, 1.0)

        loss_b = w_loss_fn(cubical(sr_n), cubical(gt_n))

        if torch.isnan(loss_b) or torch.isinf(loss_b):
            logging.warning("L_PD NaN/inf for sample %d — skipped.", b)
            continue

        total = loss_b if total is None else total + loss_b
        n_ok += 1

    if total is None or n_ok == 0:
        raise RuntimeError(
            "L_PD returned NaN/inf for all samples in this batch. "
            "Check the speed field range and normalization."
        )
    return total / n_ok


# ── Data utilities ─────────────────────────────────────────────────────────────

def _to_chw(arr: np.ndarray) -> np.ndarray:
    """Return (N, 2, H, W) regardless of input channel layout."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 4:
        if a.shape[-1] == 2:
            return np.transpose(a, (0, 3, 1, 2))
        if a.shape[1] == 2:
            return a
    raise ValueError(f"Cannot interpret shape {a.shape} as [u,v] field")


def load_data(data_dir: Path):
    """Load SR, GT, IN, idx arrays from data_dir. Returns (sr, gt, inp, idx)."""
    sr  = _to_chw(np.load(data_dir / "dataSR.npy"))
    gt  = _to_chw(np.load(data_dir / "dataGT.npy"))
    inp = _to_chw(np.load(data_dir / "dataIN.npy"))  if (data_dir / "dataIN.npy").exists()  else None
    idx = np.load(data_dir / "idx.npy")               if (data_dir / "idx.npy").exists()       else None
    return sr, gt, inp, idx


def _synthetic_batch(device: torch.device, H: int = 160, W: int = 160):
    """
    Generate a synthetic [u,v] batch with realistic wind-speed statistics
    for diagnostic use when real data is unavailable.
    mu=[0.7684, -0.4575], sig=[5.02, 5.90] from the PhIRE wind model.
    """
    rng = np.random.default_rng(seed=42)

    try:
        from scipy.ndimage import gaussian_filter
        def _smooth(x, s=6.0): return gaussian_filter(x, sigma=s)
    except ImportError:
        def _smooth(x, s=6.0): return x  # no smoothing fallback

    u_gt = _smooth(rng.normal(0.7684,  5.02455, (H, W)).astype(np.float32))
    v_gt = _smooth(rng.normal(-0.4575, 5.90170, (H, W)).astype(np.float32))
    u_sr = u_gt + rng.normal(0.0, 0.25, (H, W)).astype(np.float32)
    v_sr = v_gt + rng.normal(0.0, 0.25, (H, W)).astype(np.float32)

    gt_t = torch.tensor(np.stack([u_gt, v_gt])[None], device=device)
    sr_t = torch.tensor(np.stack([u_sr, v_sr])[None], device=device)
    return sr_t, gt_t


# ── Safety checks ─────────────────────────────────────────────────────────────

def _check_protected(path: Path, label: str) -> None:
    for p in _PROTECTED:
        try:
            path.resolve().relative_to(p.resolve())
            raise SystemExit(
                f"[error] {label} ({path}) would write inside protected "
                f"directory {p}. Aborting."
            )
        except ValueError:
            pass


# ── Diagnostic ────────────────────────────────────────────────────────────────

def run_diagnostic(
    args,
    model: RefinerNet,
    cubical,
    w_loss_fn,
    sr_t: torch.Tensor,
    gt_t: torch.Tensor,
    is_synthetic: bool,
) -> dict:
    """
    Print full loss breakdown for one sample and recommended lambda_pd values.

    Checks that L_PD gradients propagate to model parameters.
    """
    model.eval()

    with torch.enable_grad():
        # ── Forward ──────────────────────────────────────────────────────
        refined = model(sr_t)

        val_uv    = l_uv(refined, gt_t)
        val_speed = l_speed(refined, gt_t)
        val_grad  = l_grad(refined, gt_t)
        val_crit  = l_crit(refined, gt_t, pool=3, high_z=1.0)

        t0 = time.perf_counter()
        val_pd = l_pd(refined, gt_t, cubical, w_loss_fn, crop=args.pd_crop_size)
        pd_time = time.perf_counter() - t0

        weighted_total = (
            val_uv
            + args.lambda_speed * val_speed
            + args.lambda_grad  * val_grad
            + args.lambda_crit  * val_crit
            + args.lambda_pd    * val_pd
        )

    # ── Print ─────────────────────────────────────────────────────────────
    SEP = "─" * 56
    note = "  [SYNTHETIC DATA]" if is_synthetic else ""
    print(f"\n{SEP}")
    print(f"Candidate D  Loss Diagnostic{note}")
    print(SEP)
    print(f"  {'L_uv':10s} = {val_uv.item():.6f}")
    print(f"  {'L_speed':10s} = {val_speed.item():.6f}  "
          f"λ={args.lambda_speed:.4g}  → {args.lambda_speed * val_speed.item():.6f}")
    print(f"  {'L_grad':10s} = {val_grad.item():.6f}  "
          f"λ={args.lambda_grad:.4g}  → {args.lambda_grad * val_grad.item():.6f}")
    print(f"  {'L_crit':10s} = {val_crit.item():.6f}  "
          f"λ={args.lambda_crit:.4g}  → {args.lambda_crit * val_crit.item():.6f}")
    print(f"  {'L_PD':10s} = {val_pd.item():.6f}  "
          f"λ={args.lambda_pd:.4g}  → {args.lambda_pd * val_pd.item():.6f}  "
          f"[t={pd_time:.3f}s, crop={args.pd_crop_size}²]")
    print(f"  {'L_total':10s} = {weighted_total.item():.6f}")

    ratio = val_pd.item() / max(val_uv.item(), _EPS)
    print(f"\n  L_PD / L_uv      = {ratio:.4f}×")
    for pct in [10, 25, 50]:
        lam_rec = (pct / 100.0) / max(ratio, _EPS)
        print(f"  lambda_pd for {pct:2d}% of L_uv ≈ {lam_rec:.6f}")

    # ── PD gradient check ─────────────────────────────────────────────────
    print(f"\n  Checking L_PD → model param gradient flow …")
    model.zero_grad()
    refined2 = model(sr_t)
    val_pd2 = l_pd(refined2, gt_t, cubical, w_loss_fn, crop=args.pd_crop_size)

    if torch.isnan(val_pd2) or torch.isinf(val_pd2):
        raise RuntimeError("L_PD is NaN/inf. Aborting gradient check.")

    val_pd2.backward()

    has_pd_grad = any(
        p.grad is not None and (p.grad != 0).any()
        for p in model.parameters()
    )
    print(f"  L_PD → model params : {'YES ✓' if has_pd_grad else 'NO — ABORT'}")
    if not has_pd_grad:
        raise RuntimeError(
            "L_PD did not produce gradients to model parameters. "
            "Check the GUDHI patch and ensure torch_topological is correctly installed."
        )
    model.zero_grad()

    print(SEP)

    return {
        "synthetic_data": is_synthetic,
        "L_uv":   val_uv.item(),
        "L_speed": val_speed.item(),
        "L_grad":  val_grad.item(),
        "L_crit":  val_crit.item(),
        "L_PD":    val_pd.item(),
        "L_PD_time_s": round(pd_time, 4),
        "L_PD_over_L_uv": round(ratio, 4),
        "lambda_pd_10pct": round(0.10 / max(ratio, _EPS), 6),
        "lambda_pd_25pct": round(0.25 / max(ratio, _EPS), 6),
        "lambda_pd_50pct": round(0.50 / max(ratio, _EPS), 6),
        "pd_grad_to_model_params": has_pd_grad,
        "pd_crop_size": args.pd_crop_size,
    }


# ── Training epoch ─────────────────────────────────────────────────────────────

def run_epoch(
    epoch: int,
    args,
    model: RefinerNet,
    optimizer: torch.optim.Optimizer,
    cubical,
    w_loss_fn,
    sr_all: np.ndarray,
    gt_all: np.ndarray,
    device: torch.device,
    logger: logging.Logger,
) -> dict:
    model.train()
    N = sr_all.shape[0]
    indices = list(range(N))
    random.shuffle(indices)

    running = {k: 0.0 for k in ("uv", "speed", "grad", "crit", "pd", "total")}
    pd_checked_grad = False if args.lambda_pd > 0 else True  # check once

    for step, i in enumerate(indices):
        sr_t = torch.tensor(sr_all[i:i+1], device=device)  # (1, 2, H, W)
        gt_t = torch.tensor(gt_all[i:i+1], device=device)

        optimizer.zero_grad()

        refined = model(sr_t)

        val_uv    = l_uv(refined, gt_t)
        val_speed = l_speed(refined, gt_t)
        val_grad  = l_grad(refined, gt_t)
        val_crit  = l_crit(refined, gt_t, pool=3, high_z=1.0)

        loss = (
            val_uv
            + args.lambda_speed * val_speed
            + args.lambda_grad  * val_grad
            + args.lambda_crit  * val_crit
        )

        val_pd = torch.zeros((), device=device)
        use_pd = args.lambda_pd > 0.0 and (step % args.pd_every == 0)
        if use_pd:
            val_pd = l_pd(refined, gt_t, cubical, w_loss_fn, crop=args.pd_crop_size)
            if torch.isnan(val_pd) or torch.isinf(val_pd):
                raise RuntimeError(
                    f"L_PD is NaN/inf at epoch {epoch} step {step}. Aborting. "
                    "Try --pd-crop-size 64 or --lambda-pd 0 to debug."
                )
            loss = loss + args.lambda_pd * val_pd

        loss.backward()

        # Verify PD gradients reach model params on first PD step
        if use_pd and not pd_checked_grad:
            has_grad = any(
                p.grad is not None and (p.grad != 0).any()
                for p in model.parameters()
            )
            if not has_grad:
                raise RuntimeError(
                    "lambda_pd > 0 but L_PD produced no gradients to model "
                    "parameters. Check GUDHI patch. Aborting."
                )
            pd_checked_grad = True
            logger.info("[train] PD gradient check passed at epoch %d step %d.", epoch, step)

        optimizer.step()

        running["uv"]    += val_uv.item()
        running["speed"] += val_speed.item()
        running["grad"]  += val_grad.item()
        running["crit"]  += val_crit.item()
        running["pd"]    += val_pd.item()
        running["total"] += loss.item()

        if (step + 1) % 10 == 0 or step == 0:
            logger.info(
                "epoch %d  [%d/%d]  "
                "L_uv=%.4f  L_sp=%.4f  L_gr=%.4f  L_cr=%.4f  L_PD=%.4f  total=%.4f",
                epoch, step + 1, N,
                val_uv.item(), val_speed.item(), val_grad.item(),
                val_crit.item(), val_pd.item(), loss.item(),
            )

    means = {k: v / N for k, v in running.items()}
    logger.info(
        "epoch %d DONE  "
        "mean L_uv=%.5f  L_speed=%.5f  L_grad=%.5f  L_crit=%.5f  L_PD=%.5f  total=%.5f",
        epoch,
        means["uv"], means["speed"], means["grad"], means["crit"], means["pd"], means["total"],
    )
    return means


# ── Output saving ──────────────────────────────────────────────────────────────

def save_outputs(
    args,
    model: RefinerNet,
    sr_all: np.ndarray,
    gt_all: np.ndarray,
    inp_all,
    idx_all,
    device: torch.device,
    logger: logging.Logger,
) -> np.ndarray:
    """Run model inference on all samples and save to out_dir."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_protected(out_dir, "--out-dir")

    model.eval()
    N = sr_all.shape[0]
    refined_all = []

    with torch.no_grad():
        for i in range(N):
            sr_t = torch.tensor(sr_all[i:i+1], device=device)
            ref  = model(sr_t).cpu().numpy()[0]  # (2, H, W)
            refined_all.append(ref)

    refined_arr = np.stack(refined_all, axis=0)  # (N, 2, H, W)
    # Convert to (N, H, W, 2) to match baseline format
    refined_hwc = np.transpose(refined_arr, (0, 2, 3, 1))
    gt_hwc      = np.transpose(gt_all,      (0, 2, 3, 1))

    np.save(out_dir / "dataSR.npy", refined_hwc)
    np.save(out_dir / "dataGT.npy", gt_hwc)

    if inp_all is not None:
        inp_hwc = np.transpose(inp_all, (0, 2, 3, 1)) if inp_all.ndim == 4 and inp_all.shape[1] == 2 else inp_all
        np.save(out_dir / "dataIN.npy", inp_hwc)
    if idx_all is not None:
        np.save(out_dir / "idx.npy", idx_all)

    logger.info("Saved %d refined samples to %s", N, out_dir)
    return refined_arr


# ── Report ─────────────────────────────────────────────────────────────────────

def write_report(args, diag: dict, epoch_logs: list) -> None:
    rpt = Path(args.report_path)
    rpt.parent.mkdir(parents=True, exist_ok=True)

    import datetime
    with open(rpt, "w") as f:
        f.write(f"# Candidate D PD Residual Refiner Notes\n\n")
        f.write(f"**Generated:** {datetime.datetime.now().isoformat(timespec='seconds')}\n\n")

        f.write("## Configuration\n\n```\n")
        for k, v in vars(args).items():
            f.write(f"  {k:<22s} = {v}\n")
        f.write("```\n\n")

        if diag:
            f.write("## Diagnostic Loss Breakdown\n\n")
            note = "*(synthetic data — real data not found)*" if diag.get("synthetic_data") else ""
            f.write(f"{note}\n\n")
            f.write("| Term | Raw value | Weighted |\n")
            f.write("|------|-----------|----------|\n")
            f.write(f"| L_uv    | {diag['L_uv']:.6f} | {diag['L_uv']:.6f} |\n")
            f.write(f"| L_speed | {diag['L_speed']:.6f} | {args.lambda_speed * diag['L_speed']:.6f} |\n")
            f.write(f"| L_grad  | {diag['L_grad']:.6f} | {args.lambda_grad  * diag['L_grad']:.6f} |\n")
            f.write(f"| L_crit  | {diag['L_crit']:.6f} | {args.lambda_crit  * diag['L_crit']:.6f} |\n")
            f.write(f"| L_PD    | {diag['L_PD']:.6f} | {args.lambda_pd    * diag['L_PD']:.6f} "
                    f"*(λ={args.lambda_pd})* |\n\n")
            f.write(f"L_PD / L_uv = **{diag['L_PD_over_L_uv']:.4f}×**  "
                    f"(computed in {diag['L_PD_time_s']:.3f} s, crop={diag['pd_crop_size']}²)\n\n")
            f.write("Recommended lambda_pd:\n\n")
            f.write(f"- 10% of L_uv → `lambda_pd = {diag['lambda_pd_10pct']:.6f}`\n")
            f.write(f"- 25% of L_uv → `lambda_pd = {diag['lambda_pd_25pct']:.6f}`\n")
            f.write(f"- 50% of L_uv → `lambda_pd = {diag['lambda_pd_50pct']:.6f}`\n\n")
            f.write(f"PD gradient to model params: **{'YES' if diag['pd_grad_to_model_params'] else 'NO'}**\n\n")

        if epoch_logs:
            f.write("## Training Loss History\n\n")
            f.write("| Epoch | L_uv | L_speed | L_grad | L_crit | L_PD | L_total |\n")
            f.write("|-------|------|---------|--------|--------|------|---------|\n")
            for ep, m in enumerate(epoch_logs, 1):
                f.write(f"| {ep} | {m['uv']:.5f} | {m['speed']:.5f} | "
                        f"{m['grad']:.5f} | {m['crit']:.5f} | "
                        f"{m['pd']:.5f} | {m['total']:.5f} |\n")
            f.write("\n")

        f.write("## GUDHI Compatibility\n\n")
        f.write("torch_topological 0.1.9 + gudhi 3.12.0 require a one-line patch:\n\n")
        f.write("```\n")
        f.write("# In .venv_candidateD_pd/.../torch_topological/nn/cubical_complex.py\n")
        f.write("# Old: top_dimensional_cells=x.flatten()\n")
        f.write("# New: top_dimensional_cells=x.detach().cpu().numpy().flatten()\n")
        f.write("```\n\n")
        f.write("This script detects and applies the patch automatically at startup.\n")
        f.write("See `docs/candidateD_pd_gradient_smoke.md` for full documentation.\n")

    print(f"\nReport written: {rpt}")


# ── Argument parser ────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Candidate D: PyTorch residual refiner with PD Wasserstein loss.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-dir",        default=str(_DATA_IN),  help="Input data directory (dataSR.npy, dataGT.npy, …)")
    ap.add_argument("--out-dir",         default=str(_OUT_DIR),  help="Output directory for refined SR")
    ap.add_argument("--model-dir",       default=str(_MDL_DIR),  help="Directory to save model checkpoints")
    ap.add_argument("--log-path",        default=str(_LOG_PATH), help="Log file path")
    ap.add_argument("--report-path",     default=str(_RPT_PATH), help="Markdown report path")

    ap.add_argument("--epochs",          type=int,   default=3)
    ap.add_argument("--lr",              type=float, default=1e-4)
    ap.add_argument("--lambda-speed",    type=float, default=0.01)
    ap.add_argument("--lambda-grad",     type=float, default=0.05)
    ap.add_argument("--lambda-crit",     type=float, default=0.001)
    ap.add_argument("--lambda-pd",       type=float, default=0.0,
                    help="PD Wasserstein weight. Use 0 for diagnostic; set after reading L_PD/L_uv.")
    ap.add_argument("--pd-crop-size",    type=int,   default=100,
                    help="Spatial crop size for L_PD computation (100 or 160).")
    ap.add_argument("--pd-every",        type=int,   default=1,
                    help="Compute L_PD every N training steps (1=every step).")
    ap.add_argument("--residual-scale",  type=float, default=0.1,
                    help="Scale applied to residual before adding to input.")
    ap.add_argument("--seed",            type=int,   default=42)

    ap.add_argument("--diagnostic-only", action="store_true",
                    help="Print loss breakdown and exit without training.")
    ap.add_argument("--dry-run",         action="store_true",
                    help="Run 1 epoch on first 3 samples only.")

    return ap.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Logging ──────────────────────────────────────────────────────────────
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("Candidate D PD refiner — args: %s", vars(args))

    # ── Safety checks ─────────────────────────────────────────────────────────
    _check_protected(Path(args.out_dir),    "--out-dir")
    _check_protected(Path(args.model_dir),  "--model-dir")

    # ── GUDHI patch ───────────────────────────────────────────────────────────
    patch_status = _apply_gudhi_compat_patch()
    logger.info("[compat] GUDHI patch status: %s", patch_status)

    _verify_pd_gradients()
    logger.info("[verify] PD gradient check passed.")

    # ── torch_topological imports (after patch) ───────────────────────────────
    from torch_topological.nn import CubicalComplex, WassersteinDistance
    cubical   = CubicalComplex(superlevel=True)
    w_loss_fn = WassersteinDistance(q=2)

    # ── Determinism ───────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cpu")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = RefinerNet(hidden=32, kernel=3, residual_scale=args.residual_scale).to(device)
    logger.info("RefinerNet: %d parameters", model.n_params())

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    is_synthetic = False

    if (data_dir / "dataSR.npy").exists() and (data_dir / "dataGT.npy").exists():
        logger.info("Loading data from %s …", data_dir)
        sr_all, gt_all, inp_all, idx_all = load_data(data_dir)
        logger.info("Loaded: SR=%s  GT=%s", sr_all.shape, gt_all.shape)
    else:
        logger.warning(
            "Data not found at %s. Using SYNTHETIC data for diagnostic/dry-run only.",
            data_dir,
        )
        is_synthetic = True
        # Synthetic: 5 samples, 160×160
        sr_list, gt_list = [], []
        for seed_i in range(5):
            torch.manual_seed(seed_i)
            np.random.seed(seed_i)
            sr_t, gt_t = _synthetic_batch(device, H=160, W=160)
            sr_list.append(sr_t.cpu().numpy()[0])
            gt_list.append(gt_t.cpu().numpy()[0])
        sr_all   = np.stack(sr_list)   # (5, 2, 160, 160)
        gt_all   = np.stack(gt_list)
        inp_all  = sr_all.copy()
        idx_all  = np.arange(5)

    if args.dry_run:
        sr_all  = sr_all[:3]
        gt_all  = gt_all[:3]
        inp_all = inp_all[:3] if inp_all is not None else None
        idx_all = idx_all[:3] if idx_all is not None else None
        args.epochs = 1
        logger.info("[dry-run] Using %d samples, 1 epoch.", len(sr_all))

    # ── Diagnostic ────────────────────────────────────────────────────────────
    sr0 = torch.tensor(sr_all[0:1], device=device)
    gt0 = torch.tensor(gt_all[0:1], device=device)

    diag = run_diagnostic(args, model, cubical, w_loss_fn, sr0, gt0, is_synthetic)

    diag_json = Path(args.out_dir).parent / "candidateD_diagnostic.json"
    diag_json.parent.mkdir(parents=True, exist_ok=True)
    with open(diag_json, "w") as f:
        json.dump({**vars(args), **diag}, f, indent=2)
    logger.info("Diagnostic JSON: %s", diag_json)

    if args.diagnostic_only:
        write_report(args, diag, [])
        logger.info("--diagnostic-only: done.")
        return

    # ── Sanity: lambda_pd + L_PD consistency ──────────────────────────────────
    if args.lambda_pd > 0.0:
        logger.info("lambda_pd=%.4g  L_PD will be computed every %d steps.", args.lambda_pd, args.pd_every)
    else:
        logger.info("lambda_pd=0 — L_PD excluded from training loss (monitoring only via diagnostic).")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── Training ──────────────────────────────────────────────────────────────
    mdl_dir = Path(args.model_dir)
    mdl_dir.mkdir(parents=True, exist_ok=True)
    _check_protected(mdl_dir, "--model-dir")

    epoch_logs = []
    for ep in range(1, args.epochs + 1):
        logger.info("=== Epoch %d / %d ===", ep, args.epochs)
        means = run_epoch(
            ep, args, model, optimizer,
            cubical, w_loss_fn,
            sr_all, gt_all, device, logger,
        )
        epoch_logs.append(means)

        # Save checkpoint
        ckpt = mdl_dir / f"refiner_epoch{ep:02d}.pt"
        torch.save(model.state_dict(), ckpt)
        logger.info("Checkpoint: %s", ckpt)

    # ── Final model ───────────────────────────────────────────────────────────
    final_ckpt = mdl_dir / "refiner_final.pt"
    torch.save(model.state_dict(), final_ckpt)
    logger.info("Final model: %s", final_ckpt)

    # ── Save refined SR ───────────────────────────────────────────────────────
    if not is_synthetic:
        save_outputs(args, model, sr_all, gt_all, inp_all, idx_all, device, logger)
    else:
        logger.warning("Synthetic data: skipping save_outputs (no real CNN SR to refine).")

    # ── Report ────────────────────────────────────────────────────────────────
    write_report(args, diag, epoch_logs)
    logger.info("Done. Report: %s", args.report_path)


if __name__ == "__main__":
    main()
