#!/usr/bin/env python3
"""
Candidate E: PyTorch residual refiner with TTK critical-pair loss (Kissi-style).

Post-processes frozen Candidate C (or CNN baseline) output by training a small
residual CNN that minimises:

    L_total = L_uv
            + lambda_speed * L_speed
            + lambda_grad  * L_grad
            + lambda_crit  * L_crit          (adaptive pool-based max proxy)
            + lambda_ttkcv * L_ttkcv         (MSE at TTK critical vertices)
            + lambda_ttkpers * L_ttkpers     (persistence-gap loss at TTK pairs)

L_ttkcv  — for each GT persistence pair (birth_vid, death_vid), penalises the
           MSE between the SR speed at that vertex and the GT target value.
           This fixes the value at known critical vertices without computing
           full persistence diagrams at training time.

L_ttkpers — penalises the difference between the SR persistence
            (sr_speed[death_vid] - sr_speed[birth_vid]) and the GT persistence.
            Encourages the refiner to preserve topological gap structure.

These constraints come from extract_ttk_pd_critical_pairs.py, which parses GT
VTU files from a TTK pipeline run and stores the critical vertex IDs and target
scalar values in a NPZ file.

Unlike Candidate D (which uses full PD Wasserstein at every training step),
Candidate E requires NO torch_topological or gudhi — it relies only on
indexed MSE at pre-computed vertex locations.

Usage
-----
# Diagnostic only (no training):
python3 scripts/run_candidateE_ttkcrit_refiner.py \\
    --diagnostic-only \\
    --data-dir data_out_fixed/wind_mrhr_cnn \\
    --constraints ttk_runs_fixed/topology_finetuning/candidateE_constraints/ttk_pd_critical_pairs.npz \\
    --lambda-ttkcv 0.0 --lambda-ttkpers 0.0

# Full training (after reviewing diagnostic):
.venv_candidateD_pd/bin/python scripts/run_candidateE_ttkcrit_refiner.py \\
    --data-dir data_out/wind_finetune_pilot_candidateC \\
    --constraints ttk_runs_fixed/topology_finetuning/candidateE_constraints/ttk_pd_critical_pairs.npz \\
    --lambda-ttkcv 1.0 --lambda-ttkpers 0.5 --epochs 3
"""

import argparse
import logging
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

_DATA_IN   = REPO_ROOT / "data_out_fixed" / "wind_mrhr_cnn"
_OUT_DIR   = REPO_ROOT / "data_out"        / "wind_finetune_pilot_candidateE"
_MDL_DIR   = REPO_ROOT / "models_fixed"   / "topology_finetuning" / "wind_finetune_pilot_candidateE"
_LOG_PATH  = REPO_ROOT / "logs"            / "wind_finetune_pilot_candidateE.log"
_RPT_PATH  = REPO_ROOT / "docs"            / "candidateE_ttkcrit_refiner_notes.md"

_PROTECTED = [
    REPO_ROOT / "data_out_fixed" / "wind_mrhr_cnn",
    REPO_ROOT / "data_out_fixed" / "wind_mrhr_gan",
    REPO_ROOT / "data_out"       / "wind_finetune_pilot_candidateB",
    REPO_ROOT / "data_out"       / "wind_finetune_pilot_candidateC",
    REPO_ROOT / "data_out"       / "wind_finetune_pilot_candidateD",
    REPO_ROOT / "models_fixed"   / "topology_finetuning" / "wind_finetune_pilot_candidateB",
    REPO_ROOT / "models_fixed"   / "topology_finetuning" / "wind_finetune_pilot_candidateC",
    REPO_ROOT / "models_fixed"   / "topology_finetuning" / "wind_finetune_pilot_candidateD",
]

_EPS = 1e-8


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


def l_uv(sr: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(sr, gt)


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
    """Critical-value proxy: MSE at GT superlevel-set local maxima."""
    spd_sr = _speed_t(sr)
    spd_gt = _speed_t(gt)

    s4 = spd_gt.unsqueeze(1)
    local_max = F.max_pool2d(s4, kernel_size=pool, stride=1, padding=pool // 2)
    is_local_max = s4 >= local_max - _EPS

    threshold = spd_gt.mean() + high_z * spd_gt.std()
    high_mask = is_local_max.squeeze(1) & (spd_gt >= threshold)

    if not high_mask.any():
        return torch.zeros((), device=sr.device, requires_grad=True)

    return F.mse_loss(spd_sr[high_mask], spd_gt[high_mask])


def l_ttkcv(
    sr: torch.Tensor,
    birth_yx: torch.Tensor,
    death_yx: torch.Tensor,
    birth_target: torch.Tensor,
    death_target: torch.Tensor,
) -> torch.Tensor:
    """
    TTK critical-vertex loss: MSE between SR speed at TTK critical vertex
    locations and GT target scalar values from the GT persistence diagram.

    sr            : (1, 2, H, W) refined SR tensor
    birth_yx      : (K, 2) int tensor — [row, col] coordinates of birth vertices
    death_yx      : (K, 2) int tensor — [row, col] coordinates of death vertices
    birth_target  : (K,) float tensor — GT scalar value at birth vertex
    death_target  : (K,) float tensor — GT scalar value at death vertex
    """
    if birth_yx.shape[0] == 0:
        return torch.zeros((), device=sr.device, requires_grad=True)

    spd = _speed_t(sr)[0]  # (H, W)
    H, W = spd.shape

    # Clamp to valid range
    by = birth_yx[:, 0].clamp(0, H - 1)
    bx = birth_yx[:, 1].clamp(0, W - 1)
    dy = death_yx[:, 0].clamp(0, H - 1)
    dx = death_yx[:, 1].clamp(0, W - 1)

    sr_birth = spd[by, bx]
    sr_death = spd[dy, dx]

    loss_b = F.mse_loss(sr_birth, birth_target)
    loss_d = F.mse_loss(sr_death, death_target)
    return (loss_b + loss_d) * 0.5


def l_ttkpers(
    sr: torch.Tensor,
    birth_yx: torch.Tensor,
    death_yx: torch.Tensor,
    gt_persistence: torch.Tensor,
) -> torch.Tensor:
    """
    TTK persistence-gap loss: penalises the difference between the SR
    persistence (sr[death] - sr[birth]) and the GT persistence value.

    Encourages the refiner to preserve the topological gap structure
    (birth-to-death scalar difference) at TTK-identified critical pairs.
    """
    if birth_yx.shape[0] == 0:
        return torch.zeros((), device=sr.device, requires_grad=True)

    spd = _speed_t(sr)[0]
    H, W = spd.shape

    by = birth_yx[:, 0].clamp(0, H - 1)
    bx = birth_yx[:, 1].clamp(0, W - 1)
    dy = death_yx[:, 0].clamp(0, H - 1)
    dx = death_yx[:, 1].clamp(0, W - 1)

    sr_birth = spd[by, bx]
    sr_death = spd[dy, dx]

    sr_pers = torch.abs(sr_death - sr_birth)  # absolute gap (direction-agnostic)
    return F.mse_loss(sr_pers, gt_persistence)


# ── Constraint loading ─────────────────────────────────────────────────────────

class TTKConstraints:
    """
    Holds pre-computed TTK critical-pair constraints from extract_ttk_pd_critical_pairs.py.

    Provides indexed lookups (by sample position modulo n_samples) so the
    refiner can use constraints even when the number of samples in the data
    does not exactly match the number of VTU files.
    """

    def __init__(self, npz_path: Path, patch: int, device: torch.device) -> None:
        npz = np.load(npz_path, allow_pickle=True)
        self.n_samples    = int(npz["n_samples"])
        self.patch        = int(npz["patch_size"])
        self.sample_start = npz["sample_start"].astype(np.int64)
        self.sample_count = npz["sample_count"].astype(np.int64)
        self.birth_vid    = npz["birth_vid"].astype(np.int64)
        self.death_vid    = npz["death_vid"].astype(np.int64)
        self.birth_val    = npz["birth_val"].astype(np.float32)
        self.death_val    = npz["death_val"].astype(np.float32)
        self.persistence  = npz["persistence"].astype(np.float32)

        self._device = device
        self._patch  = patch

        logging.info(
            "[constraints] Loaded %d samples, %d total pairs from %s",
            self.n_samples, len(self.birth_vid), npz_path,
        )

    def get(self, sample_idx: int):
        """
        Return (birth_yx, death_yx, birth_target, death_target, gt_persistence)
        for sample sample_idx (mod n_samples).

        All returned tensors are on self._device.
        """
        i = sample_idx % self.n_samples
        start = int(self.sample_start[i])
        count = int(self.sample_count[i])

        if count == 0:
            empty_yx   = torch.zeros((0, 2), dtype=torch.long,  device=self._device)
            empty_val  = torch.zeros((0,),   dtype=torch.float32, device=self._device)
            return empty_yx, empty_yx, empty_val, empty_val, empty_val

        bvids = self.birth_vid[start : start + count]
        dvids = self.death_vid[start : start + count]
        bvals = self.birth_val[start : start + count]
        dvals = self.death_val[start : start + count]
        pvals = self.persistence[start : start + count]

        W = self._patch
        birth_yx = torch.tensor(
            np.stack([bvids // W, bvids % W], axis=1), dtype=torch.long, device=self._device
        )
        death_yx = torch.tensor(
            np.stack([dvids // W, dvids % W], axis=1), dtype=torch.long, device=self._device
        )
        return (
            birth_yx,
            death_yx,
            torch.tensor(bvals, dtype=torch.float32, device=self._device),
            torch.tensor(dvals, dtype=torch.float32, device=self._device),
            torch.tensor(pvals, dtype=torch.float32, device=self._device),
        )


def _make_synthetic_constraints(patch: int, device: torch.device) -> "TTKConstraints":
    """
    Build a synthetic TTKConstraints object for diagnostic fallback
    when no real constraints NPZ is available.

    Creates 5 synthetic samples with 8 pairs each using random critical points.
    """
    rng = np.random.default_rng(seed=0)
    n_samples = 5
    k_per = 8

    birth_vid_all = []
    death_vid_all = []
    birth_val_all = []
    death_val_all = []
    pers_all      = []
    starts = []
    counts = []
    offset = 0

    for _ in range(n_samples):
        vids = rng.integers(0, patch * patch, size=k_per * 2, endpoint=False)
        bvids = vids[:k_per]
        dvids = vids[k_per:]
        bvals = rng.uniform(0.5, 3.0, k_per).astype(np.float32)
        pvals = rng.uniform(0.1, 2.0, k_per).astype(np.float32)
        dvals = bvals + pvals

        birth_vid_all.append(bvids)
        death_vid_all.append(dvids)
        birth_val_all.append(bvals)
        death_val_all.append(dvals)
        pers_all.append(pvals)
        starts.append(offset)
        counts.append(k_per)
        offset += k_per

    # Build a minimal NPZ dict and load via a temporary file
    import tempfile, os
    tmpf = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmpf.close()
    np.savez(
        tmpf.name,
        n_samples=n_samples,
        patch_size=patch,
        persistence_frac=0.01,
        top_k=k_per,
        sample_names=np.array([f"synthetic_{i}" for i in range(n_samples)], dtype=object),
        sample_start=np.array(starts, dtype=np.int32),
        sample_count=np.array(counts, dtype=np.int32),
        birth_vid=np.concatenate(birth_vid_all).astype(np.int32),
        death_vid=np.concatenate(death_vid_all).astype(np.int32),
        birth_val=np.concatenate(birth_val_all).astype(np.float32),
        death_val=np.concatenate(death_val_all).astype(np.float32),
        persistence=np.concatenate(pers_all).astype(np.float32),
        pair_type=np.zeros(offset, dtype=np.int32),
    )
    constraints = TTKConstraints(Path(tmpf.name), patch, device)
    os.unlink(tmpf.name)
    return constraints


# ── Data utilities ─────────────────────────────────────────────────────────────

def _to_chw(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 4:
        if a.shape[-1] == 2:
            return np.transpose(a, (0, 3, 1, 2))
        if a.shape[1] == 2:
            return a
    raise ValueError(f"Cannot interpret shape {a.shape} as [u,v] field")


def load_data(data_dir: Path):
    sr  = _to_chw(np.load(data_dir / "dataSR.npy"))
    gt  = _to_chw(np.load(data_dir / "dataGT.npy"))
    inp = _to_chw(np.load(data_dir / "dataIN.npy")) if (data_dir / "dataIN.npy").exists() else None
    idx = np.load(data_dir / "idx.npy")              if (data_dir / "idx.npy").exists()      else None
    return sr, gt, inp, idx


def _synthetic_batch(device: torch.device, H: int = 160, W: int = 160):
    rng = np.random.default_rng(seed=42)
    try:
        from scipy.ndimage import gaussian_filter
        def _smooth(x, s=6.0): return gaussian_filter(x, sigma=s)
    except ImportError:
        def _smooth(x, s=6.0): return x

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
    constraints: TTKConstraints,
    sr_t: torch.Tensor,
    gt_t: torch.Tensor,
    is_synthetic: bool,
    has_real_constraints: bool,
) -> dict:
    """Print full loss breakdown and return summary dict."""
    model.eval()

    birth_yx, death_yx, birth_target, death_target, gt_pers = constraints.get(0)

    with torch.enable_grad():
        refined = model(sr_t)

        val_uv      = l_uv(refined, gt_t)
        val_speed   = l_speed(refined, gt_t)
        val_grad    = l_grad(refined, gt_t)
        val_crit    = l_crit(refined, gt_t, pool=3, high_z=1.0)

        t0 = time.perf_counter()
        val_ttkcv   = l_ttkcv(refined, birth_yx, death_yx, birth_target, death_target)
        val_ttkpers = l_ttkpers(refined, birth_yx, death_yx, gt_pers)
        ttk_time    = time.perf_counter() - t0

        weighted_total = (
            val_uv
            + args.lambda_speed   * val_speed
            + args.lambda_grad    * val_grad
            + args.lambda_crit    * val_crit
            + args.lambda_ttkcv   * val_ttkcv
            + args.lambda_ttkpers * val_ttkpers
        )

    SEP = "─" * 60
    note = "  [SYNTHETIC DATA]" if is_synthetic else ""
    cnote = "  [SYNTHETIC CONSTRAINTS]" if not has_real_constraints else ""
    print(f"\n{SEP}")
    print(f"Candidate E  TTK Critical-Pair Loss Diagnostic{note}{cnote}")
    print(SEP)
    print(f"  {'L_uv':14s} = {val_uv.item():.6f}")
    print(f"  {'L_speed':14s} = {val_speed.item():.6f}  "
          f"λ={args.lambda_speed:.4g}  → {args.lambda_speed * val_speed.item():.6f}")
    print(f"  {'L_grad':14s} = {val_grad.item():.6f}  "
          f"λ={args.lambda_grad:.4g}  → {args.lambda_grad * val_grad.item():.6f}")
    print(f"  {'L_crit':14s} = {val_crit.item():.6f}  "
          f"λ={args.lambda_crit:.4g}  → {args.lambda_crit * val_crit.item():.6f}")
    print(f"  {'L_ttkcv':14s} = {val_ttkcv.item():.6f}  "
          f"λ={args.lambda_ttkcv:.4g}  → {args.lambda_ttkcv * val_ttkcv.item():.6f}  "
          f"[{birth_yx.shape[0]} pairs, t={ttk_time*1000:.2f}ms]")
    print(f"  {'L_ttkpers':14s} = {val_ttkpers.item():.6f}  "
          f"λ={args.lambda_ttkpers:.4g}  → {args.lambda_ttkpers * val_ttkpers.item():.6f}")
    print(f"  {'L_total':14s} = {weighted_total.item():.6f}")

    ratio_cv   = val_ttkcv.item()   / max(val_uv.item(), _EPS)
    ratio_pers = val_ttkpers.item() / max(val_uv.item(), _EPS)
    print(f"\n  L_ttkcv   / L_uv = {ratio_cv:.4f}×")
    print(f"  L_ttkpers / L_uv = {ratio_pers:.4f}×")
    for pct in [10, 25, 50]:
        lam_cv   = (pct / 100.0) / max(ratio_cv,   _EPS)
        lam_pers = (pct / 100.0) / max(ratio_pers, _EPS)
        print(f"  lambda_ttkcv   for {pct:2d}% of L_uv ≈ {lam_cv:.6f}")
        print(f"  lambda_ttkpers for {pct:2d}% of L_uv ≈ {lam_pers:.6f}")

    # Gradient check for TTK losses
    print(f"\n  Checking L_ttkcv → model param gradient flow …")
    model.zero_grad()
    refined2 = model(sr_t)
    val_cv2 = l_ttkcv(refined2, birth_yx, death_yx, birth_target, death_target)
    val_cv2.backward()
    has_cv_grad = any(
        p.grad is not None and (p.grad != 0).any()
        for p in model.parameters()
    )
    print(f"  L_ttkcv → model params : {'YES ✓' if has_cv_grad else 'NO — check loss computation'}")
    model.zero_grad()

    print(SEP)

    return {
        "synthetic_data":        is_synthetic,
        "has_real_constraints":  has_real_constraints,
        "n_constraints":         int(birth_yx.shape[0]),
        "L_uv":                  val_uv.item(),
        "L_speed":               val_speed.item(),
        "L_grad":                val_grad.item(),
        "L_crit":                val_crit.item(),
        "L_ttkcv":               val_ttkcv.item(),
        "L_ttkpers":             val_ttkpers.item(),
        "L_ttkcv_over_L_uv":     round(ratio_cv,   4),
        "L_ttkpers_over_L_uv":   round(ratio_pers, 4),
        "ttk_time_ms":           round(ttk_time * 1000, 3),
        "ttkcv_grad_to_model":   has_cv_grad,
    }


# ── Training epoch ─────────────────────────────────────────────────────────────

def run_epoch(
    epoch: int,
    args,
    model: RefinerNet,
    optimizer: torch.optim.Optimizer,
    constraints: TTKConstraints,
    sr_all: np.ndarray,
    gt_all: np.ndarray,
    device: torch.device,
    logger: logging.Logger,
) -> dict:
    model.train()
    N = sr_all.shape[0]
    indices = list(range(N))
    random.shuffle(indices)

    running = {k: 0.0 for k in ("uv", "speed", "grad", "crit", "ttkcv", "ttkpers", "total")}

    for step, i in enumerate(indices):
        sr_t = torch.tensor(sr_all[i : i + 1], device=device)
        gt_t = torch.tensor(gt_all[i : i + 1], device=device)

        birth_yx, death_yx, birth_target, death_target, gt_pers = constraints.get(i)

        optimizer.zero_grad()
        refined = model(sr_t)

        val_uv      = l_uv(refined, gt_t)
        val_speed   = l_speed(refined, gt_t)
        val_grad    = l_grad(refined, gt_t)
        val_crit    = l_crit(refined, gt_t, pool=3, high_z=1.0)
        val_ttkcv   = l_ttkcv(refined, birth_yx, death_yx, birth_target, death_target)
        val_ttkpers = l_ttkpers(refined, birth_yx, death_yx, gt_pers)

        loss = (
            val_uv
            + args.lambda_speed   * val_speed
            + args.lambda_grad    * val_grad
            + args.lambda_crit    * val_crit
            + args.lambda_ttkcv   * val_ttkcv
            + args.lambda_ttkpers * val_ttkpers
        )

        loss.backward()
        optimizer.step()

        running["uv"]      += val_uv.item()
        running["speed"]   += val_speed.item()
        running["grad"]    += val_grad.item()
        running["crit"]    += val_crit.item()
        running["ttkcv"]   += val_ttkcv.item()
        running["ttkpers"] += val_ttkpers.item()
        running["total"]   += loss.item()

        if (step + 1) % 10 == 0 or step == 0:
            logger.info(
                "epoch %d  [%d/%d]  "
                "L_uv=%.4f  L_sp=%.4f  L_gr=%.4f  L_cr=%.4f  "
                "L_cv=%.4f  L_pers=%.4f  total=%.4f",
                epoch, step + 1, N,
                val_uv.item(), val_speed.item(), val_grad.item(), val_crit.item(),
                val_ttkcv.item(), val_ttkpers.item(), loss.item(),
            )

    means = {k: v / N for k, v in running.items()}
    logger.info(
        "epoch %d DONE  "
        "mean L_uv=%.5f  L_speed=%.5f  L_grad=%.5f  L_crit=%.5f  "
        "L_ttkcv=%.5f  L_ttkpers=%.5f  total=%.5f",
        epoch,
        means["uv"], means["speed"], means["grad"], means["crit"],
        means["ttkcv"], means["ttkpers"], means["total"],
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_protected(out_dir, "--out-dir")

    model.eval()
    N = sr_all.shape[0]
    refined_all = []

    with torch.no_grad():
        for i in range(N):
            sr_t = torch.tensor(sr_all[i : i + 1], device=device)
            ref  = model(sr_t).cpu().numpy()[0]
            refined_all.append(ref)

    refined_arr = np.stack(refined_all, axis=0)
    refined_hwc = np.transpose(refined_arr, (0, 2, 3, 1))
    gt_hwc      = np.transpose(gt_all,     (0, 2, 3, 1))

    np.save(out_dir / "dataSR.npy", refined_hwc)
    np.save(out_dir / "dataGT.npy", gt_hwc)

    if inp_all is not None:
        inp_hwc = (
            np.transpose(inp_all, (0, 2, 3, 1))
            if inp_all.ndim == 4 and inp_all.shape[1] == 2
            else inp_all
        )
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
        f.write("# Candidate E TTK Critical-Pair Refiner Notes\n\n")
        f.write(f"**Generated:** {datetime.datetime.now().isoformat(timespec='seconds')}\n\n")

        f.write("## Configuration\n\n```\n")
        for k, v in vars(args).items():
            f.write(f"  {k:<26s} = {v}\n")
        f.write("```\n\n")

        if diag:
            note = "*(synthetic data)*" if diag.get("synthetic_data") else ""
            cnote = " *(synthetic constraints)*" if not diag.get("has_real_constraints") else ""
            f.write(f"## Diagnostic Loss Breakdown {note}{cnote}\n\n")
            f.write(f"Constraints: {diag['n_constraints']} pairs per sample\n\n")
            f.write("| Term | Raw value | Weighted |\n")
            f.write("|------|-----------|----------|\n")
            f.write(f"| L_uv      | {diag['L_uv']:.6f} | {diag['L_uv']:.6f} |\n")
            f.write(f"| L_speed   | {diag['L_speed']:.6f} | {args.lambda_speed * diag['L_speed']:.6f} |\n")
            f.write(f"| L_grad    | {diag['L_grad']:.6f} | {args.lambda_grad  * diag['L_grad']:.6f} |\n")
            f.write(f"| L_crit    | {diag['L_crit']:.6f} | {args.lambda_crit  * diag['L_crit']:.6f} |\n")
            f.write(f"| L_ttkcv   | {diag['L_ttkcv']:.6f} | {args.lambda_ttkcv * diag['L_ttkcv']:.6f} |\n")
            f.write(f"| L_ttkpers | {diag['L_ttkpers']:.6f} | {args.lambda_ttkpers * diag['L_ttkpers']:.6f} |\n\n")
            f.write(f"L_ttkcv / L_uv   = **{diag['L_ttkcv_over_L_uv']:.4f}×**\n")
            f.write(f"L_ttkpers / L_uv = **{diag['L_ttkpers_over_L_uv']:.4f}×**\n")
            f.write(f"TTK loss time    = {diag['ttk_time_ms']:.2f} ms (vs ~100 ms for L_PD at 100×100)\n\n")
            f.write(f"L_ttkcv → model params: **{'YES' if diag['ttkcv_grad_to_model'] else 'NO'}**\n\n")

        if epoch_logs:
            f.write("## Training Loss History\n\n")
            f.write("| Epoch | L_uv | L_speed | L_grad | L_crit | L_ttkcv | L_ttkpers | L_total |\n")
            f.write("|-------|------|---------|--------|--------|---------|-----------|--------|\n")
            for ep, m in enumerate(epoch_logs, 1):
                f.write(
                    f"| {ep} | {m['uv']:.5f} | {m['speed']:.5f} | "
                    f"{m['grad']:.5f} | {m['crit']:.5f} | "
                    f"{m['ttkcv']:.5f} | {m['ttkpers']:.5f} | {m['total']:.5f} |\n"
                )
            f.write("\n")

    print(f"\nReport written: {rpt}")


# ── Argument parser ────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Candidate E: PyTorch residual refiner with TTK critical-pair loss.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-dir",        default=str(_DATA_IN),  help="Input data directory")
    ap.add_argument("--constraints",     default="",             help="Path to ttk_pd_critical_pairs.npz")
    ap.add_argument("--out-dir",         default=str(_OUT_DIR),  help="Output directory for refined SR")
    ap.add_argument("--model-dir",       default=str(_MDL_DIR),  help="Directory to save model checkpoints")
    ap.add_argument("--log-path",        default=str(_LOG_PATH), help="Log file path")
    ap.add_argument("--report-path",     default=str(_RPT_PATH), help="Markdown report path")

    ap.add_argument("--epochs",           type=int,   default=3)
    ap.add_argument("--lr",               type=float, default=1e-4)
    ap.add_argument("--lambda-speed",     type=float, default=0.01)
    ap.add_argument("--lambda-grad",      type=float, default=0.05)
    ap.add_argument("--lambda-crit",      type=float, default=0.001)
    ap.add_argument("--lambda-ttkcv",     type=float, default=1.0,
                    help="Weight for TTK critical-vertex MSE loss.")
    ap.add_argument("--lambda-ttkpers",   type=float, default=0.5,
                    help="Weight for TTK persistence-gap loss.")
    ap.add_argument("--residual-scale",   type=float, default=0.1)
    ap.add_argument("--seed",             type=int,   default=42)

    ap.add_argument("--diagnostic-only",  action="store_true",
                    help="Print loss breakdown and exit without training.")
    return ap.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Logging ───────────────────────────────────────────────────────────────
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
    )
    logger = logging.getLogger("candidateE")

    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Safety guard ─────────────────────────────────────────────────────────
    if not args.diagnostic_only:
        _check_protected(Path(args.out_dir), "--out-dir")
        _check_protected(Path(args.model_dir), "--model-dir")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = RefinerNet(hidden=32, kernel=3, residual_scale=args.residual_scale).to(device)
    logger.info("RefinerNet: %d parameters", model.n_params())

    # ── Constraints ───────────────────────────────────────────────────────────
    has_real_constraints = False
    constraints_path = Path(args.constraints) if args.constraints else None

    if constraints_path and constraints_path.exists():
        try:
            constraints = TTKConstraints(constraints_path, patch=160, device=device)
            has_real_constraints = True
            logger.info("Loaded real TTK constraints: %s", constraints_path)
        except Exception as exc:
            logger.warning("Failed to load constraints from %s: %s", constraints_path, exc)
            logger.warning("Falling back to synthetic constraints.")
            constraints = _make_synthetic_constraints(patch=160, device=device)
    else:
        if args.constraints:
            logger.warning(
                "Constraints file not found: %s  (falling back to synthetic)",
                args.constraints,
            )
        else:
            logger.info("No --constraints given; using synthetic constraints.")
        constraints = _make_synthetic_constraints(patch=160, device=device)

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    is_synthetic = False

    if (data_dir / "dataSR.npy").exists() and (data_dir / "dataGT.npy").exists():
        logger.info("Loading real data from %s …", data_dir)
        sr_all, gt_all, inp_all, idx_all = load_data(data_dir)
        logger.info(
            "Data: SR=%s  GT=%s", sr_all.shape, gt_all.shape
        )
        # Use first sample for diagnostic
        sr_t = torch.tensor(sr_all[:1], device=device)
        gt_t = torch.tensor(gt_all[:1], device=device)
    else:
        logger.warning("Data not found at %s — using synthetic batch.", data_dir)
        is_synthetic = True
        sr_t, gt_t = _synthetic_batch(device)
        sr_all = sr_t.cpu().numpy()
        gt_all = gt_t.cpu().numpy()
        inp_all = None
        idx_all = None

    # ── Diagnostic ────────────────────────────────────────────────────────────
    diag = run_diagnostic(
        args, model, constraints, sr_t, gt_t, is_synthetic, has_real_constraints
    )

    if args.diagnostic_only:
        write_report(args, diag, [])
        return

    # ── Training ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epoch_logs = []

    for epoch in range(1, args.epochs + 1):
        logger.info("=== Epoch %d / %d ===", epoch, args.epochs)
        means = run_epoch(
            epoch, args, model, optimizer, constraints,
            sr_all, gt_all, device, logger,
        )
        epoch_logs.append(means)

        # Save checkpoint
        ckpt_dir = Path(args.model_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        _check_protected(ckpt_dir, "--model-dir")
        ckpt_path = ckpt_dir / f"refiner_epoch{epoch}.pt"
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "args": vars(args)}, ckpt_path)
        logger.info("Checkpoint: %s", ckpt_path)

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_outputs(args, model, sr_all, gt_all, inp_all, idx_all, device, logger)

    # ── Report ────────────────────────────────────────────────────────────────
    write_report(args, diag, epoch_logs)


if __name__ == "__main__":
    main()
