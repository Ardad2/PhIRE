#!/usr/bin/env python3
"""
Candidate E: PyTorch residual refiner with TTK critical-pair loss (Kissi-style).

Candidate E = Candidate C losses + Candidate B level-set loss + TTK critical-pair losses.

Post-processes frozen CNN-baseline output by training a small residual CNN that
minimises:

    L_total = L_uv
            + lambda_speed    * L_speed
            + lambda_grad     * L_grad
            + lambda_crit     * L_crit           (Candidate C pool-3 maxima proxy)
            + lambda_levelset * L_levelset        (Candidate B level-set at 5/10/15 m/s)
            + lambda_ttkcv    * L_ttkcv           (MSE at TTK critical vertices)
            + lambda_ttkpers  * L_ttkpers         (persistence-gap loss at TTK pairs)

TTK critical-pair losses (Kissi-style)
---------------------------------------
Constraints come from extract_ttk_pd_critical_pairs.py (NPZ).  For each GT
persistence pair, the NPZ stores:
  - birth_vid, death_vid   : flat vertex indices in the H×W speed grid
  - birth_val, death_val   : GT scalar speed values (death_val = birth_val + persistence)
  - persistence            : |death_val - birth_val|  (unsigned, as stored by TTK)

L_ttkcv: MSE between SR speed at critical vertices and GT target values.
    loss = 0.5 * (MSE(sr[birth_yx], birth_val) + MSE(sr[death_yx], death_val))

L_ttkpers: MSE between |SR_death - SR_birth| and GT persistence.
    sr_pers = |sr_speed[death_yx] - sr_speed[birth_yx]|
    loss = MSE(sr_pers, gt_persistence)

Sign convention: persistence is stored unsigned (|death - birth|) by TTK.
The ordering of birth and death vertices depends on the filtration direction
and is not assumed here. Both L_ttkcv and L_ttkpers are direction-agnostic.

Unlike Candidate D, no torch_topological or gudhi is required.
TTK loss overhead: ~1.7 ms/sample vs ~100 ms for Candidate D's L_PD.

Usage
-----
# Step 1 — diagnostic (no training, use lambda=0):
.venv_candidateD_pd/bin/python scripts/run_candidateE_ttkcrit_refiner.py \\
    --diagnostic-only \\
    --data-dir data_out_fixed/wind_mrhr_cnn \\
    --constraints ttk_runs_fixed/topology_finetuning/candidateE_constraints/ttk_pd_critical_pairs.npz \\
    --lambda-ttkcv 0.0 --lambda-ttkpers 0.0

# Step 2 — training (after reviewing ratios from diagnostic):
.venv_candidateD_pd/bin/python scripts/run_candidateE_ttkcrit_refiner.py \\
    --data-dir data_out_fixed/wind_mrhr_cnn \\
    --constraints ttk_runs_fixed/topology_finetuning/candidateE_constraints/ttk_pd_critical_pairs.npz \\
    --lambda-ttkcv 0.001 --lambda-ttkpers 0.005 --epochs 3
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

    Output ≈ input at initialisation (final conv layer is zero-initialized).
        refined = input + residual_scale * body(input)
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
    """(B, H, W) → (B, H, W) gradient magnitude via finite differences."""
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
    """
    Candidate C critical-value proxy: MSE at GT superlevel-set local maxima.

    Local maxima are detected via max-pool (pool×pool) and thresholded at
    mean + high_z * std of GT speed. Matches Candidate C convention.
    """
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


def l_levelset(
    sr: torch.Tensor,
    gt: torch.Tensor,
    thresholds: tuple[float, ...] = (5.0, 10.0, 15.0),
    k: float = 10.0,
) -> torch.Tensor:
    """
    Candidate B level-set loss: MSE on soft binary exceedance maps at each
    wind-speed threshold.  Uses sigmoid(k * (speed - threshold)) to
    approximate the step function differentiably.
    """
    spd_sr = _speed_t(sr)   # (B, H, W)
    spd_gt = _speed_t(gt)

    total: torch.Tensor | None = None
    for t in thresholds:
        t_tensor = torch.tensor(t, device=sr.device, dtype=sr.dtype)
        ls_sr = torch.sigmoid(k * (spd_sr - t_tensor))
        ls_gt = torch.sigmoid(k * (spd_gt - t_tensor))
        term = F.mse_loss(ls_sr, ls_gt)
        total = term if total is None else total + term

    return total / len(thresholds) if total is not None else torch.zeros((), device=sr.device, requires_grad=True)


def l_ttkcv(
    sr: torch.Tensor,
    birth_yx: torch.Tensor,
    death_yx: torch.Tensor,
    birth_target: torch.Tensor,
    death_target: torch.Tensor,
) -> torch.Tensor:
    """
    TTK critical-vertex loss.

    For each GT persistence pair, penalises the MSE between the SR speed at
    the birth/death vertex locations and the GT target scalar values.

    sr            : (1, 2, H, W)
    birth_yx      : (K, 2) int64 — [row, col] coordinates of birth vertices
    death_yx      : (K, 2) int64 — [row, col] coordinates of death vertices
    birth_target  : (K,) float32 — GT scalar speed at birth vertex
    death_target  : (K,) float32 — GT scalar speed at death vertex

    persistence = |death_target - birth_target| (unsigned, TTK convention).
    """
    if birth_yx.shape[0] == 0:
        return torch.zeros((), device=sr.device, requires_grad=True)

    spd = _speed_t(sr)[0]   # (H, W)
    H, W = spd.shape

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
    TTK persistence-gap loss.

    For each GT persistence pair, penalises the MSE between the SR persistence
    (|sr[death] - sr[birth]|) and the GT persistence value.

    Sign convention: gt_persistence = |birth_val - death_val| (unsigned) as
    extracted from TTK VTU files where persistence = abs(death - birth).
    SR persistence is also computed unsigned: |sr_death - sr_birth|.
    This is direction-agnostic and matches the TTK storage convention.
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

    sr_pers = torch.abs(sr_death - sr_birth)
    return F.mse_loss(sr_pers, gt_persistence)


# ── Constraint loading ─────────────────────────────────────────────────────────

class TTKConstraints:
    """
    Holds pre-computed TTK critical-pair constraints.

    Provides O(1) lookup by sample_idx via a dict built from the NPZ
    sample_idx array.  Aborts if the NPZ is missing required keys.
    """

    def __init__(self, npz_path: Path, patch: int, device: torch.device) -> None:
        npz = np.load(npz_path, allow_pickle=True)

        required = {"n_samples", "sample_idx", "sample_start", "sample_count",
                    "birth_vid", "death_vid", "birth_val", "death_val", "persistence"}
        missing = required - set(npz.files)
        if missing:
            raise ValueError(
                f"NPZ missing required keys: {missing}. "
                "Re-run extract_ttk_pd_critical_pairs.py to regenerate."
            )

        self.n_samples    = int(npz["n_samples"])
        self.patch        = patch
        self.sample_idx   = npz["sample_idx"].astype(np.int64)
        self.sample_start = npz["sample_start"].astype(np.int64)
        self.sample_count = npz["sample_count"].astype(np.int64)
        self.birth_vid    = npz["birth_vid"].astype(np.int64)
        self.death_vid    = npz["death_vid"].astype(np.int64)
        self.birth_val    = npz["birth_val"].astype(np.float32)
        self.death_val    = npz["death_val"].astype(np.float32)
        self.persistence  = npz["persistence"].astype(np.float32)

        # Build sample_idx -> row-in-NPZ lookup dict
        self._idx_to_row: dict[int, int] = {
            int(self.sample_idx[i]): i for i in range(len(self.sample_idx))
        }
        self._device = device
        self._W = patch

        logging.info(
            "[constraints] %d samples, %d total pairs. sample_idx range [%d, %d]",
            self.n_samples, len(self.birth_vid),
            int(self.sample_idx.min()), int(self.sample_idx.max()),
        )

    def get(self, sample_idx: int):
        """
        Return (birth_yx, death_yx, birth_target, death_target, gt_persistence)
        for the given sample_idx.

        Raises KeyError if sample_idx is not in the NPZ.
        """
        if sample_idx not in self._idx_to_row:
            raise KeyError(
                f"sample_idx={sample_idx} not found in constraints NPZ. "
                f"Available indices: {sorted(self._idx_to_row)[:10]}…"
            )
        row = self._idx_to_row[sample_idx]
        start = int(self.sample_start[row])
        count = int(self.sample_count[row])

        if count == 0:
            empty_yx  = torch.zeros((0, 2), dtype=torch.long,    device=self._device)
            empty_val = torch.zeros((0,),   dtype=torch.float32, device=self._device)
            return empty_yx, empty_yx, empty_val, empty_val, empty_val

        W = self._W
        bvids = self.birth_vid[start : start + count]
        dvids = self.death_vid[start : start + count]
        bvals = self.birth_val[start : start + count]
        dvals = self.death_val[start : start + count]
        pvals = self.persistence[start : start + count]

        birth_yx = torch.tensor(
            np.stack([bvids // W, bvids % W], axis=1),
            dtype=torch.long, device=self._device,
        )
        death_yx = torch.tensor(
            np.stack([dvids // W, dvids % W], axis=1),
            dtype=torch.long, device=self._device,
        )
        return (
            birth_yx,
            death_yx,
            torch.tensor(bvals, dtype=torch.float32, device=self._device),
            torch.tensor(dvals, dtype=torch.float32, device=self._device),
            torch.tensor(pvals, dtype=torch.float32, device=self._device),
        )

    @property
    def available_indices(self) -> set[int]:
        return set(self._idx_to_row.keys())


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
    """Load SR, GT, optional IN/idx arrays. Aborts if files are missing."""
    sr_path = data_dir / "dataSR.npy"
    gt_path = data_dir / "dataGT.npy"
    if not sr_path.exists() or not gt_path.exists():
        logging.error(
            "Required data files not found in %s.\n"
            "Expected: dataSR.npy, dataGT.npy\n"
            "Run the CNN baseline or Candidate C pipeline to generate these files.",
            data_dir,
        )
        sys.exit(1)

    sr  = _to_chw(np.load(sr_path))
    gt  = _to_chw(np.load(gt_path))
    inp = _to_chw(np.load(data_dir / "dataIN.npy"))  if (data_dir / "dataIN.npy").exists()  else None
    idx = np.load(data_dir / "idx.npy")               if (data_dir / "idx.npy").exists()       else None
    return sr, gt, inp, idx


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


def _abort_nan(val: torch.Tensor, name: str) -> None:
    if torch.isnan(val) or torch.isinf(val):
        raise RuntimeError(
            f"Loss {name} is NaN/inf ({val.item():.6g}). "
            "Check input data range and constraint validity. Aborting."
        )


# ── Diagnostic ────────────────────────────────────────────────────────────────

def run_diagnostic(
    args,
    model: RefinerNet,
    constraints: TTKConstraints,
    sr_t: torch.Tensor,
    gt_t: torch.Tensor,
    first_sample_id: int,
    is_real_data: bool,
    is_real_constraints: bool,
) -> dict:
    """Print full loss breakdown and return summary dict."""
    model.eval()

    birth_yx, death_yx, birth_target, death_target, gt_pers = constraints.get(first_sample_id)

    with torch.enable_grad():
        refined = model(sr_t)

        val_uv         = l_uv(refined, gt_t)
        val_speed      = l_speed(refined, gt_t)
        val_grad       = l_grad(refined, gt_t)
        val_crit       = l_crit(refined, gt_t, pool=3, high_z=1.0)
        val_levelset   = l_levelset(refined, gt_t, thresholds=(5.0, 10.0, 15.0), k=10.0)

        t0 = time.perf_counter()
        val_ttkcv      = l_ttkcv(refined, birth_yx, death_yx, birth_target, death_target)
        val_ttkpers    = l_ttkpers(refined, birth_yx, death_yx, gt_pers)
        ttk_time       = time.perf_counter() - t0

        weighted_total = (
            val_uv
            + args.lambda_speed    * val_speed
            + args.lambda_grad     * val_grad
            + args.lambda_crit     * val_crit
            + args.lambda_levelset * val_levelset
            + args.lambda_ttkcv    * val_ttkcv
            + args.lambda_ttkpers  * val_ttkpers
        )

    for name, val in [
        ("L_uv", val_uv), ("L_speed", val_speed), ("L_grad", val_grad),
        ("L_crit", val_crit), ("L_levelset", val_levelset),
        ("L_ttkcv", val_ttkcv), ("L_ttkpers", val_ttkpers),
    ]:
        _abort_nan(val, name)

    SEP = "─" * 64
    print(f"\n{SEP}")
    print("Candidate E  TTK Critical-Pair Loss Diagnostic")
    print(SEP)
    print(f"  Real data          : {'YES' if is_real_data else 'NO'}")
    print(f"  Real constraints   : {'YES' if is_real_constraints else 'NO'}")
    print(f"  Constraint samples : {constraints.n_samples}")
    print(f"  First sample id    : {first_sample_id}")
    print(f"  Pairs for sample   : {birth_yx.shape[0]}")
    print(SEP)
    print(f"  {'L_uv':16s} = {val_uv.item():.6f}")
    print(f"  {'L_speed':16s} = {val_speed.item():.6f}  "
          f"λ={args.lambda_speed:.4g}  → {args.lambda_speed * val_speed.item():.6f}")
    print(f"  {'L_grad':16s} = {val_grad.item():.6f}  "
          f"λ={args.lambda_grad:.4g}  → {args.lambda_grad * val_grad.item():.6f}")
    print(f"  {'L_crit':16s} = {val_crit.item():.6f}  "
          f"λ={args.lambda_crit:.4g}  → {args.lambda_crit * val_crit.item():.6f}")
    print(f"  {'L_levelset':16s} = {val_levelset.item():.6f}  "
          f"λ={args.lambda_levelset:.4g}  → {args.lambda_levelset * val_levelset.item():.6f}")
    print(f"  {'L_ttkcv':16s} = {val_ttkcv.item():.6f}  "
          f"λ={args.lambda_ttkcv:.4g}  → {args.lambda_ttkcv * val_ttkcv.item():.6f}  "
          f"[{birth_yx.shape[0]} pairs, t={ttk_time*1000:.2f}ms]")
    print(f"  {'L_ttkpers':16s} = {val_ttkpers.item():.6f}  "
          f"λ={args.lambda_ttkpers:.4g}  → {args.lambda_ttkpers * val_ttkpers.item():.6f}")
    print(f"  {'L_total':16s} = {weighted_total.item():.6f}")

    ratio_cv   = val_ttkcv.item()   / max(val_uv.item(), _EPS)
    ratio_pers = val_ttkpers.item() / max(val_uv.item(), _EPS)
    print(f"\n  L_ttkcv   / L_uv = {ratio_cv:.4f}×")
    print(f"  L_ttkpers / L_uv = {ratio_pers:.4f}×")
    for pct in [10, 25, 50]:
        lam_cv   = (pct / 100.0) / max(ratio_cv,   _EPS)
        lam_pers = (pct / 100.0) / max(ratio_pers, _EPS)
        print(f"  lambda_ttkcv   for {pct:2d}% of L_uv ≈ {lam_cv:.6f}")
        print(f"  lambda_ttkpers for {pct:2d}% of L_uv ≈ {lam_pers:.6f}")

    # L_ttkcv gradient check
    print(f"\n  Checking L_ttkcv → model param gradient flow …")
    model.zero_grad()
    refined2 = model(sr_t)
    val_cv2 = l_ttkcv(refined2, birth_yx, death_yx, birth_target, death_target)
    _abort_nan(val_cv2, "L_ttkcv (grad check)")
    val_cv2.backward()
    has_cv_grad = any(
        p.grad is not None and (p.grad != 0).any()
        for p in model.parameters()
    )
    print(f"  L_ttkcv → model params : {'YES ✓' if has_cv_grad else 'NO — check loss computation'}")
    model.zero_grad()
    print(SEP)

    return {
        "is_real_data":          is_real_data,
        "is_real_constraints":   is_real_constraints,
        "n_constraint_samples":  constraints.n_samples,
        "first_sample_id":       first_sample_id,
        "n_pairs_used":          int(birth_yx.shape[0]),
        "L_uv":                  val_uv.item(),
        "L_speed":               val_speed.item(),
        "L_grad":                val_grad.item(),
        "L_crit":                val_crit.item(),
        "L_levelset":            val_levelset.item(),
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
    idx_all,
    device: torch.device,
    logger: logging.Logger,
) -> dict:
    model.train()
    N = sr_all.shape[0]
    indices = list(range(N))
    random.shuffle(indices)

    running = {k: 0.0 for k in ("uv", "speed", "grad", "crit", "levelset", "ttkcv", "ttkpers", "total")}

    for step, i in enumerate(indices):
        sr_t = torch.tensor(sr_all[i : i + 1], device=device)
        gt_t = torch.tensor(gt_all[i : i + 1], device=device)

        # Resolve actual sample id from idx.npy if available
        actual_id = int(idx_all[i]) if idx_all is not None else i

        try:
            birth_yx, death_yx, birth_target, death_target, gt_pers = constraints.get(actual_id)
        except KeyError as exc:
            logger.warning("epoch %d step %d: %s — using empty constraints.", epoch, step, exc)
            birth_yx = death_yx = torch.zeros((0, 2), dtype=torch.long, device=device)
            birth_target = death_target = gt_pers = torch.zeros((0,), device=device)

        optimizer.zero_grad()
        refined = model(sr_t)

        val_uv         = l_uv(refined, gt_t)
        val_speed      = l_speed(refined, gt_t)
        val_grad       = l_grad(refined, gt_t)
        val_crit       = l_crit(refined, gt_t, pool=3, high_z=1.0)
        val_levelset   = l_levelset(refined, gt_t)
        val_ttkcv      = l_ttkcv(refined, birth_yx, death_yx, birth_target, death_target)
        val_ttkpers    = l_ttkpers(refined, birth_yx, death_yx, gt_pers)

        for name, val in [
            ("L_uv", val_uv), ("L_speed", val_speed), ("L_grad", val_grad),
            ("L_crit", val_crit), ("L_levelset", val_levelset),
            ("L_ttkcv", val_ttkcv), ("L_ttkpers", val_ttkpers),
        ]:
            _abort_nan(val, f"{name} at epoch {epoch} step {step}")

        loss = (
            val_uv
            + args.lambda_speed    * val_speed
            + args.lambda_grad     * val_grad
            + args.lambda_crit     * val_crit
            + args.lambda_levelset * val_levelset
            + args.lambda_ttkcv    * val_ttkcv
            + args.lambda_ttkpers  * val_ttkpers
        )
        _abort_nan(loss, f"L_total at epoch {epoch} step {step}")

        loss.backward()
        optimizer.step()

        running["uv"]       += val_uv.item()
        running["speed"]    += val_speed.item()
        running["grad"]     += val_grad.item()
        running["crit"]     += val_crit.item()
        running["levelset"] += val_levelset.item()
        running["ttkcv"]    += val_ttkcv.item()
        running["ttkpers"]  += val_ttkpers.item()
        running["total"]    += loss.item()

        if (step + 1) % 10 == 0 or step == 0:
            logger.info(
                "epoch %d  [%d/%d]  "
                "L_uv=%.4f  L_sp=%.4f  L_gr=%.4f  L_cr=%.4f  "
                "L_ls=%.4f  L_cv=%.4f  L_pers=%.4f  total=%.4f",
                epoch, step + 1, N,
                val_uv.item(), val_speed.item(), val_grad.item(), val_crit.item(),
                val_levelset.item(), val_ttkcv.item(), val_ttkpers.item(), loss.item(),
            )

    means = {k: v / N for k, v in running.items()}
    logger.info(
        "epoch %d DONE  mean L_uv=%.5f  L_speed=%.5f  L_grad=%.5f  L_crit=%.5f  "
        "L_levelset=%.5f  L_ttkcv=%.5f  L_ttkpers=%.5f  total=%.5f",
        epoch,
        means["uv"], means["speed"], means["grad"], means["crit"],
        means["levelset"], means["ttkcv"], means["ttkpers"], means["total"],
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
    import datetime
    rpt = Path(args.report_path)
    rpt.parent.mkdir(parents=True, exist_ok=True)

    with open(rpt, "w") as f:
        f.write("# Candidate E TTK Critical-Pair Refiner Notes\n\n")
        f.write(f"**Generated:** {datetime.datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write(
            "Candidate E = Candidate C losses + Candidate B level-set loss "
            "+ TTK critical-pair losses (Kissi-style).\n\n"
        )

        f.write("## Configuration\n\n```\n")
        for k, v in vars(args).items():
            f.write(f"  {k:<30s} = {v}\n")
        f.write("```\n\n")

        if diag:
            f.write("## Diagnostic Loss Breakdown\n\n")
            f.write(f"- Real data          : {'YES' if diag['is_real_data'] else 'NO'}\n")
            f.write(f"- Real constraints   : {'YES' if diag['is_real_constraints'] else 'NO'}\n")
            f.write(f"- Constraint samples : {diag['n_constraint_samples']}\n")
            f.write(f"- First sample id    : {diag['first_sample_id']}\n")
            f.write(f"- Pairs used         : {diag['n_pairs_used']}\n\n")
            f.write("| Term | Raw value | λ | Weighted |\n")
            f.write("|------|-----------|---|----------|\n")
            f.write(f"| L_uv      | {diag['L_uv']:.6f} | 1.0 | {diag['L_uv']:.6f} |\n")
            f.write(f"| L_speed   | {diag['L_speed']:.6f} | {args.lambda_speed} | {args.lambda_speed * diag['L_speed']:.6f} |\n")
            f.write(f"| L_grad    | {diag['L_grad']:.6f} | {args.lambda_grad} | {args.lambda_grad * diag['L_grad']:.6f} |\n")
            f.write(f"| L_crit    | {diag['L_crit']:.6f} | {args.lambda_crit} | {args.lambda_crit * diag['L_crit']:.6f} |\n")
            f.write(f"| L_levelset| {diag['L_levelset']:.6f} | {args.lambda_levelset} | {args.lambda_levelset * diag['L_levelset']:.6f} |\n")
            f.write(f"| L_ttkcv   | {diag['L_ttkcv']:.6f} | {args.lambda_ttkcv} | {args.lambda_ttkcv * diag['L_ttkcv']:.6f} |\n")
            f.write(f"| L_ttkpers | {diag['L_ttkpers']:.6f} | {args.lambda_ttkpers} | {args.lambda_ttkpers * diag['L_ttkpers']:.6f} |\n\n")
            f.write(f"L_ttkcv / L_uv   = **{diag['L_ttkcv_over_L_uv']:.4f}×**\n")
            f.write(f"L_ttkpers / L_uv = **{diag['L_ttkpers_over_L_uv']:.4f}×**\n")
            f.write(f"TTK loss time    = {diag['ttk_time_ms']:.2f} ms\n")
            f.write(f"L_ttkcv → model  = **{'YES' if diag['ttkcv_grad_to_model'] else 'NO'}**\n\n")

            f.write("## Persistence Sign Convention\n\n")
            f.write(
                "TTK stores `persistence = |death_scalar - birth_scalar|` (unsigned).\n"
                "The extraction script stores `death_val = birth_val + persistence_raw`,\n"
                "which gives `|death_val - birth_val| = persistence_raw`.\n"
                "Both `L_ttkcv` and `L_ttkpers` are direction-agnostic:\n"
                "`L_ttkpers` penalises `|sr_death - sr_birth| vs gt_persistence`.\n\n"
            )

        if epoch_logs:
            f.write("## Training Loss History\n\n")
            f.write("| Epoch | L_uv | L_speed | L_grad | L_crit | L_levelset | L_ttkcv | L_ttkpers | L_total |\n")
            f.write("|-------|------|---------|--------|--------|------------|---------|-----------|--------|\n")
            for ep, m in enumerate(epoch_logs, 1):
                f.write(
                    f"| {ep} | {m['uv']:.5f} | {m['speed']:.5f} | "
                    f"{m['grad']:.5f} | {m['crit']:.5f} | {m['levelset']:.5f} | "
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
    ap.add_argument("--data-dir",         default=str(_DATA_IN),
                    help="Input data directory (dataSR.npy, dataGT.npy, optional idx.npy)")
    ap.add_argument("--constraints",      required=True,
                    help="Path to ttk_pd_critical_pairs.npz from extract script.")
    ap.add_argument("--out-dir",          default=str(_OUT_DIR))
    ap.add_argument("--model-dir",        default=str(_MDL_DIR))
    ap.add_argument("--log-path",         default=str(_LOG_PATH))
    ap.add_argument("--report-path",      default=str(_RPT_PATH))

    ap.add_argument("--epochs",             type=int,   default=3)
    ap.add_argument("--lr",                 type=float, default=1e-4)
    ap.add_argument("--lambda-speed",       type=float, default=0.01)
    ap.add_argument("--lambda-grad",        type=float, default=0.05)
    ap.add_argument("--lambda-crit",        type=float, default=0.001)
    ap.add_argument("--lambda-levelset",    type=float, default=0.25,
                    help="Weight for level-set loss at thresholds 5/10/15 m/s.")
    ap.add_argument("--lambda-ttkcv",       type=float, default=0.0,
                    help="Weight for TTK critical-vertex MSE loss. "
                         "Use 0.0 for diagnostic; set after reviewing ratio.")
    ap.add_argument("--lambda-ttkpers",     type=float, default=0.0,
                    help="Weight for TTK persistence-gap loss. "
                         "Use 0.0 for diagnostic; set after reviewing ratio.")
    ap.add_argument("--residual-scale",     type=float, default=0.1)
    ap.add_argument("--seed",               type=int,   default=42)
    ap.add_argument("--max-pairs-per-sample", type=int, default=0,
                    help="Cap pairs per sample at this value (0 = use all from NPZ).")

    ap.add_argument("--allow-partial-constraints", action="store_true",
                    help="Allow constraints with fewer than 168 samples without aborting.")
    ap.add_argument("--diagnostic-only",  action="store_true",
                    help="Print loss breakdown and exit without training.")
    ap.add_argument("--dry-run",          action="store_true",
                    help="Run one epoch of 5 samples and exit (do not save outputs).")
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Safety guard ─────────────────────────────────────────────────────────
    if not args.diagnostic_only and not args.dry_run:
        _check_protected(Path(args.out_dir),   "--out-dir")
        _check_protected(Path(args.model_dir), "--model-dir")

    # ── Constraints ───────────────────────────────────────────────────────────
    constraints_path = Path(args.constraints)
    if not constraints_path.exists():
        logger.error(
            "Constraints file not found: %s\n"
            "Run extract_ttk_pd_critical_pairs.py first to generate it.",
            constraints_path,
        )
        sys.exit(1)

    constraints = TTKConstraints(constraints_path, patch=160, device=device)

    # Apply optional max-pairs cap: rebuild with trimmed arrays if needed
    if args.max_pairs_per_sample > 0:
        cap = args.max_pairs_per_sample
        trimmed_count = np.minimum(constraints.sample_count, cap)
        logger.info("Capping pairs/sample at %d (was max %d).", cap, int(constraints.sample_count.max()))
        constraints.sample_count = trimmed_count

    if constraints.n_samples < 168:
        msg = (
            f"Constraints contain only {constraints.n_samples} samples (expected 168). "
            "Run extract_ttk_pd_critical_pairs.py on the full 168-sample GT VTU set. "
            "Use --allow-partial-constraints to override."
        )
        if args.allow_partial_constraints:
            logger.warning(msg)
        else:
            logger.error(msg)
            sys.exit(1)

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    sr_all, gt_all, inp_all, idx_all = load_data(data_dir)
    logger.info("Data: SR=%s  GT=%s  idx=%s", sr_all.shape, gt_all.shape,
                idx_all.shape if idx_all is not None else "None")

    # Determine first_sample_id for diagnostic
    first_sample_id = int(idx_all[0]) if idx_all is not None else 0
    if first_sample_id not in constraints.available_indices:
        # Use first available constraint index as fallback for diagnostic
        first_sample_id = sorted(constraints.available_indices)[0]
        logger.warning(
            "First data sample id %d not in constraints. "
            "Using sample id %d for diagnostic.",
            int(idx_all[0]) if idx_all is not None else 0,
            first_sample_id,
        )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = RefinerNet(hidden=32, kernel=3, residual_scale=args.residual_scale).to(device)
    logger.info("RefinerNet: %d parameters", model.n_params())

    # ── Diagnostic ────────────────────────────────────────────────────────────
    sr_t = torch.tensor(sr_all[:1], device=device)
    gt_t = torch.tensor(gt_all[:1], device=device)

    diag = run_diagnostic(
        args, model, constraints, sr_t, gt_t,
        first_sample_id=first_sample_id,
        is_real_data=True,
        is_real_constraints=True,
    )
    write_report(args, diag, [])

    if args.diagnostic_only:
        return

    # ── Training ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epoch_logs = []

    max_steps = 5 if args.dry_run else None

    for epoch in range(1, args.epochs + 1):
        logger.info("=== Epoch %d / %d ===", epoch, args.epochs)
        epoch_sr = sr_all[:max_steps] if max_steps else sr_all
        epoch_gt = gt_all[:max_steps] if max_steps else gt_all
        epoch_idx = idx_all[:max_steps] if (max_steps and idx_all is not None) else idx_all

        means = run_epoch(
            epoch, args, model, optimizer, constraints,
            epoch_sr, epoch_gt, epoch_idx, device, logger,
        )
        epoch_logs.append(means)

        if not args.dry_run:
            ckpt_dir = Path(args.model_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            _check_protected(ckpt_dir, "--model-dir")
            ckpt_path = ckpt_dir / f"refiner_epoch{epoch}.pt"
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(), "args": vars(args)},
                ckpt_path,
            )
            logger.info("Checkpoint: %s", ckpt_path)

        if args.dry_run:
            logger.info("--dry-run: stopping after epoch %d.", epoch)
            break

    if not args.dry_run:
        save_outputs(args, model, sr_all, gt_all, inp_all, idx_all, device, logger)

    write_report(args, diag, epoch_logs)


if __name__ == "__main__":
    main()
