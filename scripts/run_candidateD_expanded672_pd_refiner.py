#!/usr/bin/env python3
"""
Candidate D-expanded-672: PyTorch residual refiner with PD Wasserstein loss,
trained on expanded 672-sample seasonal CNN SR outputs, evaluated on the
original corrected 168-sample benchmark.

Architecture (unchanged from Candidate D pilot)
-----------------------------------------------
RefinerNet is a small residual CNN that post-processes frozen baseline CNN SR:

    SR_D = SR_CNN + residual_scale * body(SR_CNN)      (residual_scale = 0.1)

body is zero-initialized, so SR_D == SR_CNN at t=0.  Gradients flow only
through the RefinerNet parameters; the pretrained PhIRE CNN is never touched.

Loss configuration (unchanged from Candidate D pilot)
------------------------------------------------------
    L_total = L_uv
            + 0.01  * L_speed
            + 0.05  * L_grad
            + 0.001 * L_crit    (pool-3 maxima proxy, z=1.0)
            + 0.0   * L_PD      (PD Wasserstein; lambda_pd=0 → monitoring only)

lambda_pd=0 matches the Candidate D pilot.  PD gradients are verified at
startup so lambda_pd can be raised if desired.  Recommended lambda_pd values
from the pilot diagnostic (on synthetic data): 0.001904 (10%), 0.004759 (25%).

Workflow
--------
Phase 0  Pre-condition checks (expanded CNN SR arrays, benchmark data, paths).
Phase 1  Train RefinerNet for 3 epochs on expanded 672-sample CNN SR outputs.
         Training data:  data_out/wind_mrhr_cnn_expanded672/
           SR=dataSR.npy (672, 500, 500, 2)  GT=dataGT.npy (672, 500, 500, 2)
Phase 2  Apply trained RefinerNet to 168-sample benchmark CNN SR outputs.
         Eval input:  data_out_fixed/wind_mrhr_cnn/
         Output:      data_out/wind_finetune_candidateD_expanded672/
           dataSR.npy (168, 500, 500, 2)   dataGT.npy (168, 500, 500, 2)
           dataIN.npy (168, 100, 100, 2)   idx.npy    (168,)
Phase 3  Validate output shapes + print min/max.

Generating the expanded CNN SR training arrays (TF1 PhIREGANs, on Spark)
-------------------------------------------------------------------------
If data_out/wind_mrhr_cnn_expanded672/ does not exist, run first:

    python3 scripts/generate_cnn_expanded672_inference.py

or manually:

    python3 - <<'PY'
    import sys; sys.path.insert(0, '.')
    import tensorflow.compat.v1 as tf; tf.disable_v2_behavior()
    from PhIREGANs import PhIREGANs
    phire = PhIREGANs(
        data_type='wind_mrhr_cnn_expanded672',
        mu_sig=[[0.7684, -0.4575], [5.02455, 5.9017]],
    )
    phire.set_data_out_path('data_out/wind_mrhr_cnn_expanded672')
    phire.test_paired(
        r=[5],
        data_path='example_data_topology_expanded_672/wind_MR-HR.tfrecord',
        model_path='models/wind_mr-hr/trained_cnn/cnn',
        batch_size=1,
        save_inputs=True,
    )
    PY

Environment
-----------
Requires either:
    micromamba run -p /home/adadhwal/PhIRE/.mamba_candidateD_pd \\
        python scripts/run_candidateD_expanded672_pd_refiner.py
or:
    .venv_candidateD_pd/bin/python scripts/run_candidateD_expanded672_pd_refiner.py

Do NOT add external tee — this script mirrors stdout/stderr to the log file
internally:
    logs/wind_finetune_candidateD_expanded672.log

GUDHI compatibility
-------------------
torch_topological 0.1.9 + gudhi >= 3.10 requires a one-line patch.
This script detects the incompatibility and applies the patch automatically.
See docs/candidateD_pd_gradient_smoke.md for details.

Scientific motivation
---------------------
CandidateD-expanded-672 tests whether the direct differentiable PD-loss
approach generalises when trained on the same non-overlapping 672-sample
seasonal dataset used by CandidateUV/B/C-expanded.  Comparison with:
  candidateUV_expanded672 — isolates contribution of L_speed/L_grad/L_crit/L_PD
  candidateB_expanded672  — isolates contribution of soft level-set vs PD
  candidateC_expanded672  — isolates contribution of L_PD vs L_crit
  candidateD (pilot)      — isolates effect of expanded training data

Normalisation constants (pretrained PhIRE; never changed)
  mu  = [0.7684, -0.4575]
  sig = [5.02455, 5.9017]

Protected directories — never overwritten:
  data_out_fixed/wind_mrhr_cnn/
  data_out_fixed/wind_mrhr_gan/
  data_out/wind_finetune_pilot_candidateB/
  data_out/wind_finetune_pilot_candidateC/
  data_out/wind_finetune_pilot_candidateD/
  data_out/wind_finetune_pilot_candidateE/
  data_out/wind_finetune_pilot_candidateE2/
  data_out/wind_finetune_pilot_candidateUV/
  data_out/wind_finetune_candidateUV_expanded672/
  data_out/wind_finetune_candidateB_expanded672/
  data_out/wind_finetune_candidateC_expanded672/
  example_data_fixed/
  example_data_topology_expanded_672/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateB/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateC/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateD/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateE/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateE2/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateUV/
  models_fixed/topology_finetuning/wind_finetune_candidateUV_expanded672/
  models_fixed/topology_finetuning/wind_finetune_candidateB_expanded672/
  models_fixed/topology_finetuning/wind_finetune_candidateC_expanded672/
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

# Expanded 672-sample CNN SR arrays (Phase 1 training input)
TRAIN_CNN_DIR = REPO_ROOT / 'data_out' / 'wind_mrhr_cnn_expanded672'

# Original 168-sample benchmark CNN SR arrays (Phase 2 evaluation input)
EVAL_CNN_DIR  = REPO_ROOT / 'data_out_fixed' / 'wind_mrhr_cnn'

# Outputs
MODEL_DIR = REPO_ROOT / 'models_fixed' / 'topology_finetuning' / 'wind_finetune_candidateD_expanded672'
OUT_DIR   = REPO_ROOT / 'data_out' / 'wind_finetune_candidateD_expanded672'
LOG_PATH  = REPO_ROOT / 'logs' / 'wind_finetune_candidateD_expanded672.log'

_PROTECTED = [
    REPO_ROOT / 'data_out_fixed' / 'wind_mrhr_cnn',
    REPO_ROOT / 'data_out_fixed' / 'wind_mrhr_gan',
    REPO_ROOT / 'data_out' / 'wind_finetune_pilot_candidateB',
    REPO_ROOT / 'data_out' / 'wind_finetune_pilot_candidateC',
    REPO_ROOT / 'data_out' / 'wind_finetune_pilot_candidateD',
    REPO_ROOT / 'data_out' / 'wind_finetune_pilot_candidateE',
    REPO_ROOT / 'data_out' / 'wind_finetune_pilot_candidateE2',
    REPO_ROOT / 'data_out' / 'wind_finetune_pilot_candidateUV',
    REPO_ROOT / 'data_out' / 'wind_finetune_candidateUV_expanded672',
    REPO_ROOT / 'data_out' / 'wind_finetune_candidateB_expanded672',
    REPO_ROOT / 'data_out' / 'wind_finetune_candidateC_expanded672',
    REPO_ROOT / 'example_data_fixed',
    REPO_ROOT / 'example_data_topology_expanded_672',
    REPO_ROOT / 'models_fixed' / 'topology_finetuning' / 'wind_finetune_pilot_candidateB',
    REPO_ROOT / 'models_fixed' / 'topology_finetuning' / 'wind_finetune_pilot_candidateC',
    REPO_ROOT / 'models_fixed' / 'topology_finetuning' / 'wind_finetune_pilot_candidateD',
    REPO_ROOT / 'models_fixed' / 'topology_finetuning' / 'wind_finetune_pilot_candidateE',
    REPO_ROOT / 'models_fixed' / 'topology_finetuning' / 'wind_finetune_pilot_candidateE2',
    REPO_ROOT / 'models_fixed' / 'topology_finetuning' / 'wind_finetune_pilot_candidateUV',
    REPO_ROOT / 'models_fixed' / 'topology_finetuning' / 'wind_finetune_candidateUV_expanded672',
    REPO_ROOT / 'models_fixed' / 'topology_finetuning' / 'wind_finetune_candidateB_expanded672',
    REPO_ROOT / 'models_fixed' / 'topology_finetuning' / 'wind_finetune_candidateC_expanded672',
]

# ---------------------------------------------------------------------------
# Loss configuration — matches Candidate D pilot exactly
# ---------------------------------------------------------------------------
LR             = 1e-4    # Adam learning rate (pilot: 0.0001)
EPOCHS         = 3
LAMBDA_SPEED   = 0.01
LAMBDA_GRAD    = 0.05
LAMBDA_CRIT    = 0.001
LAMBDA_PD      = 0.0     # PD Wasserstein weight (pilot ran with 0; monitoring only)
PD_CROP_SIZE   = 100     # Spatial crop for L_PD (100×100)
PD_EVERY       = 1       # Compute L_PD every N steps
RESIDUAL_SCALE = 0.1
SEED           = 42

N_TRAIN   = 672   # expected training samples
N_EVAL    = 168   # expected evaluation samples

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

# ---------------------------------------------------------------------------
# _Tee: mirror stdout/stderr to log file (same pattern as expanded scripts)
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, stream, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file   = open(path, 'a', buffering=1)
        self._stream = stream

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def fileno(self):
        return self._stream.fileno()


sys.stdout = _Tee(sys.stdout, LOG_PATH)
sys.stderr = sys.stdout

# ---------------------------------------------------------------------------
# GUDHI compatibility shim  (identical to Candidate D pilot)
# ---------------------------------------------------------------------------

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
            "torch_topological not found.\n"
            "Run inside .mamba_candidateD_pd or .venv_candidateD_pd:\n"
            "  micromamba run -p /home/adadhwal/PhIRE/.mamba_candidateD_pd \\\n"
            "      python scripts/run_candidateD_expanded672_pd_refiner.py"
        )

    src_path = Path(_mod.__file__)
    src = src_path.read_text()

    OLD = "top_dimensional_cells=x.flatten()"
    NEW = "top_dimensional_cells=x.detach().cpu().numpy().flatten()"

    if OLD in src:
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
        return "already_patched_variant"
    else:
        logging.info(
            "[compat] GUDHI call pattern not recognized. "
            "Proceeding — _verify_pd_gradients() is the authoritative check."
        )
        return "unrecognized"


def _verify_pd_gradients() -> None:
    """Verify that CubicalComplex + WassersteinDistance produces gradients."""
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
            "Ensure you are running inside .mamba_candidateD_pd or .venv_candidateD_pd\n"
            "with the GUDHI compatibility patch applied.\n"
            "See docs/candidateD_pd_gradient_smoke.md for instructions."
        ) from e

    if x.grad is None or not (x.grad != 0).any():
        raise RuntimeError(
            "PD backward() produced no gradients on a test tensor.\n"
            "See docs/candidateD_pd_gradient_smoke.md."
        )
    nz = int((x.grad != 0).sum())
    logging.info("[verify] PD gradient check passed: %d/256 nonzero entries", nz)


# ---------------------------------------------------------------------------
# Model  (identical to Candidate D pilot)
# ---------------------------------------------------------------------------

class RefinerNet(nn.Module):
    """
    Small residual CNN refiner for [u, v] wind fields.

    SR_D = SR_CNN + residual_scale * body(SR_CNN)

    body's last conv is zero-initialized so SR_D == SR_CNN at t=0.
    """

    def __init__(
        self,
        hidden: int = 32,
        kernel: int = 3,
        residual_scale: float = RESIDUAL_SCALE,
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


# ---------------------------------------------------------------------------
# Loss functions  (identical to Candidate D pilot)
# ---------------------------------------------------------------------------

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


def l_pd(
    sr: torch.Tensor,
    gt: torch.Tensor,
    cubical,
    w_loss_fn,
    crop: int = PD_CROP_SIZE,
) -> torch.Tensor:
    """PD Wasserstein-2 loss on GT-normalized scalar speed (superlevel-set)."""
    spd_sr = _speed_t(sr)
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


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

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
    """Load SR, GT, IN, idx arrays. Returns (sr, gt, inp, idx) in (N,2,H,W)."""
    sr  = _to_chw(np.load(data_dir / 'dataSR.npy'))
    gt  = _to_chw(np.load(data_dir / 'dataGT.npy'))
    inp = _to_chw(np.load(data_dir / 'dataIN.npy')) if (data_dir / 'dataIN.npy').exists() else None
    idx = np.load(data_dir / 'idx.npy')              if (data_dir / 'idx.npy').exists()    else None
    return sr, gt, inp, idx


# ---------------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------------

def _safety_check() -> None:
    for p in _PROTECTED:
        for candidate_path, label in ((MODEL_DIR, 'MODEL_DIR'), (OUT_DIR, 'OUT_DIR')):
            try:
                candidate_path.resolve().relative_to(p.resolve())
                sys.exit(
                    f'[error] {label} ({candidate_path}) would write inside protected '
                    f'directory {p}. Aborting to avoid overwriting baselines.'
                )
            except ValueError:
                pass
    print('[safety] Output paths do not overlap any protected directory. OK.')


# ---------------------------------------------------------------------------
# Training epoch  (identical logic to Candidate D pilot)
# ---------------------------------------------------------------------------

def run_epoch(
    epoch: int,
    model: RefinerNet,
    optimizer: torch.optim.Optimizer,
    cubical,
    w_loss_fn,
    sr_all: np.ndarray,
    gt_all: np.ndarray,
    device: torch.device,
) -> dict:
    model.train()
    N = sr_all.shape[0]
    indices = list(range(N))
    random.shuffle(indices)

    running = {k: 0.0 for k in ('uv', 'speed', 'grad', 'crit', 'pd', 'total')}
    pd_checked_grad = (LAMBDA_PD == 0.0)  # skip check if lambda_pd=0

    for step, i in enumerate(indices):
        sr_t = torch.tensor(sr_all[i:i+1], device=device)
        gt_t = torch.tensor(gt_all[i:i+1], device=device)

        optimizer.zero_grad()

        refined   = model(sr_t)
        val_uv    = l_uv(refined, gt_t)
        val_speed = l_speed(refined, gt_t)
        val_grad  = l_grad(refined, gt_t)
        val_crit  = l_crit(refined, gt_t, pool=3, high_z=1.0)

        loss = (
            val_uv
            + LAMBDA_SPEED * val_speed
            + LAMBDA_GRAD  * val_grad
            + LAMBDA_CRIT  * val_crit
        )

        val_pd = torch.zeros((), device=device)
        use_pd = (LAMBDA_PD > 0.0) and (step % PD_EVERY == 0)
        if use_pd:
            val_pd = l_pd(refined, gt_t, cubical, w_loss_fn, crop=PD_CROP_SIZE)
            if torch.isnan(val_pd) or torch.isinf(val_pd):
                raise RuntimeError(
                    f"L_PD is NaN/inf at epoch {epoch} step {step}. Aborting. "
                    "Try reducing LAMBDA_PD or PD_CROP_SIZE."
                )
            loss = loss + LAMBDA_PD * val_pd

        loss.backward()

        if use_pd and not pd_checked_grad:
            has_grad = any(
                p.grad is not None and (p.grad != 0).any()
                for p in model.parameters()
            )
            if not has_grad:
                raise RuntimeError(
                    "LAMBDA_PD > 0 but L_PD produced no gradients to model parameters. "
                    "Check GUDHI patch. Aborting."
                )
            pd_checked_grad = True
            logging.info('[train] PD gradient check passed at epoch %d step %d.', epoch, step)

        optimizer.step()

        running['uv']    += val_uv.item()
        running['speed'] += val_speed.item()
        running['grad']  += val_grad.item()
        running['crit']  += val_crit.item()
        running['pd']    += val_pd.item()
        running['total'] += loss.item()

        if (step + 1) % 50 == 0 or step == 0:
            logging.info(
                'epoch %d  [%d/%d]  '
                'L_uv=%.4f  L_sp=%.4f  L_gr=%.4f  L_cr=%.4f  L_PD=%.4f  total=%.4f',
                epoch, step + 1, N,
                val_uv.item(), val_speed.item(), val_grad.item(),
                val_crit.item(), val_pd.item(), loss.item(),
            )

    means = {k: v / N for k, v in running.items()}
    logging.info(
        'epoch %d DONE  mean L_uv=%.5f  L_speed=%.5f  L_grad=%.5f  '
        'L_crit=%.5f  L_PD=%.5f  total=%.5f',
        epoch, means['uv'], means['speed'], means['grad'],
        means['crit'], means['pd'], means['total'],
    )
    return means


# ---------------------------------------------------------------------------
# Output saving + validation
# ---------------------------------------------------------------------------

def save_and_validate(
    model: RefinerNet,
    sr_all: np.ndarray,
    gt_all: np.ndarray,
    inp_all,
    idx_all,
    device: torch.device,
    n_expected: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model.eval()
    N = sr_all.shape[0]
    refined_all = []

    with torch.no_grad():
        for i in range(N):
            sr_t = torch.tensor(sr_all[i:i+1], device=device)
            ref  = model(sr_t).cpu().numpy()[0]  # (2, H, W)
            refined_all.append(ref)

    refined_arr = np.stack(refined_all, axis=0)           # (N, 2, H, W)
    refined_hwc = np.transpose(refined_arr, (0, 2, 3, 1)) # (N, H, W, 2)
    gt_hwc      = np.transpose(gt_all,      (0, 2, 3, 1))

    np.save(OUT_DIR / 'dataSR.npy', refined_hwc)
    np.save(OUT_DIR / 'dataGT.npy', gt_hwc)
    if inp_all is not None:
        inp_hwc = (
            np.transpose(inp_all, (0, 2, 3, 1))
            if (inp_all.ndim == 4 and inp_all.shape[1] == 2)
            else inp_all
        )
        np.save(OUT_DIR / 'dataIN.npy', inp_hwc)
    if idx_all is not None:
        np.save(OUT_DIR / 'idx.npy', idx_all)

    logging.info('Saved %d refined samples to %s', N, OUT_DIR)

    # ── Validation ────────────────────────────────────────────────────────
    print()
    print('=' * 64)
    print('Output validation')
    print('=' * 64)
    errors = []
    for fname, expected_shape in [
        ('idx.npy',    (n_expected,)),
        ('dataIN.npy', (n_expected, 100, 100, 2)),
        ('dataGT.npy', (n_expected, 500, 500, 2)),
        ('dataSR.npy', (n_expected, 500, 500, 2)),
    ]:
        p = OUT_DIR / fname
        if not p.exists():
            errors.append(f'  MISSING: {fname}')
            print(f'  [MISSING] {fname}')
            continue
        a = np.load(p, mmap_mode='r')
        shape_ok = (a.shape == expected_shape)
        status = 'OK' if shape_ok else 'SHAPE MISMATCH'
        print(f'  [{status}] {fname}')
        print(f'    shape  : {a.shape}  (expected {expected_shape})')
        print(f'    dtype  : {a.dtype}')
        print(f'    min/max: {float(np.nanmin(a)):.4f} / {float(np.nanmax(a)):.4f}')
        if not shape_ok:
            errors.append(f'  SHAPE MISMATCH: {fname}: got {a.shape}, expected {expected_shape}')

    if errors:
        sys.exit('[error] Output validation failed:\n' + '\n'.join(errors))
    print('[validation] All output shapes OK.')
    print('=' * 64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Logging (stdout already captured by _Tee) ─────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)s  %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    # ── Safety checks ────────────────────────────────────────────────────
    _safety_check()

    # ── Banner ───────────────────────────────────────────────────────────
    print('=' * 64)
    print('Candidate D-expanded-672: PD residual refiner')
    print('=' * 64)
    print(f'  Training CNN SR  : {TRAIN_CNN_DIR}')
    print(f'  Eval CNN SR      : {EVAL_CNN_DIR}')
    print(f'  Output model dir : {MODEL_DIR}')
    print(f'  Inference output : {OUT_DIR}')
    print(f'  Log file         : {LOG_PATH}')
    print()
    print('  Architecture:')
    print('    SR_D = SR_CNN + 0.1 * body(SR_CNN)  [body zero-init]')
    print()
    print('  Loss configuration (Candidate D pilot — unchanged):')
    print(f'  lambda_speed = {LAMBDA_SPEED}')
    print(f'  lambda_grad  = {LAMBDA_GRAD}')
    print(f'  lambda_crit  = {LAMBDA_CRIT}  (pool=3, z=1.0)')
    print(f'  lambda_pd    = {LAMBDA_PD}    (PD Wasserstein; 0 = monitoring only)')
    print()
    print('  Hyperparameters (Candidate D pilot — unchanged):')
    print(f'  lr             = {LR}')
    print(f'  epochs         = {EPOCHS}')
    print(f'  residual_scale = {RESIDUAL_SCALE}')
    print(f'  pd_crop_size   = {PD_CROP_SIZE}')
    print(f'  seed           = {SEED}')
    print()
    print('  Normalisation (pretrained; unchanged):')
    print('  mu  = [0.7684, -0.4575]')
    print('  sig = [5.02455, 5.9017]')
    print('=' * 64)
    print()

    # ── Pre-condition checks ─────────────────────────────────────────────
    if not (TRAIN_CNN_DIR / 'dataSR.npy').exists():
        sys.exit(
            f'[error] Expanded CNN SR training arrays not found: {TRAIN_CNN_DIR}\n'
            '  Generate them first (requires TF1 PhIREGANs on Spark):\n'
            '\n'
            '    python3 - <<\'PY\'\n'
            '    import sys; sys.path.insert(0, \'.\')\n'
            '    import tensorflow.compat.v1 as tf; tf.disable_v2_behavior()\n'
            '    from PhIREGANs import PhIREGANs\n'
            '    phire = PhIREGANs(\n'
            '        data_type=\'wind_mrhr_cnn_expanded672\',\n'
            '        mu_sig=[[0.7684, -0.4575], [5.02455, 5.9017]],\n'
            '    )\n'
            '    phire.set_data_out_path(\'data_out/wind_mrhr_cnn_expanded672\')\n'
            '    phire.test_paired(\n'
            '        r=[5],\n'
            '        data_path=\'example_data_topology_expanded_672/wind_MR-HR.tfrecord\',\n'
            '        model_path=\'models/wind_mr-hr/trained_cnn/cnn\',\n'
            '        batch_size=1, save_inputs=True,\n'
            '    )\n'
            '    PY'
        )

    train_sr_shape = np.load(TRAIN_CNN_DIR / 'dataSR.npy', mmap_mode='r').shape
    if train_sr_shape[0] != N_TRAIN:
        sys.exit(
            f'[error] Expanded CNN SR arrays have {train_sr_shape[0]} samples; '
            f'expected {N_TRAIN}.\n'
            f'  Regenerate: {TRAIN_CNN_DIR}'
        )
    logger.info('Training data: %s  shape=%s', TRAIN_CNN_DIR, train_sr_shape)

    if not (EVAL_CNN_DIR / 'dataSR.npy').exists():
        sys.exit(
            f'[error] Benchmark CNN SR arrays not found: {EVAL_CNN_DIR}\n'
            '  The 168-sample baseline CNN outputs are required for evaluation.'
        )
    eval_sr_shape = np.load(EVAL_CNN_DIR / 'dataSR.npy', mmap_mode='r').shape
    if eval_sr_shape[0] != N_EVAL:
        sys.exit(
            f'[error] Benchmark CNN SR arrays have {eval_sr_shape[0]} samples; '
            f'expected {N_EVAL}.'
        )
    logger.info('Eval data: %s  shape=%s', EVAL_CNN_DIR, eval_sr_shape)

    # ── Environment / PD gradient check ─────────────────────────────────
    patch_status = _apply_gudhi_compat_patch()
    logger.info('[compat] GUDHI patch status: %s', patch_status)
    _verify_pd_gradients()
    logger.info('[verify] PD gradient check passed.')

    from torch_topological.nn import CubicalComplex, WassersteinDistance
    cubical   = CubicalComplex(superlevel=True)
    w_loss_fn = WassersteinDistance(q=2)

    # ── Determinism ──────────────────────────────────────────────────────
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device('cpu')

    # ── Model ─────────────────────────────────────────────────────────────
    model = RefinerNet(hidden=32, kernel=3, residual_scale=RESIDUAL_SCALE).to(device)
    logger.info('RefinerNet: %d parameters', model.n_params())

    # ── Phase 1: Training on 672-sample expanded CNN SR outputs ───────────
    print('=' * 64)
    print('Phase 1: Training on 672-sample expanded CNN SR outputs')
    print('=' * 64)

    logger.info('Loading training data …')
    sr_train, gt_train, _inp_train, _idx_train = load_data(TRAIN_CNN_DIR)
    logger.info('Training arrays: SR=%s  GT=%s', sr_train.shape, gt_train.shape)

    if LAMBDA_PD == 0.0:
        logger.info(
            'lambda_pd=0 — L_PD excluded from training loss (diagnostic monitoring only).'
        )
    else:
        logger.info(
            'lambda_pd=%.4g — L_PD will be computed every %d steps.', LAMBDA_PD, PD_EVERY
        )

    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    epoch_logs = []

    for ep in range(1, EPOCHS + 1):
        logger.info('=== Epoch %d / %d ===', ep, EPOCHS)
        means = run_epoch(ep, model, optimizer, cubical, w_loss_fn,
                          sr_train, gt_train, device)
        epoch_logs.append(means)

        ckpt = MODEL_DIR / f'refiner_epoch{ep:02d}.pt'
        torch.save(model.state_dict(), ckpt)
        logger.info('Checkpoint: %s', ckpt)

    final_ckpt = MODEL_DIR / 'refiner_final.pt'
    torch.save(model.state_dict(), final_ckpt)
    logger.info('Final model saved: %s', final_ckpt)

    print()
    print(f'Phase 1 complete. Final checkpoint: {final_ckpt}')
    print()

    # ── Phase 2: Evaluate on 168-sample benchmark ─────────────────────────
    print('=' * 64)
    print('Phase 2: Evaluate on 168-sample benchmark CNN SR outputs')
    print('=' * 64)
    print(f'  Eval input  : {EVAL_CNN_DIR}')
    print(f'  Output dir  : {OUT_DIR}')
    print()

    logger.info('Loading evaluation data …')
    sr_eval, gt_eval, inp_eval, idx_eval = load_data(EVAL_CNN_DIR)
    logger.info('Eval arrays: SR=%s  GT=%s', sr_eval.shape, gt_eval.shape)

    save_and_validate(model, sr_eval, gt_eval, inp_eval, idx_eval, device, N_EVAL)

    # ── Epoch loss summary ────────────────────────────────────────────────
    print()
    print('=' * 64)
    print('Training loss history')
    print('=' * 64)
    print(f"  {'Epoch':>5}  {'L_uv':>8}  {'L_speed':>8}  {'L_grad':>8}  "
          f"{'L_crit':>8}  {'L_PD':>8}  {'L_total':>8}")
    for ep, m in enumerate(epoch_logs, 1):
        print(f"  {ep:>5}  {m['uv']:>8.5f}  {m['speed']:>8.5f}  {m['grad']:>8.5f}  "
              f"{m['crit']:>8.5f}  {m['pd']:>8.5f}  {m['total']:>8.5f}")

    # ── Completion summary ────────────────────────────────────────────────
    print()
    print('=' * 64)
    print('Run complete.')
    print(f'  Final checkpoint : {final_ckpt}')
    print(f'  SR outputs       : {OUT_DIR}/dataSR.npy')
    print(f'  GT outputs       : {OUT_DIR}/dataGT.npy')
    print(f'  LR inputs        : {OUT_DIR}/dataIN.npy')
    print(f'  Sample indices   : {OUT_DIR}/idx.npy')
    print()
    print('Next steps:')
    print('  Evaluate:')
    print('    python3 scripts/evaluate_finetune_candidate.py \\')
    print('      --candidate-name candidateD_expanded672 \\')
    print('      --candidate-dir  data_out/wind_finetune_candidateD_expanded672 \\')
    print('      --cnn-dir        data_out_fixed/wind_mrhr_cnn \\')
    print('      --gan-dir        data_out_fixed/wind_mrhr_gan \\')
    print('      --merged-csv     ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \\')
    print('      --out-dir        ttk_runs_fixed/topology_finetuning/candidateD_expanded672_eval')
    print()
    print('  TTK topology pipeline:')
    print('    bash scripts/run_candidate_topology_pipeline.sh \\')
    print('      --method   candidateD_expanded672 \\')
    print('      --data-dir data_out/wind_finetune_candidateD_expanded672')
    print('=' * 64)


if __name__ == '__main__':
    main()
