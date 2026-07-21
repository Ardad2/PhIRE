#!/usr/bin/env python3
"""
CandidateE2-expanded-672: PyTorch residual refiner with TTK critical-pair loss
trained on the 672-sample expanded seasonal dataset, evaluated on the 168-sample
benchmark.

Loss configuration:
    L_total = L_uv
            + 0.01  * L_speed
            + 0.05  * L_grad
            + 0.001 * L_crit          (Candidate C pool-3 local-maxima proxy)
            + 0.25  * L_levelset      (Candidate B soft level-set at 5/10/15 m/s)
            + 0.04  * L_ttkcv         (MSE at TTK critical vertices)
            + 0.02  * L_ttkpers       (persistence-gap MSE)

TTK constraints are pre-computed by build_candidateE2_fixed_constraints.py
from the expanded GT arrays.  Birth/death vertex target values are read from GT
numpy arrays directly (exact match guaranteed).

Workflow:
  Phase 1 — Training:  train RefinerNet on 672 expanded CNN SR outputs
            Input:  data_out/wind_mrhr_cnn_expanded672/  (dataSR.npy, dataGT.npy)
            Constraints: ttk_runs_fixed/topology_finetuning/
                           candidateE2_fixed_constraints/
                           ttk_pd_critical_pairs_gtvalues.npz
            Output: models_fixed/topology_finetuning/
                      wind_finetune_candidateE2_fixed/

  Phase 2 — Evaluation: apply trained refiner to 168-sample benchmark CNN SR
            Input:  data_out_fixed/wind_mrhr_cnn/  (dataSR.npy, dataGT.npy)
            Output: data_out/wind_finetune_candidateE2_fixed/
                      dataSR.npy (168, 500, 500, 2)
                      dataGT.npy (168, 500, 500, 2)
                      dataIN.npy (168, 100, 100, 2)
                      idx.npy    (168,)

Run:
  micromamba run -p /home/adadhwal/PhIRE/.mamba_candidateD_pd \\
      python scripts/run_candidateE2_expanded672_ttkcrit_refiner.py
  # or:
  .venv_candidateD_pd/bin/python \\
      scripts/run_candidateE2_expanded672_ttkcrit_refiner.py
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Repository root ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Paths ──────────────────────────────────────────────────────────────────────

TRAIN_CNN_DIR   = REPO_ROOT / "data_out" / "wind_mrhr_cnn_expanded672"
EVAL_CNN_DIR    = REPO_ROOT / "data_out_fixed" / "wind_mrhr_cnn"
CONSTRAINTS_NPZ = (
    REPO_ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateE2_fixed_constraints"
    / "ttk_pd_critical_pairs_gtvalues.npz"
)
MODEL_DIR = (
    REPO_ROOT / "models_fixed" / "topology_finetuning"
    / "wind_finetune_candidateE2_fixed"
)
OUT_DIR  = REPO_ROOT / "data_out" / "wind_finetune_candidateE2_fixed"
LOG_PATH = REPO_ROOT / "logs"    / "wind_finetune_candidateE2_fixed.log"

# ── Hyperparameters ────────────────────────────────────────────────────────────

LR             = 1e-4
EPOCHS         = 3
RESIDUAL_SCALE = 0.1
SEED           = 42

LAMBDA_SPEED    = 0.01
LAMBDA_GRAD     = 0.05
LAMBDA_CRIT     = 0.001
LAMBDA_LEVELSET = 0.25
LAMBDA_TTKCV    = 0.04
LAMBDA_TTKPERS  = 0.02

N_TRAIN = 672
N_EVAL  = 168
PATCH   = 160    # TTKConstraints vertex-ID encoding width
LOG_EVERY = 10

# ── Protected output paths ─────────────────────────────────────────────────────

_PROTECTED = [
    REPO_ROOT / "data_out_fixed" / "wind_mrhr_cnn",
    REPO_ROOT / "data_out_fixed" / "wind_mrhr_gan",
    REPO_ROOT / "data_out"       / "wind_mrhr_cnn_expanded672",
    REPO_ROOT / "data_out"       / "wind_finetune_pilot_candidateB",
    REPO_ROOT / "data_out"       / "wind_finetune_pilot_candidateC",
    REPO_ROOT / "data_out"       / "wind_finetune_pilot_candidateD",
    REPO_ROOT / "data_out"       / "wind_finetune_pilot_candidateE",
    REPO_ROOT / "data_out"       / "wind_finetune_pilot_candidateE2",
    REPO_ROOT / "data_out"       / "wind_finetune_candidateUV_expanded672",
    REPO_ROOT / "data_out"       / "wind_finetune_candidateB_expanded672",
    REPO_ROOT / "data_out"       / "wind_finetune_candidateC_expanded672",
    REPO_ROOT / "data_out"       / "wind_finetune_candidateD_expanded672",
    REPO_ROOT / "data_out"       / "wind_finetune_candidateDpd_expanded672",
    REPO_ROOT / "models_fixed"   / "topology_finetuning" / "wind_finetune_pilot_candidateB",
    REPO_ROOT / "models_fixed"   / "topology_finetuning" / "wind_finetune_pilot_candidateC",
    REPO_ROOT / "models_fixed"   / "topology_finetuning" / "wind_finetune_pilot_candidateD",
    REPO_ROOT / "models_fixed"   / "topology_finetuning" / "wind_finetune_pilot_candidateE",
    REPO_ROOT / "models_fixed"   / "topology_finetuning" / "wind_finetune_pilot_candidateE2",
    REPO_ROOT / "models_fixed"   / "topology_finetuning" / "wind_finetune_candidateUV_expanded672",
    REPO_ROOT / "models_fixed"   / "topology_finetuning" / "wind_finetune_candidateD_expanded672",
    REPO_ROOT / "models_fixed"   / "topology_finetuning" / "wind_finetune_candidateDpd_expanded672",
    REPO_ROOT / "example_data_fixed",
    REPO_ROOT / "example_data_topology_expanded_672",
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


# ── Tee: mirror stdout/stderr to log file ─────────────────────────────────────

class _Tee:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file   = open(path, "w", buffering=1)
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout   = self
        sys.stderr   = self

    def write(self, data: str) -> None:
        self._stdout.write(data)
        self._file.write(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._file.close()


# ── Model ─────────────────────────────────────────────────────────────────────

class RefinerNet(nn.Module):
    """
    Small residual CNN: refined = input + residual_scale * body(input).
    Body last-conv is zero-initialised → output equals input at t=0.
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
    spd_sr = _speed_t(sr)
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
    """MSE at TTK critical vertices (birth and death separately, averaged)."""
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

    return (F.mse_loss(sr_birth, birth_target) + F.mse_loss(sr_death, death_target)) * 0.5


def l_ttkpers(
    sr: torch.Tensor,
    birth_yx: torch.Tensor,
    death_yx: torch.Tensor,
    gt_persistence: torch.Tensor,
) -> torch.Tensor:
    """MSE between |SR_death - SR_birth| and GT persistence (unsigned)."""
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
    Pre-computed TTK critical-pair constraints. O(1) lookup by sample_idx.

    patch=160: vertex IDs encoded as vid = row * 160 + col.
    """

    def __init__(self, npz_path: Path, patch: int, device: torch.device) -> None:
        npz = np.load(npz_path, allow_pickle=True)

        required = {
            "n_samples", "sample_idx", "sample_start", "sample_count",
            "birth_vid", "death_vid", "birth_val", "death_val", "persistence",
        }
        missing = required - set(npz.files)
        if missing:
            raise ValueError(
                f"Constraints NPZ missing required keys: {missing}. "
                "Re-run build_candidateE2_fixed_constraints.py."
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

        self._idx_to_row: dict[int, int] = {
            int(self.sample_idx[i]): i for i in range(len(self.sample_idx))
        }
        self._device = device
        self._W = patch

        print(
            f"[constraints] {self.n_samples} samples, "
            f"{len(self.birth_vid)} total pairs, "
            f"sample_idx range [{int(self.sample_idx.min())}, {int(self.sample_idx.max())}]"
        )

    def get(self, sample_idx: int):
        """Return (birth_yx, death_yx, birth_target, death_target, gt_persistence)."""
        if sample_idx not in self._idx_to_row:
            raise KeyError(
                f"sample_idx={sample_idx} not found in constraints NPZ. "
                f"Available (first 10): {sorted(self._idx_to_row)[:10]}"
            )
        row   = self._idx_to_row[sample_idx]
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


def _load_cnn_dir(data_dir: Path, label: str) -> tuple:
    """Load dataSR.npy, dataGT.npy, optional dataIN.npy, optional idx.npy."""
    sr_path = data_dir / "dataSR.npy"
    gt_path = data_dir / "dataGT.npy"

    if not sr_path.exists() or not gt_path.exists():
        sys.exit(
            f"[error] {label}: required files missing in {data_dir}.\n"
            "  Expected: dataSR.npy, dataGT.npy"
        )

    sr  = _to_chw(np.load(sr_path))
    gt  = _to_chw(np.load(gt_path))
    inp = (_to_chw(np.load(data_dir / "dataIN.npy"))
           if (data_dir / "dataIN.npy").exists() else None)
    idx = (np.load(data_dir / "idx.npy")
           if (data_dir / "idx.npy").exists() else None)
    return sr, gt, inp, idx


# ── Safety checks ─────────────────────────────────────────────────────────────

def _check_protected(path: Path, label: str) -> None:
    for p in _PROTECTED:
        try:
            path.resolve().relative_to(p.resolve())
            sys.exit(
                f"[error] {label} ({path}) would write inside protected "
                f"directory {p}. Aborting."
            )
        except ValueError:
            pass


def _abort_nan(val: torch.Tensor, name: str) -> None:
    if torch.isnan(val) or torch.isinf(val):
        raise RuntimeError(
            f"Loss {name} is NaN/inf ({val.item():.6g}). Aborting."
        )


# ── Pre-condition checks ───────────────────────────────────────────────────────

def _check_preconditions() -> None:
    # Expanded training CNN SR
    for fname in ("dataSR.npy", "dataGT.npy"):
        p = TRAIN_CNN_DIR / fname
        if not p.exists():
            sys.exit(
                f"[error] Expanded CNN SR arrays not found: {p}\n"
                "  Generate them first (requires TF1):\n"
                "    python3 - <<'PY'\n"
                "    import sys; sys.path.insert(0, '.')\n"
                "    import tensorflow.compat.v1 as tf; tf.disable_v2_behavior()\n"
                "    from PhIREGANs import PhIREGANs\n"
                "    phire = PhIREGANs(\n"
                "        data_type='wind_mrhr_cnn_expanded672',\n"
                "        mu_sig=[[0.7684, -0.4575], [5.02455, 5.9017]],\n"
                "    )\n"
                "    phire.set_data_out_path('data_out/wind_mrhr_cnn_expanded672')\n"
                "    phire.test_paired(\n"
                "        r=[5],\n"
                "        data_path='example_data_topology_expanded_672/wind_MR-HR.tfrecord',\n"
                "        model_path='models/wind_mr-hr/trained_cnn/cnn',\n"
                "        batch_size=1, save_inputs=True,\n"
                "    )\n"
                "    PY"
            )

    sr_shape = np.load(TRAIN_CNN_DIR / "dataSR.npy", mmap_mode="r").shape
    if sr_shape[0] != N_TRAIN:
        sys.exit(
            f"[error] Expanded CNN SR has {sr_shape[0]} samples; expected {N_TRAIN}.\n"
            f"  Regenerate: {TRAIN_CNN_DIR / 'dataSR.npy'}"
        )

    # Benchmark CNN SR
    for fname in ("dataSR.npy", "dataGT.npy"):
        p = EVAL_CNN_DIR / fname
        if not p.exists():
            sys.exit(f"[error] Benchmark CNN SR not found: {p}")

    eval_shape = np.load(EVAL_CNN_DIR / "dataSR.npy", mmap_mode="r").shape
    if eval_shape[0] != N_EVAL:
        sys.exit(
            f"[error] Benchmark CNN SR has {eval_shape[0]} samples; expected {N_EVAL}.\n"
            f"  Expected: {EVAL_CNN_DIR / 'dataSR.npy'}"
        )

    # Constraints NPZ
    if not CONSTRAINTS_NPZ.exists():
        sys.exit(
            f"[error] Constraints NPZ not found: {CONSTRAINTS_NPZ}\n"
            "  Generate it first:\n"
            "    python3 scripts/build_candidateE2_fixed_constraints.py"
        )

    npz = np.load(CONSTRAINTS_NPZ, allow_pickle=True)
    n_constraints = int(npz["n_samples"])
    if n_constraints != N_TRAIN:
        sys.exit(
            f"[error] Constraints NPZ has {n_constraints} samples; expected {N_TRAIN}.\n"
            "  Re-run: python3 scripts/build_candidateE2_fixed_constraints.py"
        )

    # Output paths safety
    _check_protected(OUT_DIR,   "OUT_DIR")
    _check_protected(MODEL_DIR, "MODEL_DIR")


# ── Training ───────────────────────────────────────────────────────────────────

def run_epoch(
    epoch: int,
    model: RefinerNet,
    optimizer: torch.optim.Optimizer,
    constraints: TTKConstraints,
    sr_all: np.ndarray,
    gt_all: np.ndarray,
    idx_all,
    device: torch.device,
) -> dict:
    model.train()
    N = sr_all.shape[0]
    indices = list(range(N))
    random.shuffle(indices)

    running = {k: 0.0 for k in (
        "uv", "speed", "grad", "crit", "levelset", "ttkcv", "ttkpers", "total"
    )}

    for step, i in enumerate(indices):
        sr_t = torch.tensor(sr_all[i : i + 1], device=device)
        gt_t = torch.tensor(gt_all[i : i + 1], device=device)

        actual_id = int(idx_all[i]) if idx_all is not None else i

        try:
            birth_yx, death_yx, birth_target, death_target, gt_pers = (
                constraints.get(actual_id)
            )
        except KeyError as exc:
            print(f"[warn] epoch {epoch} step {step}: {exc} — using empty constraints.")
            empty_yx  = torch.zeros((0, 2), dtype=torch.long,    device=device)
            empty_val = torch.zeros((0,),   dtype=torch.float32, device=device)
            birth_yx = death_yx = empty_yx
            birth_target = death_target = gt_pers = empty_val

        optimizer.zero_grad()
        refined = model(sr_t)

        val_uv       = l_uv(refined, gt_t)
        val_speed    = l_speed(refined, gt_t)
        val_grad     = l_grad(refined, gt_t)
        val_crit     = l_crit(refined, gt_t, pool=3, high_z=1.0)
        val_levelset = l_levelset(refined, gt_t)
        val_ttkcv    = l_ttkcv(refined, birth_yx, death_yx, birth_target, death_target)
        val_ttkpers  = l_ttkpers(refined, birth_yx, death_yx, gt_pers)

        for name, val in [
            ("L_uv", val_uv), ("L_speed", val_speed), ("L_grad", val_grad),
            ("L_crit", val_crit), ("L_levelset", val_levelset),
            ("L_ttkcv", val_ttkcv), ("L_ttkpers", val_ttkpers),
        ]:
            _abort_nan(val, f"{name} epoch {epoch} step {step}")

        loss = (
            val_uv
            + LAMBDA_SPEED    * val_speed
            + LAMBDA_GRAD     * val_grad
            + LAMBDA_CRIT     * val_crit
            + LAMBDA_LEVELSET * val_levelset
            + LAMBDA_TTKCV    * val_ttkcv
            + LAMBDA_TTKPERS  * val_ttkpers
        )
        _abort_nan(loss, f"L_total epoch {epoch} step {step}")

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

        if step == 0 or (step + 1) % LOG_EVERY == 0:
            print(
                f"epoch {epoch}  [{step + 1:4d}/{N}]  "
                f"L_uv={val_uv.item():.5f}  L_sp={val_speed.item():.5f}  "
                f"L_gr={val_grad.item():.5f}  L_cr={val_crit.item():.5f}  "
                f"L_ls={val_levelset.item():.5f}  "
                f"L_cv={val_ttkcv.item():.5f}  L_pers={val_ttkpers.item():.5f}  "
                f"total={loss.item():.5f}"
            )

    means = {k: v / N for k, v in running.items()}
    print(
        f"epoch {epoch} DONE  "
        f"mean L_uv={means['uv']:.5f}  L_sp={means['speed']:.5f}  "
        f"L_gr={means['grad']:.5f}  L_cr={means['crit']:.5f}  "
        f"L_ls={means['levelset']:.5f}  L_cv={means['ttkcv']:.5f}  "
        f"L_pers={means['ttkpers']:.5f}  total={means['total']:.5f}"
    )
    return means


# ── Evaluation ─────────────────────────────────────────────────────────────────

def run_eval(
    model: RefinerNet,
    sr_all: np.ndarray,
    gt_all: np.ndarray,
    inp_all,
    idx_all,
    device: torch.device,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _check_protected(OUT_DIR, "OUT_DIR (eval)")

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

    np.save(OUT_DIR / "dataSR.npy", refined_hwc)
    np.save(OUT_DIR / "dataGT.npy", gt_hwc)

    if inp_all is not None:
        inp_hwc = (
            np.transpose(inp_all, (0, 2, 3, 1))
            if inp_all.ndim == 4 and inp_all.shape[1] == 2
            else inp_all
        )
        np.save(OUT_DIR / "dataIN.npy", inp_hwc)

    if idx_all is not None:
        np.save(OUT_DIR / "idx.npy", idx_all)

    print(f"[eval] Saved {N} refined samples to {OUT_DIR}")


# ── Output shape validation ────────────────────────────────────────────────────

def _validate_outputs() -> None:
    print("\n[validate] Checking output shapes …")
    expected = {
        "idx.npy":    (N_EVAL,),
        "dataIN.npy": (N_EVAL, 100, 100, 2),
        "dataGT.npy": (N_EVAL, 500, 500, 2),
        "dataSR.npy": (N_EVAL, 500, 500, 2),
    }
    all_ok = True
    for fname, exp_shape in expected.items():
        p = OUT_DIR / fname
        if not p.exists():
            print(f"  [FAIL] {fname}: NOT FOUND")
            all_ok = False
            continue
        arr = np.load(p, mmap_mode="r")
        if arr.shape != exp_shape:
            print(f"  [FAIL] {fname}: shape {arr.shape} != {exp_shape}")
            all_ok = False
        else:
            print(f"  [OK]   {fname}: {arr.shape}")

    if not all_ok:
        sys.exit("[error] Output shape validation failed.")
    print("[validate] All outputs OK.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    tee = _Tee(LOG_PATH)

    print("=" * 64)
    print("CandidateE2-expanded-672  TTK Critical-Pair Refiner")
    print("=" * 64)
    print(f"  TRAIN_CNN_DIR   : {TRAIN_CNN_DIR}")
    print(f"  EVAL_CNN_DIR    : {EVAL_CNN_DIR}")
    print(f"  CONSTRAINTS_NPZ : {CONSTRAINTS_NPZ}")
    print(f"  MODEL_DIR       : {MODEL_DIR}")
    print(f"  OUT_DIR         : {OUT_DIR}")
    print(f"  LR              : {LR}")
    print(f"  EPOCHS          : {EPOCHS}")
    print(f"  RESIDUAL_SCALE  : {RESIDUAL_SCALE}")
    print(f"  LAMBDA_SPEED    : {LAMBDA_SPEED}")
    print(f"  LAMBDA_GRAD     : {LAMBDA_GRAD}")
    print(f"  LAMBDA_CRIT     : {LAMBDA_CRIT}")
    print(f"  LAMBDA_LEVELSET : {LAMBDA_LEVELSET}")
    print(f"  LAMBDA_TTKCV    : {LAMBDA_TTKCV}")
    print(f"  LAMBDA_TTKPERS  : {LAMBDA_TTKPERS}")
    print(f"  SEED            : {SEED}")
    print()

    # ── Pre-conditions ────────────────────────────────────────────────────
    _check_preconditions()

    # ── Seeds ─────────────────────────────────────────────────────────────
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load training data (672 expanded) ─────────────────────────────────
    print("\n[phase 1] Loading expanded training data …")
    sr_train, gt_train, _, idx_train = _load_cnn_dir(TRAIN_CNN_DIR, "TRAIN_CNN_DIR")
    print(
        f"  SR_train: {sr_train.shape}  GT_train: {gt_train.shape}  "
        f"idx_train: {idx_train.shape if idx_train is not None else 'None'}"
    )

    # ── Load constraints ──────────────────────────────────────────────────
    print("\n[constraints] Loading constraints NPZ …")
    constraints = TTKConstraints(CONSTRAINTS_NPZ, patch=PATCH, device=device)

    # Verify all training indices are present in constraints
    if idx_train is not None:
        missing_ids = [
            int(x) for x in idx_train
            if int(x) not in constraints.available_indices
        ]
        if missing_ids:
            sys.exit(
                f"[error] {len(missing_ids)} training sample IDs not in constraints NPZ.\n"
                f"  Missing (first 10): {missing_ids[:10]}\n"
                "  Re-run: python3 scripts/build_candidateE2_fixed_constraints.py"
            )
        print(f"  All {len(idx_train)} training indices found in constraints.")

    # ── Model ─────────────────────────────────────────────────────────────
    model = RefinerNet(hidden=32, kernel=3, residual_scale=RESIDUAL_SCALE).to(device)
    print(f"\nRefinerNet: {model.n_params()} parameters")

    # ── TTK gradient sanity check ─────────────────────────────────────────
    print("\n[startup] Verifying L_ttkcv gradient flow …")
    model.zero_grad()
    sr_t0 = torch.tensor(sr_train[:1], device=device)
    gt_t0 = torch.tensor(gt_train[:1], device=device)
    first_id = int(idx_train[0]) if idx_train is not None else 0
    b_yx, d_yx, b_tgt, d_tgt, gt_p = constraints.get(first_id)
    if b_yx.shape[0] > 0:
        ref0 = model(sr_t0)
        cv0  = l_ttkcv(ref0, b_yx, d_yx, b_tgt, d_tgt)
        _abort_nan(cv0, "L_ttkcv (startup check)")
        cv0.backward()
        has_grad = any(
            p.grad is not None and (p.grad != 0).any()
            for p in model.parameters()
        )
        if not has_grad:
            raise RuntimeError(
                "L_ttkcv produces no model gradients at startup. "
                "Check the constraint NPZ and loss implementation."
            )
        print(f"  L_ttkcv gradient flow: YES (n_pairs={b_yx.shape[0]})")
    else:
        print(f"  [warn] First sample {first_id} has 0 pairs; skipping gradient check.")
    model.zero_grad()

    # ── Phase 1: Training ─────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print("Phase 1: Training on 672 expanded samples")
    print(f"{'=' * 64}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _check_protected(MODEL_DIR, "MODEL_DIR")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    t_train_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch} / {EPOCHS} ===")
        run_epoch(
            epoch, model, optimizer, constraints,
            sr_train, gt_train, idx_train, device,
        )

        ckpt_path = MODEL_DIR / f"refiner_epoch{epoch:02d}.pt"
        torch.save(
            {"epoch": epoch, "state_dict": model.state_dict(),
             "lambda_speed": LAMBDA_SPEED, "lambda_grad": LAMBDA_GRAD,
             "lambda_crit": LAMBDA_CRIT, "lambda_levelset": LAMBDA_LEVELSET,
             "lambda_ttkcv": LAMBDA_TTKCV, "lambda_ttkpers": LAMBDA_TTKPERS},
            ckpt_path,
        )
        print(f"Checkpoint: {ckpt_path}")

    final_path = MODEL_DIR / "refiner_final.pt"
    torch.save(
        {"epoch": EPOCHS, "state_dict": model.state_dict()},
        final_path,
    )
    print(f"Final model: {final_path}")
    print(f"Training wall time: {time.perf_counter() - t_train_start:.1f}s")

    # ── Phase 2: Evaluation on 168-sample benchmark ───────────────────────
    print(f"\n{'=' * 64}")
    print("Phase 2: Evaluating on 168-sample benchmark")
    print(f"{'=' * 64}")

    sr_eval, gt_eval, inp_eval, idx_eval = _load_cnn_dir(EVAL_CNN_DIR, "EVAL_CNN_DIR")
    print(
        f"  SR_eval: {sr_eval.shape}  GT_eval: {gt_eval.shape}  "
        f"idx_eval: {idx_eval.shape if idx_eval is not None else 'None'}"
    )

    run_eval(model, sr_eval, gt_eval, inp_eval, idx_eval, device)

    # ── Validate outputs ──────────────────────────────────────────────────
    _validate_outputs()

    print("\n" + "=" * 64)
    print("CandidateE2-expanded-672 complete.")
    print(f"  Refined SR : {OUT_DIR / 'dataSR.npy'}")
    print(f"  Log        : {LOG_PATH}")
    print("=" * 64)

    tee.close()


if __name__ == "__main__":
    main()
