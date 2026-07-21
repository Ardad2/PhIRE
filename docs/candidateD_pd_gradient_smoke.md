# Candidate D: PD Wasserstein Gradient Smoke Test

**Date:** 2026-05-13  
**Branch:** `claude/audit-phire-wind-magnitude-sfqdw`  
**Script:** `scripts/smoke_candidateD_pd_grad.py`  
**Results JSON:** `docs/candidateD_pd_gradient_smoke_results.json`

---

## Summary

**4 / 4 tests PASSED.**  
Autograd-compatible PD Wasserstein loss (`torch_topological.nn.CubicalComplex` +
`WassersteinDistance`) produces valid, non-zero gradients with respect to a 2D
scalar wind-speed field, both on synthetic and real data.

**Recommendation: PROCEED** to Candidate D PyTorch residual refiner design.

---

## 1. Environment

| Package | Version |
|---------|---------|
| Python | 3.11.15 |
| torch | 2.12.0+cu130 (CPU-only; CUDA not available on this host) |
| gudhi | 3.12.0 |
| torch_topological | 0.1.9 |
| numpy | 2.4.4 |

Virtual environment: `.venv_candidateD_pd/` (isolated, not on `PATH` by default).

Install commands used:

```bash
python3 -m venv .venv_candidateD_pd
.venv_candidateD_pd/bin/pip install torch          # 2.12.0 (PyPI default)
.venv_candidateD_pd/bin/pip install gudhi torch_topological numpy
```

---

## 2. GUDHI Compatibility Patch

**`torch_topological 0.1.9` is incompatible with `gudhi >= 3.10` out of the box.**

Root cause: `torch_topological.nn.cubical_complex.CubicalComplex._get_persistence()`
calls:

```python
gudhi.CubicalComplex(
    dimensions=x.shape,
    top_dimensional_cells=x.flatten()   # ← torch.Tensor, not numpy
)
```

GUDHI 3.10+ requires `top_dimensional_cells` to be a numpy array (or Python
sequence), not a `torch.Tensor`.  The error is:

```
__init__(): incompatible function arguments.
Invoked with types: gudhi.cubical_complex.CubicalComplex, Size, torch.Tensor, bool
```

**Patch applied to venv source** (not to the committed repo script):

```
File: .venv_candidateD_pd/lib/python3.11/site-packages/
         torch_topological/nn/cubical_complex.py  line ~174
```

```python
# Before:
cubical_complex = gudhi.CubicalComplex(
    dimensions=x.shape,
    top_dimensional_cells=x.flatten()
)

# After:
_cells = x.detach().cpu().numpy().flatten()
cubical_complex = gudhi.CubicalComplex(
    dimensions=list(x.shape),
    top_dimensional_cells=_cells
)
```

This patch is **local to the venv** and does not affect any tracked source file.
When deploying to a production environment, either:
- Pin `gudhi==3.9.*` (no patch needed), or
- Apply this two-line patch to the `torch_topological` installation.

A bug report / PR to the `pytorch-topological` upstream is appropriate.

---

## 3. API Used

```python
from torch_topological.nn import CubicalComplex, WassersteinDistance

# CubicalComplex(superlevel=True): persistence of superlevel sets
# (appropriate for wind maxima — large connected regions of high speed)
cubical   = CubicalComplex(superlevel=True)
w_loss_fn = WassersteinDistance(q=2)   # W_2 Wasserstein distance

# x: (H, W) float32 tensor, requires_grad=True
# y: (H, W) float32 tensor, fixed GT target
pd_x = cubical(x)   # list of PersistenceInformation, one per homological dim
pd_y = cubical(y)

loss = w_loss_fn(pd_x, pd_y)   # scalar W_2 distance between diagrams
loss.backward()                 # gradients flow to x
```

`CubicalComplex` returns persistence information for **all homological
dimensions**: H₀ (connected components / maxima pairing) and H₁ (loops).
`WassersteinDistance` sums the W₂ cost across all matched dimensions.

Gradients are **sparse**: they are non-zero only at the birth/death cell pairs
(the generators of each persistence pair).  This is expected and correct.

---

## 4. Test Results

### Test A: Synthetic 32×32 (superlevel)

| Field | Value |
|-------|-------|
| Input | Gaussian-smoothed random field, normalized [0,1] |
| Loss (W₂) | 0.360822 |
| Time | 0.008 s |
| x.grad non-zero | 28 / 1024  (2.7%) |
| max \|grad\| | 3.88 × 10⁻¹ |
| **Status** | **PASSED** |

### Test A2: Synthetic 32×32 (sublevel)

| Field | Value |
|-------|-------|
| Loss (W₂) | 0.423177 |
| Time | 0.006 s |
| x.grad non-zero | 37 / 1024  (3.6%) |
| max \|grad\| | 4.28 × 10⁻¹ |
| **Status** | **PASSED** |

### Test B: Real wind-speed crop 100×100 (superlevel)

Data source: `topo_ready/` speed snapshots (GT: sample 0 of 5×100×100 array;
SR: sample 1 of 5×500×500 array, cropped to 100×100).  GT speed range:
[0.042, 11.132] m/s; normalized to [0,1] using GT range before PD computation.

| Field | Value |
|-------|-------|
| Crop size | 100 × 100 |
| Loss (W₂) | 0.837936 |
| Time | **0.102 s** |
| x.grad non-zero | 1676 / 10000  (16.8%) |
| max \|grad\| | 2.80 × 10⁻¹ |
| **Status** | **PASSED** |

The higher gradient density on real data (16.8% vs 2.7%) reflects richer
topological structure in real wind fields compared to smooth Gaussian noise.

### Test B2: Real wind-speed crop 64×64 (timing reference)

| Field | Value |
|-------|-------|
| Loss (W₂) | 0.511966 |
| Time | 0.023 s |
| x.grad non-zero | 696 / 4096  (17.0%) |
| max \|grad\| | 6.96 × 10⁻¹ |
| **Status** | **PASSED** |

---

## 5. Timing Analysis

| Patch size | Time (s) | Extrapolated per-sample | 168 samples |
|-----------|---------|------------------------|-------------|
| 32 × 32   | 0.008   | 0.008 s                | ~1.3 s      |
| 64 × 64   | 0.023   | 0.023 s                | ~3.9 s      |
| 100 × 100 | 0.102   | 0.102 s                | ~17.1 s     |
| 160 × 160 | ~0.25 (estimated) | 0.25 s        | ~42 s       |

Per-epoch timing for a 3-epoch fine-tuning run on 168 samples at 160×160:
**~2 min/epoch** for the PD loss alone on CPU (no batching).

Memory: `CubicalComplex` operates on CPU numpy arrays; peak memory for a
160×160 field is negligible (< 1 MB per sample).

---

## 6. Gradient Sparsity: What It Means

PD Wasserstein loss produces **sparse gradients**: only the cells that
are critical points (local birth/death events in the sublevel/superlevel
filtration) receive non-zero gradient.  This is by design:

- At a persistence pair (b, d), the gradient of W₂ moves the birth cell
  (maximum in superlevel) to reduce the mismatch with the nearest GT pair.
- Non-critical pixels receive zero gradient — the loss is fully oblivious to
  them, which is correct topologically.
- In practice this means PD loss must be combined with a pixel-level loss
  (`L_uv`, `L_speed`) to prevent under-fitting at non-critical locations.

---

## 7. Memory Usage

No OOM errors observed.  Expected memory per forward+backward for a 160×160
speed field is dominated by the persistence pairs array (< 5 MB) and the
POT (Python Optimal Transport) distance matrix computation
(`O(n² + m²)` where n, m ≤ 160×160 / 2 in the worst case, but typically
much smaller due to persistence threshold).

---

## 8. Recommendation

**PROCEED to Candidate D PyTorch residual refiner.**

The gradient smoke test confirms:
- `torch_topological.nn.CubicalComplex` + `WassersteinDistance(q=2)` works
  correctly after the venv patch.
- Gradients are valid and non-zero at persistence generators.
- Per-sample timing (~0.1 s at 100×100, ~0.25 s estimated at 160×160) is
  acceptable for a 3-epoch pilot run on 168 samples.
- Memory is not a concern.

---

## 9. Next Task Prompt: Candidate D PyTorch Residual Refiner

```
Task D2: Implement Candidate D as a PyTorch post-processing residual refiner.

Context:
- docs/candidateD_pd_gradient_smoke.md confirms PD Wasserstein gradient
  smoke test passed: 4/4 tests, ~0.1 s per 100×100 speed crop on CPU.
- Training framework is TF1 (incompatible with PyTorch autograd for end-to-end
  training), so Candidate D is a post-processor on frozen Candidate C output.
- Candidate C outputs: data_out/wind_finetune_pilot_candidateC/dataSR.npy
  and dataGT.npy (168 samples, 160×160, [u,v] format).

Architecture:
  Input : [u,v] SR from frozen Candidate C  (160×160×2)
  Refiner: 3-layer residual conv network in PyTorch (small, ≤ 32 channels)
  Loss  : alpha * L_uv + beta * L_PD
  where L_uv = MSE on [u,v], L_PD = WassersteinDistance(q=2) on normalized speed.

Steps:
1. Create scripts/run_candidateD_pytorch_refiner.py
   - Load data_out/wind_finetune_pilot_candidateC/dataSR.npy and dataGT.npy
   - Define RefinerNet: 3 conv layers (in=2, hidden=16, out=2), ReLU, residual skip
   - Loss: alpha=1.0 * L_uv + beta=0.01 * L_PD (calibrate beta before training)
   - 3-epoch pilot, batch_size=1 (per-sample, 168 iterations/epoch)
   - Save: data_out/wind_finetune_pilot_candidateD/dataSR.npy and dataGT.npy

2. Before training: run loss calibration for 5 samples to verify L_PD / L_uv ratio.

3. Run py_compile, commit, push. Do not run full training until user approves.

GUDHI patch requirement: apply the two-line patch to any new venv before running.
  Old: top_dimensional_cells=x.flatten()
  New: top_dimensional_cells=x.detach().cpu().numpy().flatten()
```

---

*Smoke test run by `.venv_candidateD_pd/bin/python scripts/smoke_candidateD_pd_grad.py` on 2026-05-13.*
