# Candidate D: PD-Wasserstein Loss Feasibility Audit

**Date:** 2026-05-13  
**Branch:** `claude/audit-phire-wind-magnitude-sfqdw`  
**Audit script:** `scripts/audit_candidateD_pd_loss_feasibility.py`

---

## Context

Candidates B and C fine-tune the PhIRE [u, v] vector CNN using TF1-native
proxy losses for topology.  Candidate C adds a critical-value proxy
(`L_crit`, `lambda_crit=0.001`) on top of Candidate B's speed/gradient/
level-set losses and produces the strongest merge-tree result so far:

- MT distance improves over CNN on **102 / 168** samples.
- **11 / 20** original MT-GAN cases recovered.
- But `L_crit` is a local-maxima critical-value proxy, **not** a true
  PD-Wasserstein loss.

Candidate D proposes:

```
L = L_uv + L_speed + L_grad + L_crit + λ_pd · L_PD
```

where `L_PD` is a differentiable persistence-diagram Wasserstein loss on
scalar wind speed `sqrt(u² + v²)`.  This audit assesses whether that is
achievable in the current environment.

---

## 1. Environment Summary

| Component | Status |
|-----------|--------|
| Python    | 3.11.15 |
| numpy     | 2.4.4 |
| scipy     | 1.17.1 |
| torch     | **NOT INSTALLED** |
| CUDA      | **NOT available** (no `nvidia-smi`, no `/usr/local/cuda`) |
| TensorFlow| **NOT INSTALLED** (training uses pre-built TF1 container) |

The training environment runs TF1 inside a Docker image (`phire-ttk:latest`)
that was built for the fixed CNN/GAN baseline and is **not available for
modification** in this audit session.  Native Python (`python3`) on the host
has no deep-learning framework installed.

---

## 2. Topology Package Availability

### 2.1 Non-differentiable PD (evaluation only)

| Package | Installed | Pip-installable | Notes |
|---------|-----------|-----------------|-------|
| `gudhi 3.12.0` | NO | **YES** (`pip install gudhi`, ~4.3 MB) | CubicalComplex + Wasserstein distance; **no autograd** |
| `vtk` (TTK Python) | NO | Requires Docker | Already used via `ttkPersistenceDiagramCmd` CLI |

`gudhi.CubicalComplex` was already used in `archive/phase_b/phase_b_persistence_eval.py`
and `archive/phase_b/plot_pd_example.py` for non-differentiable PD distance
evaluation.

### 2.2 Differentiable PD loss (training)

| Package | Installed | Pip-installable | Notes |
|---------|-----------|-----------------|-------|
| `torch_topological 0.1.9` | NO | **YES** (requires `torch ~532 MB`) | **Differentiable** via PyTorch autograd |
| `topologylayer` | NO | NO (not on PyPI for Py 3.11) | Older, limited 2D support |
| `giotto-tda` | NO | Requires build | Sklearn-based; no torch autograd |
| `ripser` | NO | YES but no autograd | Fast persistent homology; no gradients |

The **only package that provides a fully differentiable PD Wasserstein loss
suitable for training is `torch_topological`**.

### 2.3 Verified API (from upstream source)

```python
from torch_topological.nn import CubicalComplex, WassersteinDistance

cubical   = CubicalComplex()
w_loss_fn = WassersteinDistance(q=2)

# 2D scalar field (H×W) — matches our 160×160 speed patches
pd_sr = cubical(speed_sr_tensor)[0]   # persistence info, dim 0 = connected components
pd_gt = cubical(speed_gt_tensor)[0]

loss_pd = w_loss_fn(pd_sr, pd_gt)
loss_pd.backward()                     # gradients flow to speed_sr_tensor
```

`CubicalComplex` computes cubical persistence homology on 2D (or 3D) float
tensors directly; this matches our speed field shape `(160, 160)`.  The
`WassersteinDistance(q=2)` computes the W₂ distance between two persistence
diagrams and supports `backward()`.

---

## 3. Gradient Smoke Test

**Status: SKIPPED** — `torch` not installed in the host Python environment.

The test would run after `pip install torch torch-topological`:

```python
import torch
import numpy as np
from torch_topological.nn import CubicalComplex, WassersteinDistance

torch.manual_seed(0)
x = torch.tensor(np.random.rand(32, 32).astype("float32"), requires_grad=True)
y = torch.tensor(np.random.rand(32, 32).astype("float32"))

cubical  = CubicalComplex()
loss_fn  = WassersteinDistance(q=2)

loss = loss_fn(cubical(x)[0], cubical(y)[0])
loss.backward()

print(x.grad is not None)              # True
print((x.grad != 0).sum().item())      # > 0
```

To run the smoke test (no model training):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-topological
python3 scripts/audit_candidateD_pd_loss_feasibility.py
```

Expected CPU torch wheel: **~260 MB** (CPU-only index).  
The full CUDA wheel is **~532 MB** and would install CUDA runtime libraries;
unnecessary since CUDA is not available on this host.

---

## 4. TF1 Compatibility Assessment

**This is the critical blocker.**

The PhIRE training loop (`PhIREGANs.train_MSE`, `PhIREGANs.train_GAN`) uses
`tensorflow.compat.v1` with session mode (`sess.run`).  PyTorch autograd and
TF1 computation graphs are **entirely separate systems** and cannot share
gradient flow.

### Route A — Hybrid gradient injection (NOT recommended)

```
TF1 forward → numpy SR speed → PyTorch CubicalComplex → numpy gradient
→ inject via tf.py_func / tf.numpy_function into TF1 backward
```

`tf.py_func` wraps a Python function as a TF1 op but **does not propagate
gradients** through it — it returns a zero-gradient by default.  Making it
differentiable requires either:
- A custom TF1 C++ op with a registered gradient (weeks of C++ work), or
- Manual gradient tape injection via `tf.custom_gradient` (TF2 only).

**Risk: HIGH.  Not viable without significant infrastructure work.**

### Route B — PyTorch post-processing refiner (medium risk)

Design Candidate D as a **separate PyTorch network** that takes the frozen
TF1 SR output as input and refines it using `L_PD`:

```
TF1 PhIRE CNN (frozen) → numpy SR → PyTorch refiner → speed field
                                      ↑
                           L_PD (CubicalComplex + WassersteinDistance)
                           + L_uv reconstruction
```

This avoids touching the TF1 training loop.  The PyTorch refiner is a
small U-Net or residual block trained purely in PyTorch.

**Risk: MEDIUM.  Scientifically valid as a post-processing stage.
Does not fine-tune the CNN generator itself, only post-processes its output.**

### Route C — Full PyTorch port (high effort, enables end-to-end PD loss)

Rewrite `PhIREGANs.train_MSE()` in PyTorch, loading the pretrained CNN
weights (convertible via ONNX or manual weight copy).  Then `L_PD` integrates
as a standard loss term.

**Risk: MEDIUM-HIGH.  Estimated 2–3 weeks.  Enables true end-to-end PD
Wasserstein fine-tuning.  Scientifically the cleanest approach.**

### Route D — Stronger TF1-native proxy (low risk, no true PD)

Strengthen `L_crit` (Candidate C) with multi-scale max-pooling, or add a
second proxy that penalises persistence pairs by approximating birth/death
via max-pool cascades.  Does NOT give true PD Wasserstein loss but is
directly trainable in TF1 and builds on validated Candidate C results.

**Risk: LOW.  Incremental improvement only.  Scientifically less rigorous
but immediately executable.**

---

## 5. Existing Repo References

Topology code already present in the repository:

| File | Role |
|------|------|
| `archive/phase_b/phase_b_persistence_eval.py` | GUDHI CubicalComplex for non-differentiable PD eval |
| `archive/phase_b/plot_pd_example.py` | GUDHI PD scatter plots |
| `archive/old_scripts/compute_real_distances_compatible.py` | GUDHI Wasserstein W1/W2 + bottleneck |
| `scripts/compute_composite_tree_distance.py` | TTK CLI PD bottleneck + MT Wasserstein |
| `scripts/compute_alt_topology_distances.py` | TTK Python PD Wasserstein (Docker) |
| `sr_network.py` (`_critical_value_loss`) | TF1-native topological proxy (Candidate C) |

The GUDHI infrastructure from `archive/phase_b/` is the closest existing
implementation to a true PD loss, but it was used only for evaluation, not
for training.

---

## 6. Scientific Motivation Recap

From Kissi et al. 2025: topology-aware losses improve scalar field
super-resolution by penalising topological mismatches in persistence diagrams.

For wind speed:
- **PD loss** targets the global birth/death structure of superlevel sets
  (e.g. storm centres, large-scale circulation maxima).
- **MT loss** (not computable in TF1) targets hierarchical nesting.
- **`L_crit` proxy** (Candidate C) approximates PD by penalising MSE at
  detected maxima, without matching birth/death pairing.

A true `L_PD` via `torch_topological.CubicalComplex` would:
1. Correctly pair births and deaths using cubical homology.
2. Compute W₂ Wasserstein distance between GT and SR persistence diagrams.
3. Propagate subgradients that move SR critical values towards GT critical values.

The hypothesis that PD loss improves PD distance without necessarily
improving MT distance remains scientifically valid and worth testing.

---

## 7. Recommendation and Risk Assessment

| Route | Description | PD Wasserstein | TF1 compatible | Risk | Effort |
|-------|-------------|---------------|---------------|------|--------|
| D1 | Stronger TF1 proxy (multi-scale L_crit) | No (proxy) | Yes | LOW | 1–2 days |
| D2 | PyTorch post-processing refiner | Yes (post-proc) | N/A | MEDIUM | 1 week |
| D3 | Full PyTorch port + L_PD training | Yes (end-to-end) | No (replaces TF1) | MEDIUM-HIGH | 2–3 weeks |
| D4 | Hybrid gradient injection | Fragile | Partial | HIGH | weeks + fragile |

**Recommended path: D1 (immediate) + D2 (next sprint)**

1. **Candidate D (immediate):** Implement Candidate D as Candidate C + a
   **multi-scale critical-value loss** in TF1:
   - Add a second `L_crit` scale with larger pool size (e.g. `crit_pool=7`)
   - Optionally include minima (`crit_include_minima=True`)
   - Optionally add a persistence-proxy term: penalise the gap between
     the `crit_pool=3` max and the `crit_pool=7` max (approximates persistence)
   - Risk: LOW — incremental extension of validated Candidate C code
   - Scientific value: demonstrates whether scale matters for topological proxy

2. **Candidate D-PyTorch (next sprint):** Install `torch` + `torch_topological`
   and design a post-processing refiner that uses true `L_PD`:
   - Input: frozen CNN SR output (160×160×2 u,v)
   - Compute speed, pass through CubicalComplex, minimise WassersteinDistance(q=2)
   - Evaluate using existing TTK pipeline (PD distance + MT distance)
   - Risk: MEDIUM — requires torch install and new PyTorch training script,
     but does NOT touch TF1 code

---

## 8. Next Task Prompt (if proceeding with D1)

```
Task D1: Implement Candidate D multi-scale critical-value loss in TF1.

Context:
- Candidate C uses a single-scale critical-value proxy with crit_pool=3,
  lambda_crit=0.001, crit_high_z=1.0, crit_include_minima=False.
- The feasibility audit (docs/candidateD_pd_loss_feasibility.md) recommends
  Candidate D as Candidate C + multi-scale topological proxy as the
  lowest-risk next step.

Changes:
1. In sr_network.py PhysicsLossConfig:
   Add: crit_pool2=7, lambda_crit2=0.001 (second scale of L_crit)
   Add: lambda_persist=0.0005 (persistence-gap proxy term)

2. In sr_network.py _build_aux_losses:
   Compute L_crit2 = _critical_value_loss(speed_hr, speed_sr, high_z=1.0,
       include_minima=True, low_z=-1.0, pool=7)
   Compute L_persist = persistence-gap proxy:
       maxima_coarse = tf.nn.max_pool2d(speed_hr, ksize=7, strides=1, padding='SAME')
       maxima_fine   = tf.nn.max_pool2d(speed_hr, ksize=3, strides=1, padding='SAME')
       persistence_map = tf.nn.relu(maxima_coarse - maxima_fine)
       L_persist = tf.reduce_mean(tf.abs(persistence_map * (speed_hr - speed_sr)))
   weighted_total += lambda_crit2 * L_crit2 + lambda_persist * L_persist

3. Create scripts/run_candidateD_finetune.py (copy of run_candidateC_crit_finetune.py)
   with updated lambda values and MODEL_OUT/DATA_OUT pointing to candidateD.

4. Run py_compile, commit, push.
Do not run training yet.
```

## 9. Next Task Prompt (if proceeding with D2 — PyTorch refiner)

```
Task D2: Install torch + torch_topological and run gradient smoke test.

Steps:
1. pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install torch-topological

2. Run: python3 scripts/audit_candidateD_pd_loss_feasibility.py
   Verify gradient smoke test passes (x.grad != None, n_nonzero > 0).

3. Create scripts/run_candidateD_pytorch_refiner.py:
   - Load data_out/wind_finetune_pilot_candidateC/dataSR.npy (168×160×160×2)
   - Load data_out/wind_finetune_pilot_candidateC/dataGT.npy
   - Define a small PyTorch residual refiner (3 conv layers) operating on speed
   - Loss: alpha * L_recon_uv + beta * L_PD (CubicalComplex + WassersteinDistance)
   - Train for 3 epochs on all 168 samples
   - Save refined SR as data_out/wind_finetune_pilot_candidateD/dataSR.npy

4. Run py_compile + smoke test, commit, push.
   Do not run full training until loss calibration is verified.
```

---

*Audit performed by `scripts/audit_candidateD_pd_loss_feasibility.py` on 2026-05-13.*
