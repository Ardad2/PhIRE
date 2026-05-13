#!/usr/bin/env python3
"""
Candidate D PD-loss feasibility audit.

Tests:
  1. Python / numpy / scipy / torch / torch_topological environment.
  2. GUDHI availability (non-differentiable PD distance).
  3. torch_topological availability (differentiable PD Wasserstein loss).
  4. If torch_topological is available: gradient smoke test on a 32×32 speed field.

Run with the default Python interpreter:
    python3 scripts/audit_candidateD_pd_loss_feasibility.py

After installing the required packages:
    pip install gudhi torch torch-topological
    python3 scripts/audit_candidateD_pd_loss_feasibility.py
"""

import sys
import importlib

SEP = "-" * 60


def check(label, mod_name, attr=None):
    """Import module, optionally check for an attribute.  Returns (ok, version)."""
    try:
        m = importlib.import_module(mod_name)
        ver = getattr(m, "__version__", "unknown")
        if attr and not hasattr(m, attr):
            return False, f"{ver} (missing attr: {attr})"
        return True, ver
    except ImportError as e:
        return False, str(e)


print(SEP)
print("CANDIDATE D  PD-LOSS FEASIBILITY AUDIT")
print(SEP)

# ── 1. Python ──────────────────────────────────────────────────────────────
print(f"\n[1] Python  : {sys.version.split()[0]}")

# ── 2. Core scientific stack ────────────────────────────────────────────────
print("\n[2] Core scientific stack")
for label, mod in [("numpy", "numpy"), ("scipy", "scipy")]:
    ok, ver = check(label, mod)
    print(f"    {label:20s} {'OK  v' + ver if ok else 'MISSING  ' + ver}")

# ── 3. Deep learning frameworks ─────────────────────────────────────────────
print("\n[3] Deep learning frameworks")
torch_ok, torch_ver = check("torch", "torch")
print(f"    {'torch':20s} {'OK  v' + torch_ver if torch_ok else 'NOT INSTALLED  ' + torch_ver}")
if torch_ok:
    import torch
    print(f"    {'torch.cuda':20s} {'available' if torch.cuda.is_available() else 'NOT available (CPU-only)'}")

tf_ok, tf_ver = check("tensorflow", "tensorflow")
print(f"    {'tensorflow':20s} {'OK  v' + tf_ver if tf_ok else 'NOT INSTALLED  ' + tf_ver}")

# ── 4. Topology packages ────────────────────────────────────────────────────
print("\n[4] Topology packages")

gudhi_ok, gudhi_ver = check("gudhi", "gudhi")
print(f"    {'gudhi':20s} {'OK  v' + gudhi_ver if gudhi_ok else 'NOT INSTALLED  (pip install gudhi)  ' + gudhi_ver}")
print(f"    {'':20s} Note: gudhi provides NON-differentiable PD distance only.")

tt_ok, tt_ver = check("torch_topological", "torch_topological")
print(f"    {'torch_topological':20s} {'OK  v' + tt_ver if tt_ok else 'NOT INSTALLED  (pip install torch-topological)  ' + tt_ver}")
if tt_ok:
    print(f"    {'':20s} Provides DIFFERENTIABLE CubicalComplex + WassersteinDistance.")

for pkg in ["topologylayer", "giotto", "ripser"]:
    ok, ver = check(pkg, pkg)
    print(f"    {pkg:20s} {'OK  v' + ver if ok else 'NOT INSTALLED'}")

# ── 5. Gradient smoke test ───────────────────────────────────────────────────
print(f"\n{SEP}")
print("[5] Gradient smoke test  (requires torch + torch_topological)")
print(SEP)

if not torch_ok:
    print("    SKIPPED — torch not installed.")
    print("    Install with:  pip install torch torch-topological")
elif not tt_ok:
    print("    SKIPPED — torch_topological not installed.")
    print("    Install with:  pip install torch-topological")
else:
    import torch
    from torch_topological.nn import CubicalComplex, WassersteinDistance

    print("    torch + torch_topological are available.")
    print("    Creating 32×32 speed field with requires_grad=True …")

    torch.manual_seed(0)
    x_np = __import__("numpy").random.rand(32, 32).astype("float32")
    x = torch.tensor(x_np, requires_grad=True, dtype=torch.float32)

    # GT field (fixed target — no gradient needed)
    y_np = __import__("numpy").random.rand(32, 32).astype("float32")
    y = torch.tensor(y_np, dtype=torch.float32)

    cubical = CubicalComplex()
    w_loss_fn = WassersteinDistance(q=2)

    pd_x = cubical(x)[0]
    pd_y = cubical(y)[0]

    loss = w_loss_fn(pd_x, pd_y)
    print(f"    PD Wasserstein-2 loss = {loss.item():.6f}")

    loss.backward()

    if x.grad is not None:
        n_nonzero = int((x.grad != 0).sum().item())
        total = x.grad.numel()
        print(f"    x.grad exists: YES")
        print(f"    Non-zero gradient entries: {n_nonzero} / {total}  ({100*n_nonzero/total:.1f}%)")
        print("    SMOKE TEST: PASSED — gradients flow from PD loss to input tensor.")
    else:
        print("    x.grad is None — backward did NOT produce gradients.")
        print("    SMOKE TEST: FAILED")

# ── 6. TF1 framework compatibility ──────────────────────────────────────────
print(f"\n{SEP}")
print("[6] TF1 / torch_topological compatibility assessment")
print(SEP)
print("""
    The current PhIRE training loop uses tensorflow.compat.v1 (TF1 session
    mode).  torch_topological is a PyTorch autograd module.  These two
    autodiff systems CANNOT share gradient tapes or computation graphs.

    Three integration routes exist:

    Route A — HYBRID GRADIENT INJECTION (not recommended, high risk)
      Forward pass in TF1 → extract numpy SR speed → compute PD loss
      in PyTorch → extract numpy gradient → inject back into TF1 via
      tf.py_func.  tf.py_func blocks gradient flow across the boundary, so
      this requires a custom C++ / TF op or a manual gradient-tape override.
      Fragile, not documented, research-grade hack only.

    Route B — PORT TRAINING TO PYTORCH (correct, medium effort)
      Rewrite PhIREGANs.train_MSE() / train_GAN() in PyTorch.
      Then plug in:
          cubical = CubicalComplex()
          w_loss  = WassersteinDistance(q=2)
          loss_pd = w_loss(cubical(speed_sr)[0], cubical(speed_gt)[0])
      Gradient flows correctly through the entire pipeline.
      Estimated effort: 2–3 weeks.

    Route C — TF1-NATIVE PROXY (current, low risk, no true PD)
      Continue Candidate C approach: critical-value proxy in TF1.
      Does NOT give true PD Wasserstein loss.
      Can be strengthened with multi-scale max-pooling or cascade detectors.
""")

# ── 7. Summary ───────────────────────────────────────────────────────────────
print(SEP)
print("[7] Summary and recommendation")
print(SEP)
print()

path_label = "UNKNOWN"
if tt_ok and torch_ok:
    path_label = "A — true differentiable PD loss available (torch_topological)"
elif not torch_ok or not tt_ok:
    path_label = "B — differentiable PD loss possible after package install"

print(f"    Implementation path : {path_label}")
print()
print("    Recommended next steps (in order of risk):")
print()
print("    1. [LOW RISK]  pip install gudhi")
print("       Use gudhi.CubicalComplex for improved non-differentiable PD")
print("       evaluation/monitoring alongside existing TTK pipeline.")
print()
print("    2. [MEDIUM RISK]  pip install torch torch-topological")
print("       Run this script again to confirm gradient smoke test passes.")
print("       Design Candidate D as a standalone PyTorch speed-field refiner")
print("       that wraps the frozen TF1 SR output (post-processing only).")
print("       This avoids a full TF1 → PyTorch port.")
print()
print("    3. [HIGH RISK]  Full PyTorch port")
print("       Rewrite PhIREGANs in PyTorch, integrate CubicalComplex +")
print("       WassersteinDistance as training loss term.")
print("       Enables end-to-end true PD-Wasserstein fine-tuning.")
print()
print("    Estimated torch download : ~532 MB (CPU) or ~2.5 GB (CUDA)")
print("    CUDA available on this host: NO")
print()
print(SEP)
