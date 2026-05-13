#!/usr/bin/env python3
"""
Candidate D: PD Wasserstein gradient smoke test.

Tests autograd-compatible PD loss (torch_topological.nn.CubicalComplex +
WassersteinDistance) on:
  A. Synthetic 32×32 smooth random field (superlevel persistence).
  A2. Synthetic 32×32 (sublevel persistence — complement check).
  B. Real wind-speed crop from topo_ready/ (sample 0 vs sample 1, 160×160).
  B2. Real wind-speed crop 64×64 — timing reference.

Prerequisites (must be run inside .venv_candidateD_pd):
    .venv_candidateD_pd/bin/python scripts/smoke_candidateD_pd_grad.py

Compatibility note
------------------
torch_topological 0.1.9 passes a torch.Tensor to gudhi.CubicalComplex(),
but gudhi >= 3.10 requires a numpy array.  A one-line patch is applied to
the venv source at:
    .venv_candidateD_pd/lib/python3.11/site-packages/
        torch_topological/nn/cubical_complex.py
  Old:  top_dimensional_cells=x.flatten()
  New:  top_dimensional_cells=x.detach().cpu().numpy().flatten()
This is documented in docs/candidateD_pd_gradient_smoke.md.
"""

import sys
import time
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch_topological.nn import CubicalComplex, WassersteinDistance

REPO_ROOT = Path(__file__).resolve().parent.parent
SEP = "=" * 60


# ── helpers ────────────────────────────────────────────────────────────────

def _normalize_01(arr: np.ndarray, ref: np.ndarray = None, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize arr to [0, 1] using ref's range (default: arr itself)."""
    if ref is None:
        ref = arr
    mn, mx = float(ref.min()), float(ref.max())
    return np.clip((arr - mn) / max(mx - mn, eps), 0.0, 1.0).astype(np.float32)


def run_pd_test(
    name: str,
    x_np: np.ndarray,
    y_np: np.ndarray,
    q: int = 2,
    superlevel: bool = True,
) -> dict:
    """
    Compute PD Wasserstein-q loss between x and y, call backward, report.

    x_np : (H, W) float32 — SR / query field (requires_grad will be set)
    y_np : (H, W) float32 — GT / target field (no gradient needed)
    Returns a dict with diagnostic fields.
    """
    result = {
        "name": name,
        "passed": False,
        "loss": None,
        "n_nonzero_grad": 0,
        "total_grad_entries": int(x_np.size),
        "max_abs_grad": None,
        "elapsed_s": None,
        "error": None,
    }

    try:
        x = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
        y = torch.tensor(y_np, dtype=torch.float32)

        cubical = CubicalComplex(superlevel=superlevel)
        w_loss_fn = WassersteinDistance(q=q)

        t0 = time.perf_counter()
        pd_x = cubical(x)
        pd_y = cubical(y)
        loss = w_loss_fn(pd_x, pd_y)
        loss.backward()
        elapsed = time.perf_counter() - t0

        result["elapsed_s"] = round(elapsed, 4)
        result["loss"] = float(loss.item())

        if x.grad is not None:
            g = x.grad.detach()
            nz = int((g != 0).sum().item())
            result["n_nonzero_grad"] = nz
            result["max_abs_grad"] = float(g.abs().max().item())
            result["passed"] = nz > 0
        else:
            result["error"] = "x.grad is None after backward()"

    except Exception as exc:
        result["error"] = str(exc)

    return result


def print_result(r: dict) -> None:
    status = "PASSED" if r["passed"] else "FAILED"
    print(f"  Status          : {status}")
    if r["loss"] is not None:
        print(f"  Loss (W_2)      : {r['loss']:.6f}")
    if r["elapsed_s"] is not None:
        print(f"  Time            : {r['elapsed_s']:.3f} s")
    if r["n_nonzero_grad"] > 0:
        total = r["total_grad_entries"]
        nz = r["n_nonzero_grad"]
        pct = 100.0 * nz / total
        print(f"  x.grad nonzero  : {nz} / {total}  ({pct:.1f}%)")
        print(f"  max |grad|      : {r['max_abs_grad']:.6e}")
    if r["error"]:
        print(f"  ERROR           : {r['error']}")


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print(SEP)
    print("CANDIDATE D  PD WASSERSTEIN GRADIENT SMOKE TEST")
    print(SEP)

    import gudhi
    import torch_topological
    print(f"\nPython           : {sys.version.split()[0]}")
    print(f"torch            : {torch.__version__}")
    print(f"torch.cuda       : {'available' if torch.cuda.is_available() else 'NOT available (CPU)'}")
    print(f"gudhi            : {gudhi.__version__}")
    print(f"torch_topological: {torch_topological.__version__}")
    print(f"numpy            : {np.__version__}")

    results = []

    # ── Test A: synthetic 32×32 superlevel ──────────────────────────────
    print(f"\n{'-'*40}")
    print("Test A: Synthetic 32×32  (superlevel — wind maxima)")
    print(f"{'-'*40}")

    rng = np.random.default_rng(seed=42)
    x_syn = _normalize_01(gaussian_filter(rng.standard_normal((32, 32)).astype(np.float32), sigma=2.0))
    y_syn = _normalize_01(gaussian_filter(rng.standard_normal((32, 32)).astype(np.float32), sigma=2.0))

    r_a = run_pd_test("synthetic_32x32_superlevel", x_syn, y_syn, q=2, superlevel=True)
    print_result(r_a)
    results.append(r_a)

    # ── Test A2: sublevel ────────────────────────────────────────────────
    print(f"\n{'-'*40}")
    print("Test A2: Synthetic 32×32  (sublevel — complement check)")
    print(f"{'-'*40}")

    r_a2 = run_pd_test("synthetic_32x32_sublevel", x_syn, y_syn, q=2, superlevel=False)
    print_result(r_a2)
    results.append(r_a2)

    # ── Tests B / B2: real wind-speed data ───────────────────────────────
    # Use topo_ready/ speed snapshots (available in-repo).
    # Two different snapshot files provide SR and GT analogues.
    speed_files = sorted((REPO_ROOT / "topo_ready").glob("*dataSR_speed.npy"))

    if len(speed_files) < 1:
        print(f"\n  WARNING: no speed files found in {REPO_ROOT / 'topo_ready'}")
        print("           Tests B/B2 will be skipped.")
        skip_b = True
    else:
        # Load two files for GT and SR; fall back to two samples from same file
        f_gt = speed_files[0]
        f_sr = speed_files[-1] if len(speed_files) > 1 else speed_files[0]
        arr_gt = np.load(f_gt)  # (N, H, W) speed
        arr_sr = np.load(f_sr)
        # Use sample 0 for GT and sample 1 (or 0) for SR
        gt_full = arr_gt[0].astype(np.float32)
        sr_full = arr_sr[min(1, arr_sr.shape[0] - 1)].astype(np.float32)
        print(f"\n  GT file  : {f_gt.name}  shape={arr_gt.shape}")
        print(f"  SR file  : {f_sr.name}  shape={arr_sr.shape}")
        skip_b = False

    # Test B: 160×160 crop
    print(f"\n{'-'*40}")
    print("Test B: Real wind-speed crop  160×160  (superlevel)")
    print(f"{'-'*40}")

    if skip_b:
        print("  SKIPPED — no speed data available.")
        r_b = {"name": "wind_speed_160x160", "passed": False, "loss": None,
               "n_nonzero_grad": 0, "total_grad_entries": 160 * 160,
               "max_abs_grad": None, "elapsed_s": None,
               "error": "no speed data in topo_ready/"}
    else:
        H = min(160, gt_full.shape[0])
        W = min(160, gt_full.shape[1])
        gt_crop = gt_full[:H, :W]
        sr_crop = sr_full[:H, :W] if sr_full.shape[0] >= H and sr_full.shape[1] >= W else gt_full[1:H+1, :W]
        sr_norm = _normalize_01(sr_crop, ref=gt_crop)
        gt_norm = _normalize_01(gt_crop)
        print(f"  Crop size        : {H}×{W}")
        print(f"  GT speed range   : [{gt_crop.min():.3f}, {gt_crop.max():.3f}]")
        print(f"  SR speed range   : [{sr_crop.min():.3f}, {sr_crop.max():.3f}]")
        r_b = run_pd_test("wind_speed_160x160", sr_norm, gt_norm, q=2, superlevel=True)

    print_result(r_b)
    results.append(r_b)

    # Test B2: 64×64 timing reference
    print(f"\n{'-'*40}")
    print("Test B2: Real wind-speed crop  64×64  (timing reference)")
    print(f"{'-'*40}")

    if skip_b:
        print("  SKIPPED.")
        r_b2 = {"name": "wind_speed_64x64", "passed": False, "loss": None,
                "n_nonzero_grad": 0, "total_grad_entries": 64 * 64,
                "max_abs_grad": None, "elapsed_s": None, "error": "no data"}
    else:
        H2 = min(64, gt_full.shape[0])
        W2 = min(64, gt_full.shape[1])
        gt2 = gt_full[:H2, :W2]
        sr2 = sr_full[:H2, :W2] if sr_full.shape[0] >= H2 else gt_full[1:H2+1, :W2]
        r_b2 = run_pd_test("wind_speed_64x64",
                            _normalize_01(sr2, ref=gt2),
                            _normalize_01(gt2),
                            q=2, superlevel=True)

    print_result(r_b2)
    results.append(r_b2)

    # ── Overall verdict ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    n_pass = sum(1 for r in results if r["passed"])
    n_total = len(results)
    print(f"VERDICT: {n_pass} / {n_total} tests passed")

    synthetic_ok = r_a["passed"]
    real_ok = r_b["passed"]

    if synthetic_ok and real_ok:
        verdict = "PROCEED — gradients flow on both synthetic and real wind-speed data."
    elif synthetic_ok and skip_b:
        verdict = "PARTIAL — synthetic gradient test passed; real data not available locally."
    elif synthetic_ok:
        verdict = "PARTIAL — synthetic gradient test passed; real crop test failed."
    else:
        verdict = "DO NOT PROCEED — gradient smoke test failed on synthetic data."

    print(f"Recommendation  : {verdict}")
    print(SEP)

    # ── JSON summary ──────────────────────────────────────────────────────
    import gudhi, torch_topological as tt
    summary = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gudhi": gudhi.__version__,
        "torch_topological": tt.__version__,
        "numpy": np.__version__,
        "gudhi_patch_applied": True,
        "gudhi_patch_description": (
            "torch_topological 0.1.9 passes torch.Tensor to gudhi.CubicalComplex; "
            "gudhi>=3.10 requires numpy array. Patched venv source: "
            "top_dimensional_cells=x.detach().cpu().numpy().flatten()"
        ),
        "tests": results,
        "n_passed": n_pass,
        "n_total": n_total,
        "verdict": verdict,
    }
    out_json = REPO_ROOT / "docs" / "candidateD_pd_gradient_smoke_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults JSON: {out_json}")


if __name__ == "__main__":
    main()
