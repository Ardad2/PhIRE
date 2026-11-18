# Phase 8 — TTK diagram extraction and W₂,₂ matching timing pilot

**Prepared 2026-11-16. Status: locally syntax/smoke-tested; NOT run on DGX Spark.**

## Purpose

Benchmark one GT and one predicted diagram computed from the **same 2,688-field PhIRE training source**, not the held-out 168-field January 2007 evaluation set. We need to know the actual time required for (a) TTK diagram extraction on the 160 × 160 speed crop and (b) H0/H1 Wasserstein matching under the project’s corrected numerical conventions before integrating the process into the native TensorFlow training loop.

This workpack does **not** update network weights, reconstruct outputs, run the 168-field benchmark, or change the published Candidate F source. It does **not** establish that a matching-derived loss differentiates through PhIRE yet.

## Source and numerical conventions

- Inputs (read-only, lazy `.npy` mmap): `data_out/wind_mrhr_cnn_expanded2688/dataGT.npy` and `dataSR.npy`. Both must have shape `(2688,500,500,2)`, containing already-denormalized physical `[u,v]`.
- Crop: `speed = sqrt(u²+v²)`, `speed[:160,:160]` in m/s, float32; VTK point ID `x + y*160` and row-major/C-order; independent XML numerical round-trip check.
- TTK: `ttkPersistenceDiagramCmd -t 1 -i <vti> -a wind_speed -o <prefix>`, using the existing `phire-ttk:latest` Docker image by default, or a native executable by request. Container startup overhead is included in the Docker timing.
- Diagram parsing: unmodified `phase6_lxxvii_update/scripts/phase6_pilot_pd_gudhi_gate.py` with the canonical `~/phire_runtime_audit_20260809_221548/recompute_pd/canonical_pd_pilot.py` decoder.
- Distance: all finite D0 and D1 birth/death pairs, `gudhi.wasserstein.wasserstein_distance(order=2, internal_p=2, matching=True, keep_essential_parts=False)` separately per dimension, combined by `hypot(D0,D1)`. Remove **exactly** zero-persistence points *only for constructing the explicit matchings*, at zero diagonal cost, then cross-check with unfiltered `pilot.distance()` W22. **No** 64-pair top-K selection or 1%-of-maximum persistence pruning is applied in this benchmark.
- Assignment cost: real-real `(b-b*)²+(d-d*)²`; unmatched point to Euclidean diagonal `(d-b)²/2`. Explicitly sum all matching costs and ensure the square root reproduces GUDHI W22. Count GT-to-diagonal, SR-to-diagonal, and real-real assignments.

## On Spark: safe sequence

Copy `phase8_ttk_w22_timing_pilot_20261116.py` into the repository root `~/PhIRE/`. Do not replace the older candidate runners.

```bash
cd ~/PhIRE

# Check that the installed GUDHI environment can verify Euclidean diagonal costs
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase8_ttk_w22_timing_pilot_20261116.py --mode selftest

# Check source arrays, scripts, Docker image and environment, with no output writes
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase8_ttk_w22_timing_pilot_20261116.py --mode preflight \
  --ttk-backend docker --sample-index 0

# Extract two diagrams and measure two H0/H1 matchings from one TRAINING sample
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase8_ttk_w22_timing_pilot_20261116.py --mode benchmark \
  --ttk-backend docker --sample-index 0 \
  --ttk-timeout 180 --match-timeout 300

# Independently hash-verify all six output files and source versions
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase8_ttk_w22_timing_pilot_20261116.py --mode verify

# View the compact numerical/timing results (avoid a pager)
python3 - <<'PY'
import json
from pathlib import Path
f=Path('ttk_runs_fixed/topology_finetuning/phase8_timing_s000_20261116/benchmark.json')
a=json.loads(f.read_text())
print('TTK GT / SR seconds:', a['ttk']['training_gt']['seconds'], a['ttk']['training_sr']['seconds'])
print('Matching seconds:', a['matching']['matching_seconds_sum'])
print('W22:', a['matching']['matching_w22'])
for k,v in a['matching']['dimensions'].items():
    print(k, v)
PY
```

The matching worker has a conservative default safety cap of **5,000 positive points across GT+SR per homology dimension**, to avoid an unbounded dense optimal-transport allocation. Exceeding it is a **failed feasibility gate**, not permission to silently prune diagrams. Reconsider the exact-matching approach and computational budget before increasing `--max-matching-points`.

If your host already has a working `ttkPersistenceDiagramCmd`, you may replace `--ttk-backend docker` with `--ttk-backend host`; document which one you use. If the training prediction arrays are missing, stop and inventory existing expanded-2688 arrays; **do not switch silently to the 168-field evaluation set**. If the output directory already exists, the script refuses overwrite; verify it or explicitly select a new `--out-dir`. Failures leave original training and evaluation outputs untouched.

When the benchmark succeeds, the isolated output directory will contain two VTIs, two port-0 VTUs, `matching.json`, `benchmark.json`, and `SHA256SUMS`. Report CPU/container TTK time separately from matching time; do not extrapolate the two cold Docker invocations to steady-state training without a persistent-container measurement.

## Audit entry to add after the run, not before

Use a *new* uniquely numbered Part or dated addendum after existing **Part CXII**. Record: script SHA-256, matching source/decoder SHA-256, `git rev-parse HEAD`, exact sample index and sample-data hashes, Spark environment versions, TTK backend, TTK GT/SR wall times, H0/H1 pair counts, matching wall times, manual matching-cost regression, source and output manifest hashes, and whether `--mode verify` completed successfully. Clearly separate numerical matching success from any subsequent gradient/training success.

The next planned, distinct gate is a synthetic **field-level** test with displaced and spurious peaks. After validating both extraction/matching and its gradient assignment, build an additive TensorFlow Candidate F derivative with separate experiment names, never overwrite F1 or the Phase 6 controls.
