# Candidate D-expanded-672 Notes

**Updated:** 2026-05-27

Candidate D-expanded-672 = Candidate D direct differentiable PD-loss residual
refiner trained on the 672-sample expanded seasonal dataset, evaluated on the
168-sample benchmark.

---

## Purpose

CandidateD-expanded-672 tests whether the direct differentiable PD-loss
approach generalises when trained on the same non-overlapping 672-sample
seasonal dataset used by CandidateUV/B/C-expanded.  This provides a fair
expanded-data comparison across the four approaches:

| Candidate | Loss objective | Training data |
|---|---|---|
| UV-expanded-672  | L_uv only | 672 seasonal samples |
| B-expanded-672   | L_uv + L_speed + L_grad + L_levelset | 672 seasonal samples |
| C-expanded-672   | B + L_crit (extrema proxy) | 672 seasonal samples |
| **D-expanded-672** | **L_uv + L_speed + L_grad + L_crit + L_PD (Wasserstein)** | **672 seasonal samples** |

### Key comparisons

| Question | Comparison |
|---|---|
| Does L_PD add value over UV-only on expanded data? | D-expanded vs UV-expanded |
| Does L_PD improve over soft level-set (L_levelset)? | D-expanded vs B-expanded |
| Does L_PD improve over extrema proxy (L_crit)? | D-expanded vs C-expanded |
| Does expanded data help Candidate D? | D-expanded vs D pilot (168 samples) |

---

## Architecture

CandidateD uses a **residual refiner** (`RefinerNet`) that post-processes the
frozen pretrained PhIRE CNN SR output:

```
SR_D = SR_CNN + residual_scale * body(SR_CNN)      (residual_scale = 0.1)
```

`body` is a 4-layer Conv-ReLU CNN (32 hidden channels, 3×3 kernels).
Its last conv layer is **zero-initialized**, so `SR_D == SR_CNN` at t=0.

The pretrained PhIRE CNN is never fine-tuned; only the RefinerNet parameters
are trained.  This is fundamentally different from Candidates UV/B/C, which
fine-tune the PhIRE CNN directly via TF1.

**RefinerNet: 14,786 parameters** (from pilot diagnostic).

---

## Loss Configuration (Candidate D pilot — unchanged)

```
L_total = L_uv
        + 0.01   * L_speed        (speed MSE)
        + 0.05   * L_grad         (gradient-magnitude MSE)
        + 0.001  * L_crit         (local-speed-maxima proxy; pool=3, z=1.0)
        + 0.0    * L_PD           (PD Wasserstein-2; lambda_pd=0 → monitoring only)
```

`lambda_pd=0` matches the Candidate D pilot (which ran in diagnostic-only mode).
PD gradients are verified at startup, so `lambda_pd` can be raised if needed.
Recommended values from the pilot diagnostic (synthetic data):

| Target | lambda_pd |
|---|---|
| 10% of L_uv | 0.001904 |
| 25% of L_uv | 0.004759 |
| 50% of L_uv | 0.009518 |

`L_PD` is the Wasserstein-2 distance between superlevel-set persistence
diagrams of scalar wind speed, computed via `torch_topological.CubicalComplex`
on a 100×100 spatial crop normalized to the GT speed range.

---

## Hyperparameters (Candidate D pilot — unchanged)

| Setting | Value |
|---|---|
| lr | 1e-4 (Adam; different from TF1 candidates which use 1e-5) |
| epochs | 3 |
| residual_scale | 0.1 |
| pd_crop_size | 100 |
| seed | 42 |

Note: Candidate D uses `lr=1e-4` (not 1e-5) because it trains a small PyTorch
residual network from scratch, not a large pretrained TF1 CNN.

---

## Two-Phase Workflow

### Phase 1: Training on expanded 672-sample CNN SR outputs

**Input**: `data_out/wind_mrhr_cnn_expanded672/`
- `dataSR.npy` (672, 500, 500, 2) — pretrained CNN SR on expanded 672 samples
- `dataGT.npy` (672, 500, 500, 2) — HR ground truth for expanded 672 samples

The RefinerNet is trained to minimize `L_total` on the 672-sample expanded
dataset CNN SR outputs, learning a residual correction.

**Output**: `models_fixed/topology_finetuning/wind_finetune_candidateD_expanded672/`
- `refiner_epoch01.pt`, `refiner_epoch02.pt`, `refiner_epoch03.pt`
- `refiner_final.pt`

### Phase 2: Evaluation on 168-sample benchmark

**Input**: `data_out_fixed/wind_mrhr_cnn/`
- `dataSR.npy` (168, 500, 500, 2) — pretrained CNN SR on 168-sample benchmark
- `dataGT.npy` (168, 500, 500, 2) — HR ground truth for 168-sample benchmark

The trained RefinerNet is applied to the 168-sample benchmark CNN SR outputs.
This produces refined SR that can be compared directly to UV/B/C-expanded-672.

**Output**: `data_out/wind_finetune_candidateD_expanded672/`
- `dataSR.npy` (168, 500, 500, 2)
- `dataGT.npy` (168, 500, 500, 2)
- `dataIN.npy` (168, 100, 100, 2) — MR inputs (copied from benchmark CNN dir)
- `idx.npy` (168,)

---

## Pre-conditions (Spark)

### Step 0: Generate expanded 672-sample CNN SR training arrays

This step requires TF1 PhIREGANs (Spark). Only needed once.

```bash
# Option A: inline Python
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
```

Verify:
```bash
python3 -c "
import numpy as np
for f in ['dataSR','dataGT','dataIN','idx']:
    a = np.load(f'data_out/wind_mrhr_cnn_expanded672/{f}.npy', mmap_mode='r')
    print(f, a.shape, a.dtype)
"
# Expected:
# dataSR (672, 500, 500, 2) float32
# dataGT (672, 500, 500, 2) float32
# dataIN (672, 100, 100, 2) float32
# idx    (672,)             int64
```

### Step 1: Also ensure the 672-sample TFRecord exists

```bash
python3 scripts/build_wind_mrhr_expanded_dataset.py \
  --out-dir example_data_topology_expanded_672
```

### Step 2: Run Candidate D-expanded-672 training

```bash
micromamba run -p /home/adadhwal/PhIRE/.mamba_candidateD_pd \
    python scripts/run_candidateD_expanded672_pd_refiner.py
# or:
.venv_candidateD_pd/bin/python scripts/run_candidateD_expanded672_pd_refiner.py
```

Log is written automatically to:
`logs/wind_finetune_candidateD_expanded672.log`

Do NOT also use external `tee` — the script tees internally.

### Step 3: Verify outputs

```bash
python3 - <<'PY'
from pathlib import Path
import numpy as np

d = Path("data_out/wind_finetune_candidateD_expanded672")
for name in ["idx.npy", "dataIN.npy", "dataGT.npy", "dataSR.npy"]:
    p = d / name
    print(name, "exists=", p.exists())
    if p.exists():
        a = np.load(p, mmap_mode="r")
        print("  shape:", a.shape, "dtype:", a.dtype)
        print("  min/max:", float(np.nanmin(a)), float(np.nanmax(a)))
PY
```

Expected shapes:
```
idx.npy    (168,)
dataIN.npy (168, 100, 100, 2)
dataGT.npy (168, 500, 500, 2)
dataSR.npy (168, 500, 500, 2)
```

---

## Post-training Steps (Spark)

### 1. Scalar/proxy evaluation

```bash
python3 scripts/evaluate_finetune_candidate.py \
  --candidate-name candidateD_expanded672 \
  --candidate-dir  data_out/wind_finetune_candidateD_expanded672 \
  --cnn-dir        data_out_fixed/wind_mrhr_cnn \
  --gan-dir        data_out_fixed/wind_mrhr_gan \
  --merged-csv     ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \
  --out-dir        ttk_runs_fixed/topology_finetuning/candidateD_expanded672_eval
```

### 2. TTK topology pipeline

```bash
rm -rf ttk_runs_fixed/topology_finetuning/candidateD_expanded672_vti \
       ttk_runs_fixed/topology_finetuning/candidateD_expanded672_topology

bash scripts/run_candidate_topology_pipeline.sh \
  --method candidateD_expanded672 \
  --data-dir data_out/wind_finetune_candidateD_expanded672 \
  --vti-dir ttk_runs_fixed/topology_finetuning/candidateD_expanded672_vti \
  --out-base ttk_runs_fixed/topology_finetuning/candidateD_expanded672_topology \
  --n-samples 168 \
  2>&1 | tee logs/candidateD_expanded672_topology_pipeline.log
```

### 3. Final topology comparison

```bash
python3 scripts/build_candidate_topology_comparison.py \
  --candidate-name candidateD_expanded672 \
  --candidate-results ttk_runs_fixed/topology_finetuning/candidateD_expanded672_topology/phase_c_final/phase_c_results.csv \
  --baseline-results   ttk_runs_fixed/combined/phase_c_results.csv \
  --candidate-idx      data_out/wind_finetune_candidateD_expanded672/idx.npy \
  --out-dir            ttk_runs_fixed/topology_finetuning/candidateD_expanded672_topology \
  --report-path        docs/topology_finetuning_candidateD_expanded672_topology_eval.md
```

### Completion checks

```bash
find ttk_runs_fixed/topology_finetuning/candidateD_expanded672_vti -name "*.vti" | wc -l
find ttk_runs_fixed/topology_finetuning/candidateD_expanded672_topology/pd -name "*.vtu" | wc -l
find ttk_runs_fixed/topology_finetuning/candidateD_expanded672_topology/mt -name "*_port_0.vtu" | wc -l
wc -l ttk_runs_fixed/topology_finetuning/candidateD_expanded672_topology/phase_c_final/phase_c_results.csv 2>/dev/null || true
```

Expected final counts: 336 / 336 / 336 / 169 (168 rows + header)

---

## Abort Behavior

| Scenario | Exit |
|---|---|
| Expanded CNN SR arrays missing | 1 (clear error + generation command) |
| Expanded CNN SR has wrong sample count | 1 (clear error) |
| Benchmark CNN SR missing | 1 (clear error) |
| Output overlaps protected path | 1 (safety check) |
| PD gradient check fails | RuntimeError (environment issue) |
| L_PD NaN/inf during training | RuntimeError (abort early) |
| Output shape validation fails | 1 (shape mismatch report) |

---

## Environment Notes

| Environment | Command |
|---|---|
| micromamba | `micromamba run -p /home/adadhwal/PhIRE/.mamba_candidateD_pd python scripts/run_candidateD_expanded672_pd_refiner.py` |
| venv | `.venv_candidateD_pd/bin/python scripts/run_candidateD_expanded672_pd_refiner.py` |

Both `.mamba_candidateD_pd` and `.venv_candidateD_pd` exist in this repository.
Use whichever matches the Spark setup. The script auto-patches the GUDHI
compatibility issue (torch_topological 0.1.9 + gudhi >= 3.10) at startup.
