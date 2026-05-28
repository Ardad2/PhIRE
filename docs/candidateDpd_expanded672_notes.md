# Candidate Dpd-expanded-672 Notes

**Updated:** 2026-05-27

Candidate Dpd-expanded-672 = genuine direct PD-loss residual refiner trained on
the 672-sample expanded seasonal dataset, evaluated on the 168-sample benchmark.

---

## Relationship to candidateD_expanded672

These two scripts differ in exactly one constant:

| Script | LAMBDA_PD | PD role |
|---|---|---|
| `run_candidateD_expanded672_pd_refiner.py` | `0.0` | monitoring only (pilot-faithful) |
| `run_candidateDpd_expanded672_pd_refiner.py` | `0.001904` | **genuine training loss** |

`candidateD_expanded672` faithfully reproduces the Candidate D pilot configuration
(which ran in `--diagnostic-only` mode with `lambda_pd=0`).  It confirms the
refiner architecture works on expanded data, but L_PD contributes nothing to the
gradient.

`candidateDpd_expanded672` is the genuine direct PD-loss experiment: L_PD enters
L_total with weight 0.001904, so topology-diagram matching actively shapes the
learned residual during training.

---

## Architecture

RefinerNet post-processes the frozen pretrained PhIRE CNN SR output:

```
SR_Dpd = SR_CNN + 0.1 * body(SR_CNN)      [body zero-initialized]
```

The pretrained PhIRE CNN is never modified; only RefinerNet parameters are trained.

---

## Loss Configuration

```
L_total = L_uv
        + 0.01     * L_speed        (speed MSE)
        + 0.05     * L_grad         (gradient-magnitude MSE)
        + 0.001    * L_crit         (local-speed-maxima proxy; pool=3, z=1.0)
        + 0.001904 * L_PD           (PD Wasserstein-2; genuine training)
```

`lambda_pd = 0.001904` targets 10% of L_uv contribution.
Derivation from the Candidate D pilot diagnostic (synthetic data):

```
L_PD / L_uv ≈ 52.53×
lambda_pd = 0.10 / 52.53 ≈ 0.001904
```

`L_PD` is the Wasserstein-2 distance between superlevel-set persistence diagrams
of GT-normalized scalar wind speed, computed via `torch_topological.CubicalComplex`
on a 100×100 spatial crop per sample.

---

## Hyperparameters

| Setting | Value | Note |
|---|---|---|
| lr | 1e-4 | Adam; same as Candidate D pilot |
| epochs | 3 | |
| residual_scale | 0.1 | |
| pd_crop_size | 100 | 100×100 spatial crop for L_PD |
| pd_every | 1 | L_PD computed every training step |
| seed | 42 | |

---

## Two-Phase Workflow

### Phase 1: Training on 672 expanded CNN SR outputs

**Input**: `data_out/wind_mrhr_cnn_expanded672/`
- `dataSR.npy` (672, 500, 500, 2) — pretrained CNN SR on expanded 672 samples
- `dataGT.npy` (672, 500, 500, 2) — HR ground truth

**Output**: `models_fixed/topology_finetuning/wind_finetune_candidateDpd_expanded672/`
- `refiner_epoch01.pt`, `refiner_epoch02.pt`, `refiner_epoch03.pt`
- `refiner_final.pt`

### Phase 2: Evaluation on 168-sample benchmark

**Input**: `data_out_fixed/wind_mrhr_cnn/`
**Output**: `data_out/wind_finetune_candidateDpd_expanded672/`
- `dataSR.npy` (168, 500, 500, 2)
- `dataGT.npy` (168, 500, 500, 2)
- `dataIN.npy` (168, 100, 100, 2)
- `idx.npy` (168,)

---

## Key Comparisons

| Question | Comparison |
|---|---|
| Does L_PD add value over no-PD on expanded data? | Dpd-expanded vs D-expanded |
| Does L_PD improve over UV-only on expanded data? | Dpd-expanded vs UV-expanded |
| Does L_PD beat soft level-set (L_levelset)? | Dpd-expanded vs B-expanded |
| Does L_PD beat extrema proxy (L_crit)? | Dpd-expanded vs C-expanded |
| Does expanded data improve genuine PD training? | Dpd-expanded vs D pilot (168 samples) |

---

## Spark Commands

### Step 0: Generate expanded CNN SR training arrays (once, TF1 required)

```bash
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

### Step 1: Training

```bash
micromamba run -p /home/adadhwal/PhIRE/.mamba_candidateD_pd \
    python scripts/run_candidateDpd_expanded672_pd_refiner.py
# or:
.venv_candidateD_pd/bin/python scripts/run_candidateDpd_expanded672_pd_refiner.py
```

Log written to `logs/wind_finetune_candidateDpd_expanded672.log`.
Do NOT add external `tee` — the script tees internally.

### Step 2: Verify outputs

```bash
python3 - <<'PY'
from pathlib import Path
import numpy as np

d = Path("data_out/wind_finetune_candidateDpd_expanded672")
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

### Step 3: Scalar/proxy evaluation

```bash
python3 scripts/evaluate_finetune_candidate.py \
  --candidate-name candidateDpd_expanded672 \
  --candidate-dir  data_out/wind_finetune_candidateDpd_expanded672 \
  --cnn-dir        data_out_fixed/wind_mrhr_cnn \
  --gan-dir        data_out_fixed/wind_mrhr_gan \
  --merged-csv     ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \
  --out-dir        ttk_runs_fixed/topology_finetuning/candidateDpd_expanded672_eval
```

### Step 4: TTK topology pipeline

```bash
rm -rf ttk_runs_fixed/topology_finetuning/candidateDpd_expanded672_vti \
       ttk_runs_fixed/topology_finetuning/candidateDpd_expanded672_topology

bash scripts/run_candidate_topology_pipeline.sh \
  --method candidateDpd_expanded672 \
  --data-dir data_out/wind_finetune_candidateDpd_expanded672 \
  --vti-dir ttk_runs_fixed/topology_finetuning/candidateDpd_expanded672_vti \
  --out-base ttk_runs_fixed/topology_finetuning/candidateDpd_expanded672_topology \
  --n-samples 168 \
  2>&1 | tee logs/candidateDpd_expanded672_topology_pipeline.log
```

### Step 5: Completion checks

```bash
find ttk_runs_fixed/topology_finetuning/candidateDpd_expanded672_vti -name "*.vti" | wc -l
find ttk_runs_fixed/topology_finetuning/candidateDpd_expanded672_topology/pd -name "*.vtu" | wc -l
find ttk_runs_fixed/topology_finetuning/candidateDpd_expanded672_topology/mt -name "*_port_0.vtu" | wc -l
wc -l ttk_runs_fixed/topology_finetuning/candidateDpd_expanded672_topology/phase_c_final/phase_c_results.csv 2>/dev/null || true
```

Expected: 336 / 336 / 336 / 169

### Step 6: Topology comparison report

```bash
python3 scripts/build_candidate_topology_comparison.py \
  --candidate-name candidateDpd_expanded672 \
  --candidate-results ttk_runs_fixed/topology_finetuning/candidateDpd_expanded672_topology/phase_c_final/phase_c_results.csv \
  --baseline-results   ttk_runs_fixed/combined/phase_c_results.csv \
  --candidate-idx      data_out/wind_finetune_candidateDpd_expanded672/idx.npy \
  --out-dir            ttk_runs_fixed/topology_finetuning/candidateDpd_expanded672_topology \
  --report-path        docs/topology_finetuning_candidateDpd_expanded672_topology_eval.md
```

---

## Abort Behavior

| Scenario | Exit |
|---|---|
| Expanded CNN SR arrays missing | 1 (clear error + generation command) |
| Expanded CNN SR has wrong sample count | 1 |
| Benchmark CNN SR missing | 1 |
| Output overlaps protected path | 1 (safety check) |
| PD gradient check fails at startup | RuntimeError |
| L_PD NaN/inf during training | RuntimeError (abort early) |
| L_PD produces no model gradients on first step | RuntimeError |
| Output shape validation fails | 1 (shape mismatch report) |

---

## Environment Notes

Both environments exist in this repository:

| Environment | Command |
|---|---|
| micromamba | `micromamba run -p /home/adadhwal/PhIRE/.mamba_candidateD_pd python scripts/run_candidateDpd_expanded672_pd_refiner.py` |
| venv | `.venv_candidateD_pd/bin/python scripts/run_candidateDpd_expanded672_pd_refiner.py` |

The script auto-patches the GUDHI compatibility issue at startup.
