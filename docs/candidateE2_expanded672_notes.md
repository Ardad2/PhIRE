# Candidate E2-expanded-672 Notes

**Updated:** 2026-05-28

Candidate E2-expanded-672 = TTK-guided critical-pair loss residual refiner trained on
the 672-sample expanded seasonal dataset, evaluated on the 168-sample benchmark.

---

## Purpose

CandidateE2-expanded-672 tests whether explicit TTK critical-vertex and persistence-gap
supervision generalises on the 672-sample expanded dataset.  It extends the Candidate E
pilot (168 samples, argparse) with:

- Hardcoded configuration for direct comparison with D/Dpd/UV/B/C-expanded-672
- "Corrected" constraint targets: birth_val/death_val read from GT numpy arrays
  (exact match guaranteed), not from TTK VTU files
- Constraints generated without TTK using scipy local maxima on GT speed crops

| Candidate | Key losses | Training data |
|---|---|---|
| UV-expanded-672   | L_uv only | 672 seasonal samples |
| B-expanded-672    | + L_speed + L_grad + L_levelset | 672 seasonal samples |
| C-expanded-672    | B + L_crit | 672 seasonal samples |
| D-expanded-672    | C + L_PD (λ=0; monitoring) | 672 seasonal samples |
| Dpd-expanded-672  | C + L_PD (λ=0.001904; genuine) | 672 seasonal samples |
| **E2-expanded-672** | **C + L_levelset + L_ttkcv + L_ttkpers** | **672 seasonal samples** |

---

## Architecture

RefinerNet post-processes the frozen pretrained PhIRE CNN SR output:

```
SR_E2 = SR_CNN + 0.1 * body(SR_CNN)      [body zero-initialized]
```

The pretrained PhIRE CNN is never modified; only RefinerNet parameters are trained.

---

## Loss Configuration

```
L_total = L_uv
        + 0.01  * L_speed        (speed MSE)
        + 0.05  * L_grad         (gradient-magnitude MSE)
        + 0.001 * L_crit         (local-speed-maxima proxy; pool=3, z=1.0)
        + 0.25  * L_levelset     (soft level-set at 5/10/15 m/s)
        + 0.04  * L_ttkcv        (MSE at TTK critical vertices)
        + 0.02  * L_ttkpers      (persistence-gap MSE)
```

### L_ttkcv

MSE between SR speed at birth/death vertex locations and GT target values:

```
L_ttkcv = 0.5 * (MSE(sr_speed[birth_yx], birth_val) + MSE(sr_speed[death_yx], death_val))
```

### L_ttkpers

MSE between SR persistence and GT persistence:

```
sr_pers = |sr_speed[death_yx] - sr_speed[birth_yx]|
L_ttkpers = MSE(sr_pers, gt_persistence)
```

Persistence is stored unsigned (|birth_val - death_val|), direction-agnostic.

### Constraint generation (no TTK required)

`build_candidateE2_expanded672_constraints.py` generates constraints from GT numpy arrays
using scipy/numpy local extrema detection:

- Birth vertices = local maxima of GT speed in 160×160 crop
  (scipy `maximum_filter` with kernel 5×5)
- Death vertex = global minimum of 160×160 crop (same for all pairs per sample)
- Persistence = birth_val − death_val
- Filter: persistence ≥ 0.01 × speed_range
- Take top-64 pairs by persistence (descending)
- Sanity check: stored birth_val/death_val verified against GT at each vertex

---

## Hyperparameters

| Setting | Value | Note |
|---|---|---|
| lr | 1e-4 | Adam |
| epochs | 3 | |
| residual_scale | 0.1 | |
| patch | 160 | vertex-ID encoding width for TTKConstraints |
| top_k | 64 | max pairs per sample (from constraint builder) |
| seed | 42 | |

---

## Two-Phase Workflow

### Phase 1: Training on 672 expanded CNN SR outputs

**Input**: `data_out/wind_mrhr_cnn_expanded672/`
- `dataSR.npy` (672, 500, 500, 2)
- `dataGT.npy` (672, 500, 500, 2)

**Constraints**: `ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_constraints/ttk_pd_critical_pairs_gtvalues.npz`

**Output**: `models_fixed/topology_finetuning/wind_finetune_candidateE2_expanded672/`
- `refiner_epoch01.pt`, `refiner_epoch02.pt`, `refiner_epoch03.pt`
- `refiner_final.pt`

### Phase 2: Evaluation on 168-sample benchmark

**Input**: `data_out_fixed/wind_mrhr_cnn/`

**Output**: `data_out/wind_finetune_candidateE2_expanded672/`
- `dataSR.npy` (168, 500, 500, 2)
- `dataGT.npy` (168, 500, 500, 2)
- `dataIN.npy` (168, 100, 100, 2)
- `idx.npy` (168,)

---

## Key Comparisons

| Question | Comparison |
|---|---|
| Does TTK-CV loss add value over no-TTK on expanded data? | E2-expanded vs D-expanded |
| Does TTK-CV improve over UV-only on expanded data? | E2-expanded vs UV-expanded |
| Does TTK-CV beat soft level-set alone? | E2-expanded vs B-expanded |
| Does TTK-CV beat extrema proxy? | E2-expanded vs C-expanded |
| Does TTK-CV beat genuine PD loss? | E2-expanded vs Dpd-expanded |

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

### Step 1: Build E2 constraints (no TTK required)

```bash
micromamba run -p /home/adadhwal/PhIRE/.mamba_candidateD_pd \
    python scripts/build_candidateE2_expanded672_constraints.py
# or:
.venv_candidateD_pd/bin/python scripts/build_candidateE2_expanded672_constraints.py
```

Verify:
```bash
python3 -c "
import numpy as np
npz = np.load('ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_constraints/ttk_pd_critical_pairs_gtvalues.npz', allow_pickle=True)
print('n_samples:', int(npz['n_samples']))
print('total pairs:', len(npz['birth_vid']))
print('sample_idx range:', [int(npz['sample_idx'].min()), int(npz['sample_idx'].max())])
"
```

### Step 2: Train CandidateE2-expanded-672

```bash
micromamba run -p /home/adadhwal/PhIRE/.mamba_candidateD_pd \
    python scripts/run_candidateE2_expanded672_ttkcrit_refiner.py
# or:
.venv_candidateD_pd/bin/python \
    scripts/run_candidateE2_expanded672_ttkcrit_refiner.py
```

Log written to `logs/wind_finetune_candidateE2_expanded672.log`.
Do NOT add external `tee` — the script tees internally.

### Step 3: Verify outputs

```bash
python3 - <<'PY'
from pathlib import Path
import numpy as np

d = Path("data_out/wind_finetune_candidateE2_expanded672")
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

### Step 4: Scalar/proxy evaluation

```bash
python3 scripts/evaluate_finetune_candidate.py \
  --candidate-name candidateE2_expanded672 \
  --candidate-dir  data_out/wind_finetune_candidateE2_expanded672 \
  --cnn-dir        data_out_fixed/wind_mrhr_cnn \
  --gan-dir        data_out_fixed/wind_mrhr_gan \
  --merged-csv     ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \
  --out-dir        ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_eval
```

### Step 5: TTK topology pipeline

```bash
rm -rf ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_vti \
       ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_topology

bash scripts/run_candidate_topology_pipeline.sh \
  --method candidateE2_expanded672 \
  --data-dir data_out/wind_finetune_candidateE2_expanded672 \
  --vti-dir ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_vti \
  --out-base ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_topology \
  --n-samples 168 \
  2>&1 | tee logs/candidateE2_expanded672_topology_pipeline.log
```

### Step 6: Completion checks

```bash
find ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_vti -name "*.vti" | wc -l
find ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_topology/pd -name "*.vtu" | wc -l
find ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_topology/mt -name "*_port_0.vtu" | wc -l
wc -l ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_topology/phase_c_final/phase_c_results.csv 2>/dev/null || true
```

Expected: 336 / 336 / 336 / 169

### Step 7: Topology comparison report

```bash
python3 scripts/build_candidate_topology_comparison.py \
  --candidate-name candidateE2_expanded672 \
  --candidate-results ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_topology/phase_c_final/phase_c_results.csv \
  --baseline-results   ttk_runs_fixed/combined/phase_c_results.csv \
  --candidate-idx      data_out/wind_finetune_candidateE2_expanded672/idx.npy \
  --out-dir            ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_topology \
  --report-path        docs/topology_finetuning_candidateE2_expanded672_topology_eval.md
```

---

## Abort Behavior

| Scenario | Exit |
|---|---|
| Expanded CNN SR arrays missing | 1 (clear error + generation command) |
| Expanded CNN SR has wrong sample count | 1 |
| Benchmark CNN SR missing | 1 |
| Constraints NPZ missing | 1 (clear error + build command) |
| Constraints NPZ has wrong sample count | 1 |
| Training indices not in constraints | 1 |
| Output overlaps protected path | 1 (safety check) |
| L_ttkcv produces no model gradients at startup | RuntimeError |
| Any loss NaN/inf during training | RuntimeError |
| Output shape validation fails | 1 |

---

## Environment Notes

| Environment | Command |
|---|---|
| micromamba | `micromamba run -p /home/adadhwal/PhIRE/.mamba_candidateD_pd python scripts/run_candidateE2_expanded672_ttkcrit_refiner.py` |
| venv | `.venv_candidateD_pd/bin/python scripts/run_candidateE2_expanded672_ttkcrit_refiner.py` |

No torch_topological or gudhi required (unlike Dpd). TTK constraint overhead:
~1–2 ms/sample vs ~100 ms for L_PD in Dpd-expanded.
