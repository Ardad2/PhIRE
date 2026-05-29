# Candidate C-expanded-2688 Notes

**Updated:** 2026-05-29

Candidate C-expanded-2688 = Candidate C loss configuration trained on the
2688-sample expanded seasonal dataset, evaluated on the 168-sample benchmark.

---

## Purpose

Tests whether the Candidate C topology-aware proxy losses remain robust as the
training set doubles again from 1344 to 2688 samples (four 168-hour windows per
season instead of two).  All hyperparameters and loss weights are identical to
candidateC_expanded1344 so the comparison is controlled: the only experimental
variable is training data volume.

If CandidateC-expanded-2688 looks promising, run **CandidateUV-expanded-2688**
as the matching UV-only ablation to isolate the effect of the C auxiliary losses
vs. the effect of larger training data alone.

| Question | Comparison |
|---|---|
| Does more intra-season data help? | candidateC_expanded2688 vs candidateC_expanded1344 (same loss; 2688 vs 1344 samples) |
| Does topology loss remain valuable at 2688 samples? | candidateC_expanded2688 vs candidateUV_expanded2688 (if created) |
| Does more data help at all? | candidateC_expanded2688 vs candidateC (same loss; 2688 vs 168 training samples) |

---

## Training Configuration

| Setting | Value |
|---|---|
| Training data | `example_data_topology_expanded_2688/wind_MR-HR.tfrecord` |
| Training samples | 2688 (16 windows × 168 hours) |
| Evaluation data | `example_data_fixed/wind_MR-HR.tfrecord` (168-sample benchmark) |
| Source checkpoint | `models/wind_mr-hr/trained_cnn/cnn` |
| N_epochs | 3 |
| learning_rate | 1e-5 |
| batch_size | 4 |
| optimizer | Adam |
| mu | [0.7684, -0.4575] (pretrained; unchanged) |
| sig | [5.02455, 5.9017] (pretrained; unchanged) |

### Seasonal windows (2688 samples)

| Window | WTK start | WTK end | Notes |
|--------|-----------|---------|-------|
| winter_1 | 336 | 503 | same as candidateC_expanded672 |
| winter_2 | 504 | 671 | same as candidateC_expanded1344 |
| winter_3 | 672 | 839 | new |
| winter_4 | 840 | 1007 | new |
| spring_1 | 2160 | 2327 | same as candidateC_expanded672 |
| spring_2 | 2328 | 2495 | same as candidateC_expanded1344 |
| spring_3 | 2496 | 2663 | new |
| spring_4 | 2664 | 2831 | new |
| summer_1 | 4344 | 4511 | same as candidateC_expanded672 |
| summer_2 | 4512 | 4679 | same as candidateC_expanded1344 |
| summer_3 | 4680 | 4847 | new |
| summer_4 | 4848 | 5015 | new |
| fall_1 | 6552 | 6719 | same as candidateC_expanded672 |
| fall_2 | 6720 | 6887 | same as candidateC_expanded1344 |
| fall_3 | 6888 | 7055 | new |
| fall_4 | 7056 | 7223 | new |

All windows are non-overlapping with the benchmark (WTK 0..167).

## Loss Configuration (Candidate C1 — identical to candidateC_expanded1344)

```
L_total = L_uv
        + 0.01  * L_speed
        + 0.05  * L_grad
        + 0.0   * L_wpd
        + 0.25  * L_levelset   (sigmoid 5/10/15 m/s, k=10)
        + 0.001 * L_crit       (pool-3 maxima above mean + 1σ)
```

---

## Status

Scripts created and `py_compile` verified. Training data
(`example_data_topology_expanded_2688/wind_MR-HR.tfrecord`) must be generated
first on Spark. TF1 is not available on the dev machine.

**Pre-conditions before training (Spark):**

1. Generate 2688-sample expanded dataset:
   ```bash
   python3 scripts/build_wind_mrhr_expanded_dataset_2688.py \
     --out-dir example_data_topology_expanded_2688
   ```
2. Verify checkpoint:
   ```bash
   ls models/wind_mr-hr/trained_cnn/cnn.{index,meta,data-00000-of-00001}
   ```
3. Verify benchmark data:
   ```bash
   ls example_data_fixed/wind_MR-HR.tfrecord
   ```

---

## Spark Commands

### Step 1: Dry run (verify HSDS connection and crop)

```bash
python3 scripts/build_wind_mrhr_expanded_dataset_2688.py \
  --out-dir example_data_topology_expanded_2688 --dry-run
```

### Step 2: Build 2688-sample dataset

```bash
python3 scripts/build_wind_mrhr_expanded_dataset_2688.py \
  --out-dir example_data_topology_expanded_2688
```

Expected output files:
```
example_data_topology_expanded_2688/wind_MR-HR.tfrecord
example_data_topology_expanded_2688/wind_LR-MR.tfrecord
example_data_topology_expanded_2688/hr_stack.npy    (2688, 500, 500, 2)
example_data_topology_expanded_2688/mr_stack.npy    (2688, 100, 100, 2)
example_data_topology_expanded_2688/lr_stack.npy    (2688,  10,  10, 2)
example_data_topology_expanded_2688/manifest.csv
example_data_topology_expanded_2688/stats.json
```

### Step 3: Verify dataset

```bash
python3 - <<'PY'
from pathlib import Path
import numpy as np

d = Path("example_data_topology_expanded_2688")
for name in ["hr_stack.npy", "mr_stack.npy", "lr_stack.npy"]:
    p = d / name
    if p.exists():
        a = np.load(p, mmap_mode="r")
        print(name, "shape:", a.shape, "dtype:", a.dtype)
        speed = np.sqrt(a[...,0]**2 + a[...,1]**2)
        print("  speed min/max:", float(speed.min()), float(speed.max()))
    else:
        print(name, "MISSING")
PY
```

Expected shapes:
```
hr_stack.npy (2688, 500, 500, 2)
mr_stack.npy (2688, 100, 100, 2)
lr_stack.npy (2688,  10,  10, 2)
```

### Step 4: Train CandidateC-expanded-2688

```bash
python3 scripts/run_candidateC_expanded2688_finetune.py
```

Log written to `logs/wind_finetune_candidateC_expanded2688.log`.
Do NOT add external `tee` — the script tees internally.

### Step 5: Verify outputs

```bash
python3 - <<'PY'
from pathlib import Path
import numpy as np

d = Path("data_out/wind_finetune_candidateC_expanded2688")
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

### Step 6: Scalar/physics/domain evaluation

```bash
python3 scripts/evaluate_finetune_candidate.py \
  --candidate-name candidateC_expanded2688 \
  --candidate-dir  data_out/wind_finetune_candidateC_expanded2688 \
  --cnn-dir        data_out_fixed/wind_mrhr_cnn \
  --gan-dir        data_out_fixed/wind_mrhr_gan \
  --merged-csv     ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \
  --out-dir        ttk_runs_fixed/topology_finetuning/candidateC_expanded2688_eval
```

### Step 7: TTK topology pipeline

```bash
rm -rf ttk_runs_fixed/topology_finetuning/candidateC_expanded2688_vti \
       ttk_runs_fixed/topology_finetuning/candidateC_expanded2688_topology

bash scripts/run_candidate_topology_pipeline.sh \
  --method candidateC_expanded2688 \
  --data-dir data_out/wind_finetune_candidateC_expanded2688 \
  --vti-dir ttk_runs_fixed/topology_finetuning/candidateC_expanded2688_vti \
  --out-base ttk_runs_fixed/topology_finetuning/candidateC_expanded2688_topology \
  --n-samples 168 \
  2>&1 | tee logs/candidateC_expanded2688_topology_pipeline.log
```

### Step 8: Completion checks

```bash
find ttk_runs_fixed/topology_finetuning/candidateC_expanded2688_vti -name "*.vti" | wc -l
find ttk_runs_fixed/topology_finetuning/candidateC_expanded2688_topology/pd -name "*.vtu" | wc -l
find ttk_runs_fixed/topology_finetuning/candidateC_expanded2688_topology/mt -name "*_port_0.vtu" | wc -l
wc -l ttk_runs_fixed/topology_finetuning/candidateC_expanded2688_topology/phase_c_final/phase_c_results.csv 2>/dev/null || true
```

Expected: 336 / 336 / 336 / 169

### Step 9: Topology comparison report

```bash
python3 scripts/build_candidate_topology_comparison.py \
  --candidate-name candidateC_expanded2688 \
  --candidate-results ttk_runs_fixed/topology_finetuning/candidateC_expanded2688_topology/phase_c_final/phase_c_results.csv \
  --baseline-results   ttk_runs_fixed/combined/phase_c_results.csv \
  --candidate-idx      data_out/wind_finetune_candidateC_expanded2688/idx.npy \
  --out-dir            ttk_runs_fixed/topology_finetuning/candidateC_expanded2688_topology \
  --report-path        docs/topology_finetuning_candidateC_expanded2688_topology_eval.md
```

---

## Key Design Decision: Inference on 168-Sample Benchmark

Phase 1 trains on the 2688-sample expanded TFRecord.
Phase 2 runs `test_paired()` on `example_data_fixed/wind_MR-HR.tfrecord`
(the 168-sample corrected benchmark). This means:

- The model never trains on the benchmark evaluation samples (WTK 0..167).
- Evaluation outputs are directly comparable with candidateC_expanded1344,
  candidateC_expanded672, and the CNN/GAN baselines (all evaluated on the
  same 168 samples).
- The 168-sample benchmark acts as a held-out test set for generalization.

---

## Next Step: UV Ablation

If CandidateC-expanded-2688 shows improvement over candidateC_expanded1344,
create **CandidateUV-expanded-2688** as the matching UV-only ablation.  Both
train on the same 2688-sample dataset; comparing them isolates whether the
improvement comes from the Candidate C auxiliary losses or from additional
training data alone.

---

## Abort Behavior

| Scenario | Exit |
|---|---|
| `example_data_topology_expanded_2688/wind_MR-HR.tfrecord` missing | 1 (clear error + build command) |
| `example_data_fixed/wind_MR-HR.tfrecord` missing | 1 (clear error) |
| Checkpoint file missing | 1 (clear error) |
| Output would overlap protected baseline | 1 (safety check) |
| Post-inference: output array missing | 1 |
| Post-inference: output array wrong shape | 1 |
