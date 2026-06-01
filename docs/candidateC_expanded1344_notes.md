# Candidate C-expanded-1344 Notes

**Updated:** 2026-05-29

Candidate C-expanded-1344 = Candidate C loss configuration trained on the
1344-sample expanded seasonal dataset, evaluated on the 168-sample benchmark.

---

## Purpose

Tests whether doubling the training data from 672 to 1344 samples (two
168-hour windows per season instead of one) yields further gains over
candidateC_expanded672, with all other hyperparameters and loss weights held
constant.

| Question | Comparison |
|---|---|
| Does more intra-season data help? | candidateC_expanded1344 vs candidateC_expanded672 (same loss; 1344 vs 672 samples) |
| Does more data help at all? | candidateC_expanded1344 vs candidateC (same loss; 1344 vs 168 training samples) |
| Is topology loss worth it at 1344? | candidateC_expanded1344 vs UV-expanded (if created; loss vs no-loss on 1344) |

---

## Training Configuration

| Setting | Value |
|---|---|
| Training data | `example_data_topology_expanded_1344/wind_MR-HR.tfrecord` |
| Training samples | 1344 (8 windows × 168 hours) |
| Evaluation data | `example_data_fixed/wind_MR-HR.tfrecord` (168-sample benchmark) |
| Source checkpoint | `models/wind_mr-hr/trained_cnn/cnn` |
| N_epochs | 3 |
| learning_rate | 1e-5 |
| batch_size | 4 |
| optimizer | Adam |
| mu | [0.7684, -0.4575] (pretrained; unchanged) |
| sig | [5.02455, 5.9017] (pretrained; unchanged) |

### Seasonal windows (1344 samples)

| Window | WTK start | WTK end | Notes |
|--------|-----------|---------|-------|
| winter_1 | 336 | 503 | same as candidateC_expanded672 |
| winter_2 | 504 | 671 | new — immediately follows winter_1 |
| spring_1 | 2160 | 2327 | same as candidateC_expanded672 |
| spring_2 | 2328 | 2495 | new — immediately follows spring_1 |
| summer_1 | 4344 | 4511 | same as candidateC_expanded672 |
| summer_2 | 4512 | 4679 | new — immediately follows summer_1 |
| fall_1 | 6552 | 6719 | same as candidateC_expanded672 |
| fall_2 | 6720 | 6887 | new — immediately follows fall_1 |

All windows are non-overlapping with the benchmark (WTK 0..167).

## Loss Configuration (Candidate C1 — identical to candidateC_expanded672)

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
(`example_data_topology_expanded_1344/wind_MR-HR.tfrecord`) must be generated
first on Spark. TF1 is not available on the dev machine.

**Pre-conditions before training (Spark):**

1. Generate 1344-sample expanded dataset:
   ```bash
   python3 scripts/build_wind_mrhr_expanded_dataset_1344.py \
     --out-dir example_data_topology_expanded_1344
   ```
2. Verify checkpoint:
   ```bash
   ls models/wind_mr-hr/trained_cnn/cnn.{index,meta,data-00000-of-00001}
   ```
3. Verify benchmark data:
   ```bash
   ls example_data_fixed/wind_MR-HR.tfrecord
   ```
4. Run training:
   ```bash
   python3 scripts/run_candidateC_expanded1344_finetune.py \
     2>&1 | tee logs/wind_finetune_candidateC_expanded1344.log
   ```
5. Verify outputs:
   ```
   data_out/wind_finetune_candidateC_expanded1344/dataSR.npy
   data_out/wind_finetune_candidateC_expanded1344/dataGT.npy
   data_out/wind_finetune_candidateC_expanded1344/dataIN.npy
   data_out/wind_finetune_candidateC_expanded1344/idx.npy
   models_fixed/topology_finetuning/wind_finetune_candidateC_expanded1344/
   ```

---

## Spark Commands

### Step 0: Dry run (verify HSDS connection and crop)

```bash
python3 scripts/build_wind_mrhr_expanded_dataset_1344.py \
  --out-dir example_data_topology_expanded_1344 --dry-run
```

### Step 1: Build 1344-sample dataset

```bash
python3 scripts/build_wind_mrhr_expanded_dataset_1344.py \
  --out-dir example_data_topology_expanded_1344
```

Expected output files:
```
example_data_topology_expanded_1344/wind_MR-HR.tfrecord
example_data_topology_expanded_1344/wind_LR-MR.tfrecord
example_data_topology_expanded_1344/hr_stack.npy    (1344, 500, 500, 2)
example_data_topology_expanded_1344/mr_stack.npy    (1344, 100, 100, 2)
example_data_topology_expanded_1344/lr_stack.npy    (1344,  10,  10, 2)
example_data_topology_expanded_1344/manifest.csv
example_data_topology_expanded_1344/stats.json
```

### Step 2: Train CandidateC-expanded-1344

```bash
python3 scripts/run_candidateC_expanded1344_finetune.py
```

Log written to `logs/wind_finetune_candidateC_expanded1344.log`.
Do NOT add external `tee` — the script tees internally.

### Step 3: Verify outputs

```bash
python3 - <<'PY'
from pathlib import Path
import numpy as np

d = Path("data_out/wind_finetune_candidateC_expanded1344")
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
  --candidate-name candidateC_expanded1344 \
  --candidate-dir  data_out/wind_finetune_candidateC_expanded1344 \
  --cnn-dir        data_out_fixed/wind_mrhr_cnn \
  --gan-dir        data_out_fixed/wind_mrhr_gan \
  --merged-csv     ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \
  --out-dir        ttk_runs_fixed/topology_finetuning/candidateC_expanded1344_eval
```

### Step 5: TTK topology pipeline

```bash
rm -rf ttk_runs_fixed/topology_finetuning/candidateC_expanded1344_vti \
       ttk_runs_fixed/topology_finetuning/candidateC_expanded1344_topology

bash scripts/run_candidate_topology_pipeline.sh \
  --method candidateC_expanded1344 \
  --data-dir data_out/wind_finetune_candidateC_expanded1344 \
  --vti-dir ttk_runs_fixed/topology_finetuning/candidateC_expanded1344_vti \
  --out-base ttk_runs_fixed/topology_finetuning/candidateC_expanded1344_topology \
  --n-samples 168 \
  2>&1 | tee logs/candidateC_expanded1344_topology_pipeline.log
```

### Step 6: Completion checks

```bash
find ttk_runs_fixed/topology_finetuning/candidateC_expanded1344_vti -name "*.vti" | wc -l
find ttk_runs_fixed/topology_finetuning/candidateC_expanded1344_topology/pd -name "*.vtu" | wc -l
find ttk_runs_fixed/topology_finetuning/candidateC_expanded1344_topology/mt -name "*_port_0.vtu" | wc -l
wc -l ttk_runs_fixed/topology_finetuning/candidateC_expanded1344_topology/phase_c_final/phase_c_results.csv 2>/dev/null || true
```

Expected: 336 / 336 / 336 / 169

### Step 7: Topology comparison report

```bash
python3 scripts/build_candidate_topology_comparison.py \
  --candidate-name candidateC_expanded1344 \
  --candidate-results ttk_runs_fixed/topology_finetuning/candidateC_expanded1344_topology/phase_c_final/phase_c_results.csv \
  --baseline-results   ttk_runs_fixed/combined/phase_c_results.csv \
  --candidate-idx      data_out/wind_finetune_candidateC_expanded1344/idx.npy \
  --out-dir            ttk_runs_fixed/topology_finetuning/candidateC_expanded1344_topology \
  --report-path        docs/topology_finetuning_candidateC_expanded1344_topology_eval.md
```

---

## Post-training Steps (Spark)

### Evaluation (compare to CNN baseline and candidateC_expanded672)

```bash
python3 scripts/evaluate_finetune_candidate.py \
  --candidate-name candidateC_expanded1344 \
  --candidate-dir  data_out/wind_finetune_candidateC_expanded1344 \
  --cnn-dir        data_out_fixed/wind_mrhr_cnn \
  --gan-dir        data_out_fixed/wind_mrhr_gan \
  --merged-csv     ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \
  --out-dir        ttk_runs_fixed/topology_finetuning/candidateC_expanded1344_eval
```

Outputs:
- `ttk_runs_fixed/topology_finetuning/candidateC_expanded1344_eval/all_sample_metrics_candidateC_expanded1344.csv`
- `ttk_runs_fixed/topology_finetuning/candidateC_expanded1344_eval/pairwise_cnn_vs_candidateC_expanded1344.csv`
- `docs/topology_finetuning_candidateC_expanded1344_eval.md`

### TTK topology pipeline

```bash
bash scripts/run_candidate_topology_pipeline.sh \
  --method   candidateC_expanded1344 \
  --data-dir data_out/wind_finetune_candidateC_expanded1344
```

---

## Key Design Decision: Inference on 168-Sample Benchmark

Phase 1 trains on the 1344-sample expanded TFRecord.
Phase 2 runs `test_paired()` on `example_data_fixed/wind_MR-HR.tfrecord`
(the 168-sample corrected benchmark). This means:

- The model never trains on the benchmark evaluation samples (WTK 0..167).
- Evaluation outputs are directly comparable with candidateC_expanded672,
  Candidate C, and the CNN/GAN baselines (all evaluated on the same 168
  samples).
- The 168-sample benchmark acts as a held-out test set for generalization.

---

## Abort Behavior

| Scenario | Exit |
|---|---|
| `example_data_topology_expanded_1344/wind_MR-HR.tfrecord` missing | 1 (clear error + build command) |
| `example_data_fixed/wind_MR-HR.tfrecord` missing | 1 (clear error) |
| Checkpoint file missing | 1 (clear error) |
| Output would overlap protected baseline | 1 (safety check) |
