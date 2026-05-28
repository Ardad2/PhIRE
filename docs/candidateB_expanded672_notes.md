# Candidate B-expanded-672 Notes

**Updated:** 2026-05-23

Candidate B-expanded-672 = Candidate B physics/level-set losses trained on
the 672-sample expanded seasonal dataset, evaluated on the 168-sample benchmark.

---

## Purpose

Ablation and generalisation test. Key comparisons:

| Question | Comparison |
|---|---|
| Does L_crit add value on expanded data? | candidateB_expanded672 vs candidateC_expanded672 |
| Does more data help Candidate B? | candidateB_expanded672 vs candidateB (168 training samples) |
| Does physics loss help vs UV-only? | candidateB_expanded672 vs candidateUV (if trained on 672) |

---

## Training Configuration

| Setting | Value |
|---|---|
| Training data | `example_data_topology_expanded_672/wind_MR-HR.tfrecord` |
| Training samples | 672 (winter 336–503, spring 2160–2327, summer 4344–4511, fall 6552–6719) |
| Evaluation data | `example_data_fixed/wind_MR-HR.tfrecord` (168-sample benchmark) |
| Source checkpoint | `models/wind_mr-hr/trained_cnn/cnn` |
| N_epochs | 3 |
| learning_rate | 1e-5 |
| batch_size | 4 |
| optimizer | Adam |
| mu | [0.7684, -0.4575] (pretrained; unchanged) |
| sig | [5.02455, 5.9017] (pretrained; unchanged) |

## Loss Configuration (Candidate B)

```
L_total = L_uv
        + 0.01  * L_speed
        + 0.05  * L_grad
        + 0.0   * L_wpd
        + 0.25  * L_levelset   (sigmoid 5/10/15 m/s, k=10)
        + 0.0   * L_crit       (disabled — differentiates B from C)
```

---

## Status

Script created and `py_compile` verified. Requires Spark (TF1 + expanded
TFRecord).

**Pre-conditions before training (Spark):**

1. Generate expanded dataset:
   ```bash
   python3 scripts/build_wind_mrhr_expanded_dataset.py \
     --out-dir example_data_topology_expanded_672
   ```
2. Run training:
   ```bash
   python3 scripts/run_candidateB_expanded672_finetune.py \
     2>&1 | tee logs/wind_finetune_candidateB_expanded672.log
   ```
3. Verify outputs:
   ```
   data_out/wind_finetune_candidateB_expanded672/dataSR.npy
   data_out/wind_finetune_candidateB_expanded672/dataGT.npy
   data_out/wind_finetune_candidateB_expanded672/dataIN.npy
   data_out/wind_finetune_candidateB_expanded672/idx.npy
   models_fixed/topology_finetuning/wind_finetune_candidateB_expanded672/
   ```

---

## Post-training Steps (Spark)

### 1. Evaluation

```bash
python3 scripts/evaluate_finetune_candidate.py \
  --candidate-name candidateB_expanded672 \
  --candidate-dir  data_out/wind_finetune_candidateB_expanded672 \
  --cnn-dir        data_out_fixed/wind_mrhr_cnn \
  --gan-dir        data_out_fixed/wind_mrhr_gan \
  --merged-csv     ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \
  --out-dir        ttk_runs_fixed/topology_finetuning/candidateB_expanded672_eval
```

### 2. TTK topology pipeline

```bash
bash scripts/run_candidate_topology_pipeline.sh \
  --method   candidateB_expanded672 \
  --data-dir data_out/wind_finetune_candidateB_expanded672
```

---

## Abort Behavior

| Scenario | Exit |
|---|---|
| Expanded TFRecord missing | 1 (clear error + build command) |
| Benchmark TFRecord missing | 1 (clear error) |
| Checkpoint missing | 1 (clear error) |
| Output overlaps protected path | 1 (safety check) |
