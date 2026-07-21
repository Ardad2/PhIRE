# Candidate C-expanded-672 Notes

**Updated:** 2026-05-23

Candidate C-expanded-672 = Candidate C loss configuration trained on the
672-sample expanded seasonal dataset, evaluated on the 168-sample benchmark.

---

## Purpose

Tests whether the Candidate C topology-aware objective generalises across
seasons.  Key comparisons:

| Question | Comparison |
|---|---|
| Does more data help? | candidateC_expanded672 vs candidateC (same loss, 672 vs 168 training samples) |
| Does loss matter? | candidateC_expanded672 vs candidateUV_expanded (if created; loss vs no-loss on 672) |
| Does diversity transfer? | Does spring/summer/fall diversity improve winter-benchmark performance? |

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

## Loss Configuration (Candidate C1)

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

Script created and `py_compile` verified. Training data
(`example_data_topology_expanded_672/wind_MR-HR.tfrecord`) must be generated
first on Spark. TF1 is not available on the dev machine.

**Pre-conditions before training (Spark):**

1. Generate expanded dataset:
   ```bash
   python3 scripts/build_wind_mrhr_expanded_dataset.py \
     --out-dir example_data_topology_expanded_672
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
   python3 scripts/run_candidateC_expanded672_finetune.py \
     2>&1 | tee logs/wind_finetune_candidateC_expanded672.log
   ```
5. Verify outputs:
   ```
   data_out/wind_finetune_candidateC_expanded672/dataSR.npy
   data_out/wind_finetune_candidateC_expanded672/dataGT.npy
   data_out/wind_finetune_candidateC_expanded672/dataIN.npy
   data_out/wind_finetune_candidateC_expanded672/idx.npy
   models_fixed/topology_finetuning/wind_finetune_candidateC_expanded672/
   ```

---

## Post-training Steps (Spark)

### 1. Evaluation (compare to CNN baseline and Candidate C)

```bash
python3 scripts/evaluate_finetune_candidate.py \
  --candidate-name candidateC_expanded672 \
  --candidate-dir  data_out/wind_finetune_candidateC_expanded672 \
  --cnn-dir        data_out_fixed/wind_mrhr_cnn \
  --gan-dir        data_out_fixed/wind_mrhr_gan \
  --merged-csv     ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \
  --out-dir        ttk_runs_fixed/topology_finetuning/candidateC_expanded672_eval
```

Outputs:
- `ttk_runs_fixed/topology_finetuning/candidateC_expanded672_eval/all_sample_metrics_candidateC_expanded672.csv`
- `ttk_runs_fixed/topology_finetuning/candidateC_expanded672_eval/pairwise_cnn_vs_candidateC_expanded672.csv`
- `docs/topology_finetuning_candidateC_expanded672_eval.md`

### 2. TTK topology pipeline

```bash
bash scripts/run_candidate_topology_pipeline.sh \
  --method   candidateC_expanded672 \
  --data-dir data_out/wind_finetune_candidateC_expanded672
```

---

## Key Design Decision: Inference on 168-Sample Benchmark

Phase 1 trains on the 672-sample expanded TFRecord.
Phase 2 runs `test_paired()` on `example_data_fixed/wind_MR-HR.tfrecord`
(the 168-sample corrected benchmark). This means:

- The model never trains on the benchmark evaluation samples (WTK 0..167).
- Evaluation outputs are directly comparable with Candidate C, Candidate UV,
  and the CNN/GAN baselines (all evaluated on the same 168 samples).
- The 168-sample benchmark acts as a held-out test set for generalization.

---

## Abort Behavior

| Scenario | Exit |
|---|---|
| `example_data_topology_expanded_672/wind_MR-HR.tfrecord` missing | 1 (clear error + build command) |
| `example_data_fixed/wind_MR-HR.tfrecord` missing | 1 (clear error) |
| Checkpoint file missing | 1 (clear error) |
| Output would overlap protected baseline | 1 (safety check) |
