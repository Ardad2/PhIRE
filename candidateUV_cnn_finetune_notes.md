# Candidate UV UV-only Ablation Fine-tuning Notes

**Updated:** 2026-05-22

Candidate UV = pretrained CNN fine-tuned for 3 epochs using **only L_uv**
(direct reconstruction loss). All auxiliary physics and topology losses are
disabled (lambda = 0).

---

## Purpose

Ablation control for Candidates B and C. Tests whether their improvements
over the CNN baseline are due to the auxiliary physics/topology losses or
simply to fine-tuning on the 168-sample evaluation set.

If CandidateUV ≈ CandidateC → improvements are from fine-tuning, not losses.
If CandidateUV ≪ CandidateC → auxiliary losses drive the improvement.

---

## Training Configuration

| Setting | Value |
|---|---|
| Source checkpoint | `models/wind_mr-hr/trained_cnn/cnn` |
| Data | `example_data_fixed/wind_MR-HR.tfrecord` (168 samples) |
| N_epochs | 3 |
| learning_rate | 1e-5 |
| batch_size | 4 |
| optimizer | Adam (same as B/C) |
| **lambda_speed** | **0.0 (disabled)** |
| **lambda_grad** | **0.0 (disabled)** |
| **lambda_wpd** | **0.0 (disabled)** |
| **lambda_levelset** | **0.0 (disabled)** |
| **lambda_crit** | **0.0 (disabled)** |
| diagnostic_mode | True (auxiliary loss magnitudes logged but weight = 0) |

```
L_total = L_uv
```

---

## Status

Script created and `py_compile` verified on dev machine. Training data
(`example_data_fixed/wind_MR-HR.tfrecord`) is not present in the dev
environment — run on Spark where TF1 and data are available.

**Pre-conditions before training (Spark):**

1. Ensure `example_data_fixed/wind_MR-HR.tfrecord` exists.
2. Ensure `models/wind_mr-hr/trained_cnn/cnn.{index,meta,data-00000-of-00001}` exist.
3. Run training:
   ```bash
   python3 scripts/run_candidateUV_cnn_finetune.py \
     2>&1 | tee logs/wind_finetune_pilot_candidateUV.log
   ```
4. Verify outputs:
   ```
   data_out/wind_finetune_pilot_candidateUV/dataSR.npy
   data_out/wind_finetune_pilot_candidateUV/dataGT.npy
   data_out/wind_finetune_pilot_candidateUV/dataIN.npy
   data_out/wind_finetune_pilot_candidateUV/idx.npy
   models_fixed/topology_finetuning/wind_finetune_pilot_candidateUV/
   ```

---

## Post-training Steps (Spark)

### 1. Evaluation

```bash
python3 scripts/evaluate_finetune_candidate.py \
  --candidate-name candidateUV \
  --candidate-dir  data_out/wind_finetune_pilot_candidateUV \
  --cnn-dir        data_out_fixed/wind_mrhr_cnn \
  --gan-dir        data_out_fixed/wind_mrhr_gan \
  --merged-csv     ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \
  --out-dir        ttk_runs_fixed/topology_finetuning/candidateUV_eval
```

Outputs:
- `ttk_runs_fixed/topology_finetuning/candidateUV_eval/all_sample_metrics_candidateUV.csv`
- `ttk_runs_fixed/topology_finetuning/candidateUV_eval/pairwise_cnn_vs_candidateUV.csv`
- `ttk_runs_fixed/topology_finetuning/candidateUV_eval/winner_counts_candidateUV.csv`
- `ttk_runs_fixed/topology_finetuning/candidateUV_eval/adjacent_cluster_table_candidateUV.csv`
- `docs/topology_finetuning_candidateUV_eval.md`

### 2. TTK topology pipeline

```bash
bash scripts/run_candidate_topology_pipeline.sh \
  --method   candidateUV \
  --data-dir data_out/wind_finetune_pilot_candidateUV
```

Outputs in `ttk_runs_fixed/topology_finetuning/candidateUV_topology/`.

---

## Interpretation Key

| Outcome | Conclusion |
|---|---|
| candidateUV PSNR ≈ candidateC PSNR | B/C improvement is from fine-tuning, not losses |
| candidateUV PSNR ≪ candidateC PSNR | Auxiliary losses drive the B/C improvement |
| candidateUV PD/MT ≈ candidateC PD/MT | Topology loss (L_crit) adds no topology benefit |
| candidateUV PD/MT ≫ candidateC PD/MT | L_crit genuinely improves topology preservation |

---

## Abort Behavior Verified

| Scenario | Exit |
|---|---|
| TFRecord missing | 1 (clear error: `[error] TFRecord not found: ...`) |
| Checkpoint missing | 1 (clear error: `[error] Checkpoint not found: ...`) |
| Output would overlap protected baseline | 1 (safety check) |
