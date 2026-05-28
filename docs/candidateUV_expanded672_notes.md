# Candidate UV-expanded-672 Notes

**Updated:** 2026-05-26

Candidate UV-expanded-672 = UV-only ablation fine-tuning trained on the
672-sample expanded seasonal dataset, evaluated on the 168-sample benchmark.

---

## Purpose

CandidateUV-expanded-672 is the expanded-data UV-only ablation control.
It trains on the same 672 seasonally diverse samples as CandidateB-expanded-672
and CandidateC-expanded-672, but with no auxiliary physics or topology losses.
This isolates the effect of expanded fine-tuning alone versus the effect of
the auxiliary losses.

### Key Comparisons

| Question | Comparison |
|---|---|
| Do physics losses add value on expanded data? | candidateUV_expanded672 vs candidateB_expanded672 |
| Does L_crit add value on expanded data? | candidateUV_expanded672 vs candidateC_expanded672 |
| Does more data help UV-only fine-tuning? | candidateUV_expanded672 vs candidateUV (168-sample pilot) |
| Does UV-only fine-tuning beat the pretrained CNN? | candidateUV_expanded672 vs CNN baseline |

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

## Loss Configuration (UV-only ablation)

```
L_total = L_uv   (reconstruction loss only)

lambda_speed    = 0.0  (disabled)
lambda_grad     = 0.0  (disabled)
lambda_wpd      = 0.0  (disabled)
lambda_levelset = 0.0  (disabled)
lambda_crit     = 0.0  (disabled)
```

Auxiliary losses are computed and logged via `diagnostic_mode=True` but
contribute zero weight to the training objective.

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
   python3 scripts/run_candidateUV_expanded672_finetune.py \
     2>&1 | tee logs/wind_finetune_candidateUV_expanded672.log
   ```
3. Verify outputs:
   ```
   data_out/wind_finetune_candidateUV_expanded672/dataSR.npy
   data_out/wind_finetune_candidateUV_expanded672/dataGT.npy
   data_out/wind_finetune_candidateUV_expanded672/dataIN.npy
   data_out/wind_finetune_candidateUV_expanded672/idx.npy
   models_fixed/topology_finetuning/wind_finetune_candidateUV_expanded672/
   ```

---

## Post-training Steps (Spark)

### 1. Scalar/proxy evaluation

```bash
python3 scripts/evaluate_finetune_candidate.py \
  --candidate-name candidateUV_expanded672 \
  --candidate-dir  data_out/wind_finetune_candidateUV_expanded672 \
  --cnn-dir        data_out_fixed/wind_mrhr_cnn \
  --gan-dir        data_out_fixed/wind_mrhr_gan \
  --merged-csv     ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \
  --out-dir        ttk_runs_fixed/topology_finetuning/candidateUV_expanded672_eval
```

### 2. TTK topology pipeline

```bash
rm -rf ttk_runs_fixed/topology_finetuning/candidateUV_expanded672_vti \
       ttk_runs_fixed/topology_finetuning/candidateUV_expanded672_topology

bash scripts/run_candidate_topology_pipeline.sh \
  --method candidateUV_expanded672 \
  --data-dir data_out/wind_finetune_candidateUV_expanded672 \
  --vti-dir ttk_runs_fixed/topology_finetuning/candidateUV_expanded672_vti \
  --out-base ttk_runs_fixed/topology_finetuning/candidateUV_expanded672_topology \
  --n-samples 168 \
  2>&1 | tee logs/candidateUV_expanded672_topology_pipeline.log
```

### 3. Final topology comparison

```bash
python3 scripts/build_candidate_topology_comparison.py \
  --candidate-name candidateUV_expanded672 \
  --candidate-results ttk_runs_fixed/topology_finetuning/candidateUV_expanded672_topology/phase_c_final/phase_c_results.csv \
  --baseline-results   ttk_runs_fixed/combined/phase_c_results.csv \
  --candidate-idx      data_out/wind_finetune_candidateUV_expanded672/idx.npy \
  --out-dir            ttk_runs_fixed/topology_finetuning/candidateUV_expanded672_topology \
  --report-path        docs/topology_finetuning_candidateUV_expanded672_topology_eval.md
```

### Completion checks

```bash
find ttk_runs_fixed/topology_finetuning/candidateUV_expanded672_vti -name "*.vti" | wc -l
find ttk_runs_fixed/topology_finetuning/candidateUV_expanded672_topology/pd -name "*.vtu" | wc -l
find ttk_runs_fixed/topology_finetuning/candidateUV_expanded672_topology/mt -name "*_port_0.vtu" | wc -l
wc -l ttk_runs_fixed/topology_finetuning/candidateUV_expanded672_topology/phase_c_final/phase_c_results.csv
```

Expected final counts: 336 / 336 / 336 / 169 (168 rows + header)

---

## Abort Behavior

| Scenario | Exit |
|---|---|
| Expanded TFRecord missing | 1 (clear error + build command) |
| Benchmark TFRecord missing | 1 (clear error) |
| Checkpoint missing | 1 (clear error) |
| Output overlaps protected path | 1 (safety check) |
