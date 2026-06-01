# Candidate UV-expanded-2688 Notes

**Updated:** 2026-05-29

Candidate UV-expanded-2688 = UV-only ablation fine-tuning trained on the
2688-sample expanded seasonal dataset, evaluated on the 168-sample benchmark.

---

## Purpose

CandidateUV-expanded-2688 is the data-volume control for CandidateC-expanded-2688.
CandidateC-expanded-2688 improved PD to 22.4944, beat CNN on PD for all 168 samples,
and recovered 10/20 original MT-GAN cases.  UV-expanded-2688 trains on the same
2688-sample dataset with the same hyperparameters but uses L_uv only, isolating
whether those gains come from Candidate C's auxiliary topology-aware proxy losses
or from larger-data fine-tuning alone.

| Question | Comparison |
|---|---|
| Do C's auxiliary losses explain the PD gains at 2688 samples? | candidateUV_expanded2688 vs candidateC_expanded2688 (same data, loss vs no-loss) |
| Does more data help UV-only? | candidateUV_expanded2688 vs candidateUV_expanded1344 (same loss, 2688 vs 1344 samples) |
| Does UV fine-tuning help at all at 2688 samples? | candidateUV_expanded2688 vs CNN baseline |

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

## Loss Configuration (UV-only ablation)

```
L_total = L_uv

lambda_speed    = 0.0   (disabled)
lambda_grad     = 0.0   (disabled)
lambda_wpd      = 0.0   (disabled)
lambda_levelset = 0.0   (disabled)
lambda_crit     = 0.0   (disabled)
```

Auxiliary losses are still computed and logged (`diagnostic_mode=True`) so
their raw magnitudes are visible in the log, but they contribute zero weight
to parameter updates.

---

## Status

Script created and `py_compile` verified. Training data
(`example_data_topology_expanded_2688/wind_MR-HR.tfrecord`) must be generated
first on Spark using `scripts/build_wind_mrhr_expanded_dataset_2688.py`.
TF1 is not available on the dev machine.

**Pre-conditions before training (Spark):**

1. Generate 2688-sample expanded dataset (if not already done for C-expanded-2688):
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

### Step 1: Training

```bash
python3 scripts/run_candidateUV_expanded2688_finetune.py
```

Log written to `logs/wind_finetune_candidateUV_expanded2688.log`.
Do NOT add external `tee` — the script tees internally.

### Step 2: Output verification

```bash
python3 - <<'PY'
from pathlib import Path
import numpy as np

d = Path("data_out/wind_finetune_candidateUV_expanded2688")
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

### Step 3: Scalar/physics/domain evaluation

```bash
python3 scripts/evaluate_finetune_candidate.py \
  --candidate-name candidateUV_expanded2688 \
  --candidate-dir  data_out/wind_finetune_candidateUV_expanded2688 \
  --cnn-dir        data_out_fixed/wind_mrhr_cnn \
  --gan-dir        data_out_fixed/wind_mrhr_gan \
  --merged-csv     ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \
  --out-dir        ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_eval
```

### Step 4: TTK topology pipeline

```bash
rm -rf ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_vti \
       ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology

bash scripts/run_candidate_topology_pipeline.sh \
  --method candidateUV_expanded2688 \
  --data-dir data_out/wind_finetune_candidateUV_expanded2688 \
  --vti-dir ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_vti \
  --out-base ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology \
  --n-samples 168 \
  2>&1 | tee logs/candidateUV_expanded2688_topology_pipeline.log
```

### Step 5: Completion checks

```bash
find ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_vti -name "*.vti" | wc -l
find ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/pd -name "*.vtu" | wc -l
find ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/mt -name "*_port_0.vtu" | wc -l
wc -l ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/phase_c_final/phase_c_results.csv 2>/dev/null || true
```

Expected: 336 / 336 / 336 / 169

### Step 6: Topology comparison report

```bash
python3 scripts/build_candidate_topology_comparison.py \
  --candidate-name candidateUV_expanded2688 \
  --candidate-results ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/phase_c_final/phase_c_results.csv \
  --baseline-results   ttk_runs_fixed/combined/phase_c_results.csv \
  --candidate-idx      data_out/wind_finetune_candidateUV_expanded2688/idx.npy \
  --out-dir            ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology \
  --report-path        docs/topology_finetuning_candidateUV_expanded2688_topology_eval.md
```

---

## Key Design Decision: Inference on 168-Sample Benchmark

Phase 1 trains on the 2688-sample expanded TFRecord.
Phase 2 runs `test_paired()` on `example_data_fixed/wind_MR-HR.tfrecord`
(the 168-sample corrected benchmark). This means:

- The model never trains on the benchmark evaluation samples (WTK 0..167).
- Evaluation outputs are directly comparable with candidateC_expanded2688,
  candidateUV_expanded1344, and the CNN/GAN baselines (all evaluated on
  the same 168 samples).
- The 168-sample benchmark acts as a held-out test set for generalization.

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
