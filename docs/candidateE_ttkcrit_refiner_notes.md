# Candidate E TTK Critical-Pair Refiner Notes

**Generated:** 2026-05-20T02:33:23

## Configuration

```
  data_dir                   = data_out_fixed/wind_mrhr_cnn
  constraints                = ttk_runs_fixed/topology_finetuning/candidateE_constraints/ttk_pd_critical_pairs.npz
  out_dir                    = /home/user/PhIRE/data_out/wind_finetune_pilot_candidateE
  model_dir                  = /home/user/PhIRE/models_fixed/topology_finetuning/wind_finetune_pilot_candidateE
  log_path                   = /home/user/PhIRE/logs/wind_finetune_pilot_candidateE.log
  report_path                = /home/user/PhIRE/docs/candidateE_ttkcrit_refiner_notes.md
  epochs                     = 3
  lr                         = 0.0001
  lambda_speed               = 0.01
  lambda_grad                = 0.05
  lambda_crit                = 0.001
  lambda_ttkcv               = 0.0
  lambda_ttkpers             = 0.0
  residual_scale             = 0.1
  seed                       = 42
  diagnostic_only            = True
```

## Diagnostic Loss Breakdown *(synthetic data)*

Constraints: 64 pairs per sample

| Term | Raw value | Weighted |
|------|-----------|----------|
| L_uv      | 0.063170 | 0.063170 |
| L_speed   | 0.060805 | 0.000608 |
| L_grad    | 0.205792 | 0.010290 |
| L_crit    | 0.087321 | 0.000087 |
| L_ttkcv   | 20.722279 | 0.000000 |
| L_ttkpers | 2.911602 | 0.000000 |

L_ttkcv / L_uv   = **328.0415×**
L_ttkpers / L_uv = **46.0918×**
TTK loss time    = 1.66 ms (vs ~100 ms for L_PD at 100×100)

L_ttkcv → model params: **YES**

