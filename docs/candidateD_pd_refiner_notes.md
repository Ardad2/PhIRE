# Candidate D PD Residual Refiner Notes

**Generated:** 2026-05-13T21:39:37

## Configuration

```
  data_dir               = /home/user/PhIRE/data_out_fixed/wind_mrhr_cnn
  out_dir                = /home/user/PhIRE/data_out/wind_finetune_pilot_candidateD
  model_dir              = /home/user/PhIRE/models_fixed/topology_finetuning/wind_finetune_pilot_candidateD
  log_path               = /home/user/PhIRE/logs/wind_finetune_pilot_candidateD.log
  report_path            = /home/user/PhIRE/docs/candidateD_pd_refiner_notes.md
  epochs                 = 3
  lr                     = 0.0001
  lambda_speed           = 0.01
  lambda_grad            = 0.05
  lambda_crit            = 0.001
  lambda_pd              = 0.0
  pd_crop_size           = 100
  pd_every               = 1
  residual_scale         = 0.1
  seed                   = 42
  diagnostic_only        = True
  dry_run                = False
```

## Diagnostic Loss Breakdown

*(synthetic data — real data not found)*

| Term | Raw value | Weighted |
|------|-----------|----------|
| L_uv    | 0.063170 | 0.063170 |
| L_speed | 0.060805 | 0.000608 |
| L_grad  | 0.205792 | 0.010290 |
| L_crit  | 0.087321 | 0.000087 |
| L_PD    | 3.318372 | 0.000000 *(λ=0.0)* |

L_PD / L_uv = **52.5311×**  (computed in 0.038 s, crop=100²)

Recommended lambda_pd:

- 10% of L_uv → `lambda_pd = 0.001904`
- 25% of L_uv → `lambda_pd = 0.004759`
- 50% of L_uv → `lambda_pd = 0.009518`

PD gradient to model params: **YES**

## GUDHI Compatibility

torch_topological 0.1.9 + gudhi 3.12.0 require a one-line patch:

```
# In .venv_candidateD_pd/.../torch_topological/nn/cubical_complex.py
# Old: top_dimensional_cells=x.flatten()
# New: top_dimensional_cells=x.detach().cpu().numpy().flatten()
```

This script detects and applies the patch automatically at startup.
See `docs/candidateD_pd_gradient_smoke.md` for full documentation.
