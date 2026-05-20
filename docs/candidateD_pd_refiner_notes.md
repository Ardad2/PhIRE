# Candidate D PD Residual Refiner Notes

**Generated:** 2026-05-18T18:29:45

## Configuration

```
  data_dir               = data_out_fixed/wind_mrhr_cnn
  out_dir                = data_out/wind_finetune_pilot_candidateD
  model_dir              = models_fixed/topology_finetuning/wind_finetune_pilot_candidateD
  log_path               = logs/wind_finetune_pilot_candidateD.log
  report_path            = docs/candidateD_pd_refiner_notes.md
  epochs                 = 3
  lr                     = 0.0001
  lambda_speed           = 0.01
  lambda_grad            = 0.05
  lambda_crit            = 0.001
  lambda_pd              = 0.2
  pd_crop_size           = 100
  pd_every               = 1
  residual_scale         = 0.1
  seed                   = 42
  diagnostic_only        = False
  dry_run                = False
```

## Diagnostic Loss Breakdown



| Term | Raw value | Weighted |
|------|-----------|----------|
| L_uv    | 1.119733 | 1.119733 |
| L_speed | 1.647115 | 0.016471 |
| L_grad  | 0.631257 | 0.031563 |
| L_crit  | 9.610543 | 0.009611 |
| L_PD    | 0.512224 | 0.102445 *(λ=0.2)* |

L_PD / L_uv = **0.4575×**  (computed in 0.023 s, crop=100²)

Recommended lambda_pd:

- 10% of L_uv → `lambda_pd = 0.218602`
- 25% of L_uv → `lambda_pd = 0.546506`
- 50% of L_uv → `lambda_pd = 1.093012`

PD gradient to model params: **YES**

## Training Loss History

| Epoch | L_uv | L_speed | L_grad | L_crit | L_PD | L_total |
|-------|------|---------|--------|--------|------|---------|
| 1 | 0.91826 | 1.26547 | 0.57514 | 5.98122 | 0.74356 | 1.11436 |
| 2 | 0.90682 | 1.24683 | 0.57876 | 6.10384 | 0.75010 | 1.10435 |
| 3 | 0.89439 | 1.22776 | 0.58632 | 6.24507 | 0.76090 | 1.09441 |

## GUDHI Compatibility

torch_topological 0.1.9 + gudhi 3.12.0 require a one-line patch:

```
# In .venv_candidateD_pd/.../torch_topological/nn/cubical_complex.py
# Old: top_dimensional_cells=x.flatten()
# New: top_dimensional_cells=x.detach().cpu().numpy().flatten()
```

This script detects and applies the patch automatically at startup.
See `docs/candidateD_pd_gradient_smoke.md` for full documentation.
