# Physics/topology loss calibration diagnostic

**Generated:** 2026-05-12

## Configuration

| Parameter | Value |
|---|---|
| data_path | `example_data_fixed/wind_MR-HR.tfrecord` |
| model_path | `models/wind_mr-hr/trained_cnn/cnn` |
| n_records (found / expected) | 168 / 168 |
| batch_size | 4 |
| n_batches processed | 42 |
| r | [5] |
| mu | [0.7684, -0.4575] |
| sig | [5.02455, 5.9017] |
| levelset thresholds (m/s) | [5.0, 10.0, 15.0] |
| levelset temperature k | 10.0 |
| lambda_crit | 0.001 |
| crit_high_z | 1.0 |
| crit_include_minima | False |
| crit_low_z | -1.0 |
| crit_pool | 3 |

## Loss term descriptions

| Term | Description |
|---|---|
| L_uv | Normalized [u,v] MSE — the baseline training loss; dimensionless. |
| L_speed | Physical wind speed MSE (m²/s²); requires per-channel denormalization. |
| L_grad | Physical speed-gradient magnitude MSE; backward finite differences on interior grid. |
| L_wpd | MAE on speed³ wind-power-density proxy (m³/s³); MAE used to keep magnitudes tractable. |
| L_levelset | Dimensionless soft-mask MSE at physical speed thresholds via sigmoid; topology-inspired superlevel-set proxy. |
| L_crit | MSE focused on GT local-speed maxima above an adaptive per-sample threshold; critical-value proxy for superlevel-set topology (inspired by Kissi et al.). |

## Summary statistics

| Term | Mean | Std | Min | Max |
|---|---|---|---|---|
| L_uv | 0.031759 | 0.010682 | 0.011806 | 0.059748 |
| L_speed | 1.270521 | 0.447020 | 0.468921 | 2.374957 |
| L_grad | 0.575292 | 0.209157 | 0.158201 | 1.045843 |
| L_wpd | 231.670907 | 107.115664 | 48.855553 | 485.663300 |
| L_levelset | 0.034349 | 0.008266 | 0.014784 | 0.049501 |
| L_crit | 6.064671 | 2.511113 | 1.441270 | 11.939748 |

## Ratios to L_uv

| Term | Ratio (×L_uv) |
|---|---|
| L_speed | 40.0046× |
| L_grad | 18.1141× |
| L_wpd | 7294.5599× |
| L_levelset | 1.0815× |
| L_crit | 190.9567× |

## Candidate lambda configurations

WPD is set to `lambda_wpd=0` in candidates A–C because L_wpd magnitudes are large even after switching to MAE; Candidate D includes a small WPD weight for reference.

### Candidate A (conservative)

`lambda_speed=0.005  lambda_grad=0.01  lambda_wpd=0.0  lambda_levelset=0.1  lambda_crit=0.0`

| Term | Raw mean | Lambda | Weighted contrib | % of L_uv | Regime |
|---|---|---|---|---|---|
| L_speed | 1.270521 | 0.005 | 0.006353 | 20.00% | moderate (10–50 %) |
| L_grad | 0.575292 | 0.01 | 0.005753 | 18.11% | moderate (10–50 %) |
| L_wpd | 231.670907 | 0.0 | 0.000000 | 0.00% | negligible (<1 %) |
| L_levelset | 0.034349 | 0.1 | 0.003435 | 10.82% | moderate (10–50 %) |
| L_crit | 6.064671 | 0.0 | 0.000000 | 0.00% | negligible (<1 %) |
| L_uv (baseline) | 0.031759 | — | 0.031759 | 100.00% | — |
| **total_loss** | | | **0.047300** | | |

### Candidate B (moderate)

`lambda_speed=0.01  lambda_grad=0.05  lambda_wpd=0.0  lambda_levelset=0.25  lambda_crit=0.0`

| Term | Raw mean | Lambda | Weighted contrib | % of L_uv | Regime |
|---|---|---|---|---|---|
| L_speed | 1.270521 | 0.01 | 0.012705 | 40.00% | moderate (10–50 %) |
| L_grad | 0.575292 | 0.05 | 0.028765 | 90.57% | large (50–100 %) |
| L_wpd | 231.670907 | 0.0 | 0.000000 | 0.00% | negligible (<1 %) |
| L_levelset | 0.034349 | 0.25 | 0.008587 | 27.04% | moderate (10–50 %) |
| L_crit | 6.064671 | 0.0 | 0.000000 | 0.00% | negligible (<1 %) |
| L_uv (baseline) | 0.031759 | — | 0.031759 | 100.00% | — |
| **total_loss** | | | **0.081817** | | |

### Candidate C (aggressive)

`lambda_speed=0.02  lambda_grad=0.05  lambda_wpd=0.0  lambda_levelset=0.5  lambda_crit=0.0`

| Term | Raw mean | Lambda | Weighted contrib | % of L_uv | Regime |
|---|---|---|---|---|---|
| L_speed | 1.270521 | 0.02 | 0.025410 | 80.01% | large (50–100 %) |
| L_grad | 0.575292 | 0.05 | 0.028765 | 90.57% | large (50–100 %) |
| L_wpd | 231.670907 | 0.0 | 0.000000 | 0.00% | negligible (<1 %) |
| L_levelset | 0.034349 | 0.5 | 0.017175 | 54.08% | large (50–100 %) |
| L_crit | 6.064671 | 0.0 | 0.000000 | 0.00% | negligible (<1 %) |
| L_uv (baseline) | 0.031759 | — | 0.031759 | 100.00% | — |
| **total_loss** | | | **0.103109** | | |

### Candidate D (moderate + WPD)

`lambda_speed=0.01  lambda_grad=0.05  lambda_wpd=1e-05  lambda_levelset=0.25  lambda_crit=0.0`

| Term | Raw mean | Lambda | Weighted contrib | % of L_uv | Regime |
|---|---|---|---|---|---|
| L_speed | 1.270521 | 0.01 | 0.012705 | 40.00% | moderate (10–50 %) |
| L_grad | 0.575292 | 0.05 | 0.028765 | 90.57% | large (50–100 %) |
| L_wpd | 231.670907 | 1e-05 | 0.002317 | 7.29% | minor (1–10 %) |
| L_levelset | 0.034349 | 0.25 | 0.008587 | 27.04% | moderate (10–50 %) |
| L_crit | 6.064671 | 0.0 | 0.000000 | 0.00% | negligible (<1 %) |
| L_uv (baseline) | 0.031759 | — | 0.031759 | 100.00% | — |
| **total_loss** | | | **0.084133** | | |

### Candidate C1 (B + crit-value)

`lambda_speed=0.01  lambda_grad=0.05  lambda_wpd=0.0  lambda_levelset=0.25  lambda_crit=0.001`

| Term | Raw mean | Lambda | Weighted contrib | % of L_uv | Regime |
|---|---|---|---|---|---|
| L_speed | 1.270521 | 0.01 | 0.012705 | 40.00% | moderate (10–50 %) |
| L_grad | 0.575292 | 0.05 | 0.028765 | 90.57% | large (50–100 %) |
| L_wpd | 231.670907 | 0.0 | 0.000000 | 0.00% | negligible (<1 %) |
| L_levelset | 0.034349 | 0.25 | 0.008587 | 27.04% | moderate (10–50 %) |
| L_crit | 6.064671 | 0.001 | 0.006065 | 19.10% | moderate (10–50 %) |
| L_uv (baseline) | 0.031759 | — | 0.031759 | 100.00% | — |
| **total_loss** | | | **0.087881** | | |

### Candidate C2 (B + stronger crit-value)

`lambda_speed=0.01  lambda_grad=0.05  lambda_wpd=0.0  lambda_levelset=0.25  lambda_crit=0.0025`

| Term | Raw mean | Lambda | Weighted contrib | % of L_uv | Regime |
|---|---|---|---|---|---|
| L_speed | 1.270521 | 0.01 | 0.012705 | 40.00% | moderate (10–50 %) |
| L_grad | 0.575292 | 0.05 | 0.028765 | 90.57% | large (50–100 %) |
| L_wpd | 231.670907 | 0.0 | 0.000000 | 0.00% | negligible (<1 %) |
| L_levelset | 0.034349 | 0.25 | 0.008587 | 27.04% | moderate (10–50 %) |
| L_crit | 6.064671 | 0.0025 | 0.015162 | 47.74% | moderate (10–50 %) |
| L_uv (baseline) | 0.031759 | — | 0.031759 | 100.00% | — |
| **total_loss** | | | **0.096978** | | |

## Output files

- Per-batch CSV: `ttk_runs_fixed/topology_finetuning/loss_magnitude_diagnostic.csv`
- This report: `docs/topology_finetuning_loss_diagnostic.md`

## Notes

- L_uv is the normalized [u,v] MSE loss used in baseline pretraining; it is dimensionless (both channels normalized to ~unit variance).
- L_speed and L_grad are in physical units (m²/s² and grid-cell² respectively) after per-channel denormalization: `u_phys = sig[0]*u_norm + mu[0]`, `v_phys = sig[1]*v_norm + mu[1]`, `speed = sqrt(u² + v² + ε)`.
- L_wpd uses MAE on `speed³` (not MSE) to keep the magnitude range tractable; even so it remains ~100–300× L_uv and should only be enabled with very small lambda.
- L_levelset is a soft superlevel-set proxy, not a merge-tree loss. It uses `sigmoid(k*(speed−τ))` at physical thresholds [5,10,15] m/s.
- L_crit is a critical-value proxy loss: MSE is concentrated at GT local-speed maxima that exceed `mean + crit_high_z * std` per sample.  These maxima correspond to superlevel-set component births.  Inspired by Kissi et al. (not a full PD loss).
- No checkpoint was modified during this diagnostic run.
