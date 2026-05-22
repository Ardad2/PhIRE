# Candidate E TTK Critical-Pair Refiner Notes

**Generated:** 2026-05-20T13:48:02

Candidate E = Candidate C losses + Candidate B level-set loss + TTK critical-pair losses (Kissi-style).

## Configuration

```
  data_dir                       = data_out_fixed/wind_mrhr_cnn
  constraints                    = ttk_runs_fixed/topology_finetuning/candidateE2_constraints/ttk_pd_critical_pairs_gtvalues.npz
  out_dir                        = /home/adadhwal/PhIRE/data_out/wind_finetune_pilot_candidateE
  model_dir                      = /home/adadhwal/PhIRE/models_fixed/topology_finetuning/wind_finetune_pilot_candidateE
  log_path                       = /home/adadhwal/PhIRE/logs/wind_finetune_pilot_candidateE.log
  report_path                    = /home/adadhwal/PhIRE/docs/candidateE_ttkcrit_refiner_notes.md
  epochs                         = 3
  lr                             = 0.0001
  lambda_speed                   = 0.01
  lambda_grad                    = 0.05
  lambda_crit                    = 0.001
  lambda_levelset                = 0.25
  lambda_ttkcv                   = 0.04
  lambda_ttkpers                 = 0.02
  residual_scale                 = 0.1
  seed                           = 42
  max_pairs_per_sample           = 0
  allow_partial_constraints      = False
  diagnostic_only                = False
  dry_run                        = True
```

## Diagnostic Loss Breakdown

- Real data          : YES
- Real constraints   : YES
- Constraint samples : 168
- First sample id    : 0
- Pairs used         : 64

| Term | Raw value | λ | Weighted |
|------|-----------|---|----------|
| L_uv      | 1.119733 | 1.0 | 1.119733 |
| L_speed   | 1.647115 | 0.01 | 0.016471 |
| L_grad    | 0.631257 | 0.05 | 0.031563 |
| L_crit    | 9.610543 | 0.001 | 0.009611 |
| L_levelset| 0.036278 | 0.25 | 0.009069 |
| L_ttkcv   | 1.395823 | 0.04 | 0.055833 |
| L_ttkpers | 2.865361 | 0.02 | 0.057307 |

L_ttkcv / L_uv   = **1.2466×**
L_ttkpers / L_uv = **2.5590×**
TTK loss time    = 0.39 ms
L_ttkcv → model  = **YES**

## Persistence Sign Convention

TTK stores `persistence = |death_scalar - birth_scalar|` (unsigned).
The extraction script stores `death_val = birth_val + persistence_raw`,
which gives `|death_val - birth_val| = persistence_raw`.
Both `L_ttkcv` and `L_ttkpers` are direction-agnostic:
`L_ttkpers` penalises `|sr_death - sr_birth| vs gt_persistence`.

## Training Loss History

| Epoch | L_uv | L_speed | L_grad | L_crit | L_levelset | L_ttkcv | L_ttkpers | L_total |
|-------|------|---------|--------|--------|------------|---------|-----------|--------|
| 1 | 1.20168 | 1.75312 | 0.69673 | 10.85547 | 0.03603 | 1.49146 | 2.35060 | 1.38058 |

