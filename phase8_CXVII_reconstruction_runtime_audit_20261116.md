# Part CXVII — Phase 8: reconstruction-quality trade-off and worker timing for the eight-step pilot (November 16, 2026)

**Status: OBSERVED METRIC REVIEW; no new model training or experiment executed for this entry.** This addendum follows the verified four-field/eight-update dynamic persistence rematching pilot in Part CXVI. Evidence is the user's DGX Spark terminal output from November 16, 2026: eight saved per-step JSON records were read from `ttk_runs_fixed/topology_finetuning/phase8_multisample_rematch_20261116_r01/train/steps/iter_*/step.json`. The notes were previously committed as `25c29beb` (Part CXVI). No new checkpoint, persistence diagram, or model measurement is claimed here.

## Verified measurements printed from the saved step JSON files

The user ran a read-only Python aggregation requiring exactly eight step records, computing arithmetic means of `L_uv_after_control`, `L_uv_after_dynamic`, `L_grad_after_control`, `L_grad_after_dynamic`, and the recorded `seconds_match_worker` and `seconds_eval_worker` for the eight chronological steps. Those eight evaluations reuse four optimization samples twice; they are **not independent samples or a held-out evaluation**.

| Post-update metric averaged across 8 steps | Control | Dynamic PD | Dynamic minus control | Relative to control |
|---|---:|---:|---:|---:|
| Normalized `L_uv` (lower better) | 0.01337039 | 0.013414497 | +0.000044107321 | +0.330% (slightly worse) |
| Gradient `L_grad` (lower better) | 0.20363793 | 0.20324127 | −0.00039665774 | −0.195% (slightly better) |
| True recomputed `W22` from Part CXVI (lower better) | ≈21.443504 | ≈21.284972 | −0.158531464 | −0.739% (better) |

The `W22` means are derived from the eight verified paired results already recorded in Part CXVI, not from a new evaluation; the `L_uv` and `L_grad` means came from the follow-up step-JSON aggregation. The `L_uv` results are normalized-UV losses, **not physical speed MAE/RMSE**. `L_grad` reflects the experiment's implemented gradient metric; no new spatially stratified or physical-unit validation has been performed.

| Recorded external worker timing | Mean per update | Sum across 8 updates |
|---|---:|---:|
| `seconds_match_worker` | 1.935 s | 15.476 s |
| `seconds_eval_worker` | 2.467 s | 19.736 s |
| Both workers, combined | 4.402 s | 35.212 s |

These are worker wall times recorded by the pilot, not the entire TensorFlow training/runtime budget. The matching worker is needed by the prototype to refresh its predicted PD and assignments before each dynamic update; the after-update evaluation worker is a **diagnostic** that measures both arms' true `W22` and would not necessarily be run on every iteration in scaled training. Do not present 4.402 s as the intrinsic dynamic-loss overhead or extrapolate directly to 2,688 training samples.

## Interpretation and next decision gate

- The eight-step experiment shows a consistent within-training topology advantage while indicating a small **pointwise vector loss penalty** (+0.33%) and a small gradient-loss improvement (−0.195%) at this particular diagnostic weight `lambda_dynamic = 1e-5`, learning rate `1e-6` and order `(0,1,2,3,0,1,2,3)`. Neither the magnitude nor its trade-off is yet reliable across independent source fields, other seeds/orders, crops, or loss weights.
- **Next PLANNED test, NOT RUN:** Repeat the same frozen initialization and exact eight-step trajectories in a new versioned, non-overwriting runner that retains each final arm in memory for no-update evaluation on previously unoptimized examples. Choose evaluation samples only after verifying their underlying original field/week IDs: the 2,688-field set contains expanded records, so sequential records (e.g., 4–7) cannot be assumed source-independent. Group/stratify appropriately, preserving the sealed 168-field benchmark for a later final evaluation.
- The follow-up should report paired true `W22` and D0/D1 decomposition, normalized `L_uv`, physical speed MAE/RMSE, `L_grad`, and gradient magnitudes on both training and unoptimized development fields. Preserve frozen source hashes, initial checkpoint hashes, sample lineage, per-step timings, and compare with a matching control. No code or data modifications are authorized by this audit entry.

**Research conclusion:** The positive topology result is now accompanied by a modest pointwise trade-off and concrete worker costs. Generalization and computational scaling remain open questions.
