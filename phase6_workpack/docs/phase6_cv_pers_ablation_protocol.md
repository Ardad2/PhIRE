# Phase 6A — CV vs PERS fixed-pair ablation (prepared, NOT trained)

**Reference:** Ardad2/PhIRE, tag `topoinvis-2026-arxiv-v1.1` (2026-10-10). Manuscript F1 = `candidateF_grad_E2_low_expanded2688`.

## Scientific question
What does fixed-pair critical-value supervision contribute beyond gradient loss, what does persistence-gap supervision contribute, and do they interact? The fixed target pairs are the 64 selected finite GT PD pairs in the top-left 160x160 crop, with the original repaired GT-value constraints. This experiment must not be confused with prediction-dependent PD matching, which is a separate phase.

## 2x2 design

| Arm | run name | L_grad | L_CV | L_PERS | E2 training loop |
|---|---|---:|---:|---:|---|
| none (procedural control) | `candidateF_grad_pairzero_expanded2688` | 0.05 | 0 | 0 | custom |
| CV-only | `candidateF_grad_CV_only_expanded2688` | 0.05 | 0.004 | 0 | custom |
| PERS-only | `candidateF_grad_PERS_only_expanded2688` | 0.05 | 0 | 0.002 | custom |
| both (published F1) | `candidateF_grad_E2_low_expanded2688` | 0.05 | 0.004 | 0.002 | custom |

All share the pretrained CNN initialization, normalized `L_uv`, 2688-sample training TFRecord, fixed TTK constraint NPZ, three epochs, batch size four, Adam 1e-5 and fixed evaluation benchmark. The **new procedural-control arm** makes the exact same TF1 custom-loop and constraint-loader path available for all four configurations. The previously published gradient-only experiment used the plain `pretrain()` path, so it is scientifically useful but not a perfect code-path matched control for the full factorial comparison. No random-seed controls are added; report this as a single-run exploratory ablation until replicated.

## Installation on Spark (from the repo root)

Copy `scripts/run_candidateF_cv_pers_ablation.py`, `scripts/test_phase6_2x2_static.py`, and `scripts/plot_phase5_spatial_summary.py` into the repo's `scripts/` directory. Do not modify or replace `scripts/run_candidateF_expanded2688_finetune.py`. The launcher checks its exact Git blob SHA against the v1.1 tag and refuses to run when the base has changed.

## Preflight: no training or output modifications

```bash
python3 scripts/test_phase6_2x2_static.py
python3 scripts/run_candidateF_cv_pers_ablation.py --print-config
for VARIANT in grad_pair_zero grad_cv_only grad_pers_only; do
  PYTHONNOUSERSITE=1 /usr/bin/python3 scripts/run_candidateF_cv_pers_ablation.py \
    --variant "$VARIANT" --dry-run || exit 1
done
```

The --dry-run requires the original TensorFlow environment and both data TFRecords, the original CNN checkpoint, and constraint NPZ. `--print-config` and the static test only require standard-library Python and the unmodified F script.

## Execute new runs (only after all preflights pass)

```bash
set -euo pipefail
for VARIANT in grad_pair_zero grad_cv_only grad_pers_only; do
  PYTHONNOUSERSITE=1 /usr/bin/python3 scripts/run_candidateF_cv_pers_ablation.py \
    --variant "$VARIANT"
done
```

The runner performs training, checkpoint save, 168-field paired inference, and exact GT/input/index alignment checks. It prints the subsequent evaluation commands. Every arm writes to its own model/inference/log path. It never reruns or overwrites the published F1.

## Evaluate outputs

For each new `<run>`:

```bash
python3 scripts/evaluate_finetune_candidate.py \
  --candidate-name <run> \
  --candidate-dir data_out/wind_finetune_<run> \
  --cnn-dir data_out_fixed/wind_mrhr_cnn \
  --gan-dir data_out_fixed/wind_mrhr_gan \
  --out-dir ttk_runs_fixed/topology_finetuning/<run>_eval

bash scripts/run_candidate_topology_pipeline.sh \
  --method <run> \
  --data-dir data_out/wind_finetune_<run> \
  --vti-dir ttk_runs_fixed/topology_finetuning/<run>_topology_vti \
  --out-base ttk_runs_fixed/topology_finetuning/<run>_topology \
  --n-samples 168 --threads 1 --skip-viz
```

**Important:** The topology pipeline's historical `pd_distance` output is superseded and MUST NOT be reported as the paper's bottleneck/W2,inf/W2,2. Run the repository's corrected-PD audit on the newly produced diagrams, checking the manifest/path conventions for the new runs, and independently cross-check GUDHI. Reuse the original 168-field benchmark and the same D0/D1/nonfinite-pair conventions. The revised PD audit for these new run names has **not** been automated or executed in this work package.

## Analysis plan

For a metric where *lower* is better, let Y00 = procedural control, Y10 = CV-only, Y01 = PERS-only, Y11 = both (F1). Report paired per-field differences and aggregate summaries:

- Marginal CV effect (averaged across persistence levels): `0.5*((Y10-Y00)+(Y11-Y01))`.
- Marginal PERS effect (averaged across CV levels): `0.5*((Y01-Y00)+(Y11-Y10))`.
- Interaction: `Y11-Y10-Y01+Y00`.

Also report the direct CV-only vs PERS-only contrast. Prefer bootstrap confidence intervals that resample blocks (days or contiguous meteorological episodes) rather than 168 independent hours, and repeat training with controlled seeds to estimate optimizer variation. Since F1 was selected on this benchmark, characterize these results as within-benchmark exploratory comparisons, not an independent hold-out confirmation.

Primary metrics: corrected d_B, W2,inf and W2,2, MT distance, GT-side spatial persistence coverage and SR-side coverage, locality-constrained cost at radii 4, 8, 16, 32, 64 and infinity. Secondary: PSNR-uv, speed SSIM, WPD bias/MAE, spatial gradient and physical exceedance metrics. Do not combine spatial coverage and constrained cost into one score.

## Phase5E presentation figures

The plotting utility reads the **actual released CSVs**:

```
spatial_pd/phase5e_all168_locality_analysis/all168_locality_macro_summary.csv
spatial_pd/phase5e_all168_closeout_summary/stable_radius_histogram.csv
```

```bash
python3 scripts/plot_phase5_spatial_summary.py --out-dir phase6_figures
```

It creates three independent line charts and a stable-advantage histogram, each in PNG and SVG format. Graphs are descriptive, not independent validation. For qualitative spatial panels, use actual canonical `GT/CNN/UV/F1` wind fields, map selected birth/death critical-point pairs, and show r=4/r=16 matching outcomes. Include both representative examples and samples 162/163 as exception cases. This requires the canonical VTI/VTU artifacts on Spark; no synthetic feature maps should be passed off as experimental data.

## Geographic generalization protocol (next stage)

Keep 168-hour and 2688-sample cutoffs as historically frozen. Construct separately versioned multi-region TFRecords with canonical C-order GT [u,v] and the same 100m WIND Toolkit variables, resolution, normalization and downsampling. Document region (lat/lon and grid indices), time/season, UTC timestamps, upstream HSDS dataset version and sample ID. Make a clear distinction between sample-level and meteorological-episode-level independence.

Suggested comparisons: (1) same-region equal-size temporal expansion, (2) equal-size geographically diverse fine-tuning, (3) larger diverse fine-tuning; select weights/models using a held-out validation period and evaluate on unseen regions and nonadjacent time windows. Prevent overlap or nearby-crop leakage, regenerate GT TTK constraints for every training crop, and use the same topology crop / physical spatial-resolution convention across models. Retain the previous 168-field data solely as an established *legacy* benchmark; do not use it to choose a new method then claim a novel held-out advantage on it.

## Provenance / status

- Frozen paper release: `topoinvis-2026-arxiv-v1.1` (untouched).
- Phase5E locality experiment: already CLOSED; this package only plots its frozen aggregates.
- Phase6A launcher/static tests: prepared locally; **no GPU/TF training performed here**.
- Corrected PD + MT metrics for the proposed ablation: **not available until new runs and evaluations complete**.
