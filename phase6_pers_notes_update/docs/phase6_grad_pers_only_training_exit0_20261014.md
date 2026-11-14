# Phase 6 PERS-only training / evaluation handoff

---

# Part LXXI — Phase 6 PERS-only: successful training and paired-inference pipeline; quantitative evaluation pending (October 14, 2026)

## LXXI.1 New user-supplied evidence and scope

On **October 14, 2026, at approximately 00:11 Central Time**, the third new Phase 6 arm, `grad_pers_only`, returned to a Spark `tmux` shell and printed **`Training pipeline exit status: 0`**. The final user-supplied excerpt contains the native runner's post-completion commands for cheap evaluation and expensive topology extraction. These commands appear only at the end of `scripts/run_candidateF_expanded2688_finetune.py`, after its three-epoch training, paired inference, and `_validate_post_inference(...)` call. A successful process exit strongly supports completion of all steps; **the detailed PERS-only validation log, model-output inventory, loss statistics, and cheap-evaluation results were not included in the supplied excerpt**. This section deliberately distinguishes those uninspected values from verified facts.

Source of executable variant wrapper (Spark local): `scripts/run_candidateF_cv_pers_ablation.py`. Underlying validated runner: `scripts/run_candidateF_expanded2688_finetune.py`. Source workpack: `Phase6_CV_PERS_Spatial_Workpack.zip`; its `docs/phase6_cv_pers_ablation_protocol.md` defines the matched 2x2 design. Static checks and full PERS-only dry run passed earlier; see Part LXV.

## LXXI.2 Fixed experimental definition (unchanged)

| Configuration | Value |
|---|---|
| Variant | `grad_pers_only` |
| Method | `candidateF_grad_PERS_only_expanded2688` |
| Model | Existing pretrained PhIRE CNN, **fine-tuned** (no new SR architecture) |
| Objective | `L_uv + 0.05 L_grad + 0.002 L_TTKpers` |
| `lambda_uv` / `lambda_grad` | `1.0` / `0.05` |
| `lambda_ttkcv` / `lambda_ttkpers` | `0.0` / `0.002` |
| Other auxiliary lambdas | `lambda_speed=0`, `lambda_levelset=0`, `lambda_crit=0` (plus WPD disabled) |
| Training TFRecord | `example_data_topology_expanded_2688/wind_MR-HR.tfrecord` (2,688 samples) |
| Fixed benchmark TFRecord | `example_data_fixed/wind_MR-HR.tfrecord` (168 samples; not used for training) |
| Fixed-pair constraints | `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded2688_constraints/ttk_pd_critical_pairs_gtvalues.npz` |
| Constraints | Repaired E2 GT-value anchors, 64 selected persistence pairs per training sample; validated earlier |
| Fine-tuning | Three epochs, minibatch size four, Adam learning rate `1e-5`, pretrained normalization `mu=[[0.7684,-0.4575]]`, `sig=[[5.02455,5.9017]]` |
| Execution path | Matched E2 custom TF1-compatibility training path; **not** the older plain `pretrain()` gradient-only run |
| Runtime | Spark ARM64, `/usr/bin/python3` Python 3.12, TensorFlow 2.20 via user site; `env -u PYTHONNOUSERSITE` preserves visibility |

All four matched cells:

| Cell | Variant | Gradient | Critical value (CV) | Persistence gap (PERS) | Status of training / cheap eval (at this entry) |
|---|---|---:|---:|---:|---|
| 00 | `grad_pair_zero` | 0.05 | 0 | 0 | **PASS / PASS**, see Parts LXVII–LXVIII |
| 10 | `grad_cv_only` | 0.05 | 0.004 | 0 | **PASS / PASS**, see Part LXX |
| 01 | `grad_pers_only` | 0.05 | 0 | 0.002 | **EXIT 0 / NOT YET REPORTED** |
| 11 | `grad_e2_low` (original F1) | 0.05 | 0.004 | 0.002 | Archived completed / archived completed |

This is a **single-run exploratory ablation**, without replicated training seeds. The 168 fields are consecutive hours and should not be treated as 168 statistically independent meteorological events. The F1 model was previously selected using the same established benchmark; this is not a new independent test set.

## LXXI.3 Actual execution status and expected artifacts

Invoked in the separate `tmux` session `phase6_pers_only`, with console logging:

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
env -u PYTHONNOUSERSITE /usr/bin/python3 \
  scripts/run_candidateF_cv_pers_ablation.py \
  --variant grad_pers_only \
  2>&1 | tee logs/phase6_grad_pers_only_console.log
printf 'Training pipeline exit status: %s\n' "$?"
```

**Observed terminal result (exact):**

```text
Training pipeline exit status: 0
```

The excerpt also printed `Next steps: Cheap scalar/domain evaluation (run this first)` and the TTK command. The commands were advisory only: neither evaluator nor TTK runs as a consequence of the training launcher. No PERS-only cheap-evaluation/PD/MT/spatial measurements are reported at this point.

Paths to inspect, **not claimed independently verified until the inventory command runs**:

```text
logs/phase6_grad_pers_only_console.log
logs/wind_finetune_candidateF_grad_PERS_only_expanded2688.log
models_fixed/topology_finetuning/wind_finetune_candidateF_grad_PERS_only_expanded2688/cnn/cnn.{data-00000-of-00001,index,meta}
models_fixed/topology_finetuning/wind_finetune_candidateF_grad_PERS_only_expanded2688/cnn00001/cnn.*
models_fixed/topology_finetuning/wind_finetune_candidateF_grad_PERS_only_expanded2688/cnn00002/cnn.*
models_fixed/topology_finetuning/wind_finetune_candidateF_grad_PERS_only_expanded2688/cnn00003/cnn.*
data_out/wind_finetune_candidateF_grad_PERS_only_expanded2688/{idx,dataIN,dataGT,dataSR}.npy
```

### Verification command (read-only)

```bash
cd ~/PhIRE
METHOD=candidateF_grad_PERS_only_expanded2688
LOG="logs/wind_finetune_${METHOD}.log"
OUT="data_out/wind_finetune_${METHOD}"
MODEL="models_fixed/topology_finetuning/wind_finetune_${METHOD}"

grep -E \
  'Fine-tuning complete|Run complete|Post-inference validation complete|^\[OK\]|^\[WARNING\]|^\[report\]|Checkpoint:' \
  "$LOG" | tail -45
ls -lh "$OUT"/{idx,dataIN,dataGT,dataSR}.npy
ls -lh "$MODEL"/cnn/cnn.*
```

Check exact `idx.npy == 0..167`, finite arrays of shapes `(168,)`, `(168,100,100,2)` and `(168,500,500,2)`, and `np.array_equal=True`, `max_abs_diff=0` for both evaluation GT and IN against the fixed baseline. Inspect **GT/SR scalar-speed ranges** and `Mean absolute scalar-speed error` for severe overshoot. The CV-only run's max SR speed was 39.3331 m/s vs GT max 33.9885 m/s; this should prompt per-sample inspection but is **not** an automatic fatal failure. The PERS-only max is as yet unknown.

## LXXI.4 Next action: inexpensive numerical/domain evaluation (not yet run)

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
METHOD=candidateF_grad_PERS_only_expanded2688

env -u PYTHONNOUSERSITE /usr/bin/python3 \
  scripts/evaluate_finetune_candidate.py \
  --candidate-name "$METHOD" \
  --candidate-dir "data_out/wind_finetune_${METHOD}" \
  --cnn-dir data_out_fixed/wind_mrhr_cnn \
  --gan-dir data_out_fixed/wind_mrhr_gan \
  --merged-csv ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \
  --out-dir "ttk_runs_fixed/topology_finetuning/${METHOD}_eval" \
  2>&1 | tee "logs/phase6_${METHOD}_cheap_eval.log"

printf 'Evaluation exit status: %s\n' "$?"
```

Expected **only after successful execution**:

```text
docs/topology_finetuning_candidateF_grad_PERS_only_expanded2688_eval.md
ttk_runs_fixed/topology_finetuning/candidateF_grad_PERS_only_expanded2688_eval/all_sample_metrics_candidateF_grad_PERS_only_expanded2688.csv
ttk_runs_fixed/topology_finetuning/candidateF_grad_PERS_only_expanded2688_eval/pairwise_cnn_vs_candidateF_grad_PERS_only_expanded2688.csv
ttk_runs_fixed/topology_finetuning/candidateF_grad_PERS_only_expanded2688_eval/winner_counts_candidateF_grad_PERS_only_expanded2688.csv
ttk_runs_fixed/topology_finetuning/candidateF_grad_PERS_only_expanded2688_eval/adjacent_cluster_table_candidateF_grad_PERS_only_expanded2688.csv
logs/phase6_candidateF_grad_PERS_only_expanded2688_cheap_eval.log
```

The evaluator should produce four methods × 168 benchmark fields = **672 rows** if successful. This is an **expectation**, not an observed result for PERS-only. As with gradient-only and CV-only, the local NumPy 2.4.2/scikit-image ABI mismatch may cause SSIM to be `NaN`; do not equate missing SSIM with a scientifically measured zero, and do not modify the successful TF installation merely to suppress that warning. Extract the new report's actual PSNRuv, speed MAE/RMSE, gradient MAE/W1, WPD MAE/W1/bias, PSD errors, exceedance errors, component curve L1, and within-168 win counts.

For quick inspection after the cheap evaluation, without importing NumPy or pandas:

```bash
cd ~/PhIRE
/usr/bin/python3 - <<'PYMETRICS'
import csv
from pathlib import Path
method='candidateF_grad_PERS_only_expanded2688'
p=(Path('ttk_runs_fixed/topology_finetuning') / (method+'_eval') /
   ('pairwise_cnn_vs_'+method+'.csv'))
with p.open(newline='') as f:
    rows={r['metric']: r for r in csv.DictReader(f)}
for metric in ('psnruv','speed_mae','speed_rmse','grad_mae','grad_w1',
               'wpd_mae','wpd_w1','psd_log_l2','psd_slope_abs_delta',
               'exceed_abs_t5','exceed_abs_t10','exceed_abs_t15',
               'comp_curve_l1','ssim'):
    r=rows.get(metric)
    if r:
        print(f"{metric:22s} CNN={r['mean_baseline_cnn']}  "
              f"PERS={r['mean_'+method]}  delta={r['delta']}  "
              f"improved={r['n_improved']}/{r['n_valid']}")
PYMETRICS
```

After obtaining the report, complete matched side-by-side analysis of control, CV-only, PERS-only and archived F1 (historical F1 report: `docs/topology_finetuning_candidateF_grad_E2_low_expanded2688_eval.md`). For lower-is-better metric `Y`, cell-effect analysis should explicitly report:

```text
CV main effect   = ((Y_CV - Y_control) + (Y_F1 - Y_PERS))/2
PERS main effect = ((Y_PERS - Y_control) + (Y_F1 - Y_CV))/2
interaction      = Y_F1 - Y_CV - Y_PERS + Y_control
```

These are descriptive within-benchmark factorial contrasts; controlling training randomness and accounting for temporal dependence are future work. **Do not infer PD, MT or spatial persistence improvements from component-count proxy metrics.**

## LXXI.5 Not started: corrected topology and spatial persistence evaluations

The runner's printed TTK command is not evidence that TTK ran. The historical TTK output `pd_distance` is **not** the accepted corrected-PD analysis. When the cheap evaluations are confirmed non-catastrophic, compute topology consistently across the three new variants using the same 160×160 top-left scalar-speed patch and TTK conventions as F1, then recompute corrected `d_B`, `W_{2,∞}`, `W_{2,2}`, and MT distance. Validate new diagrams with the independently audited GUDHI procedure; do **not** quote the legacy pipeline's superseded PD distance as the corrected value. Follow with the identical Phase 5E locality-constrained matching (radii 4, 8, 16, 32, 64, infinity), reporting GT coverage separately from constrained matching cost.

Do not begin expensive TTK work solely because training exited 0: first inspect PERS-only numerical/domain metrics, including spectral regressions and possible wind-speed overshoots. Keep original published checkpoints, fixed benchmark data, F1 results and frozen tag `topoinvis-2026-arxiv-v1.1` untouched.

## LXXI.6 Provenance / Git worktree / outstanding evidence

This documentation addition is generated separately from Spark; it is **not** evidence of executing a command or editing the user's remote repository. User previously worked on `master`, with unrelated uncommitted changes under `figures/paper_sample78/` and `scripts/paper_sample78_figure.py`. Apply only a patch matching the local document's latest `# Part ...` section (`git apply --check` first). The last explicitly shown successful documentation update was earlier than the optional Part LXX patch: do not assume Part LXX is already present until checking. Do not run `git reset --hard`, discard unrelated work, or stage every untracked file.

**Evidence still requested:** PERS-only validation `grep`/checkpoint inventory; PERS-only evaluator output and full Markdown report; any overshoot samples; eventually checkpoint/data hashes and corrected PD/MT/spatial results. The completed training of all three new arms is an important milestone, but **scientific comparison of all four factorial cells is not finished until PERS-only evaluation and paired topology metrics are available**.
