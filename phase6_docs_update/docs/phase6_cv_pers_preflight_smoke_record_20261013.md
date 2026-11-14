# Phase 6 CV/PERS Ablation — 2026-10-13 Spark Preflight and Compatibility Record

> This is an extracted copy of Part LXV in the master repository notes, `dataset_generation_and_repair_notes.md`. Its main purpose is to provide the standalone Phase 6 status and reproducibility commands.

# Part LXV — Phase 6: Fixed-Pair Critical-Value/Persistence-Gap Ablation Preflight and TF Compatibility (2026-10-13)

**Evidence status: PRE-FLIGHT + GRAPH SMOKE TEST ONLY.** All three new-variant dry runs passed; the original pretrained CNN checkpoint was restored; the gradient-only PhIRE forward loss was finite; and an optimizer graph was built. **No Phase 6 optimizer step, training epoch, new checkpoint, paired inference, TTK evaluation, or new scientific metric was executed in the work reported here.** The previously frozen TopoInVis/arXiv v1.1 and Phase 5E conclusions remain unchanged. Date refers to the Oct 12 23:xx CDT Spark session continuing into Oct 13, 2026; the smoke log includes the timestamp `2026-10-12 23:57:38`.

## LXV.1 Frozen repository baseline and newly added files

- Source release: `https://github.com/Ardad2/PhIRE/releases/tag/topoinvis-2026-arxiv-v1.1`.
- Master notes: repository root `dataset_generation_and_repair_notes.md`. The notes uploaded for this update had Git blob SHA-1 `ece956f1ba1ca624e6fc446203145074ade78d15`, **identical to the copy in the frozen release** before this addendum. Do not rewrite or force-move the publication release tag when recording new work; commit follow-up work to a separate active branch.
- Workpack transferred to Spark: `Phase6_CV_PERS_Spatial_Workpack.zip`.
- Extracted workpack: `phase6_workpack/`.
- Files installed from workpack into existing repository `scripts/`:
  - `scripts/run_candidateF_cv_pers_ablation.py` — parameterized wrapper for three new ablation variants; delegates to the existing Candidate F training runner.
  - `scripts/test_phase6_2x2_static.py` — non-TensorFlow static check of the four factorial configurations.
  - `scripts/plot_phase5_spatial_summary.py` — recreates presentation figures from frozen Phase 5E aggregate summaries.
- Existing native runner **left unchanged**: `scripts/run_candidateF_expanded2688_finetune.py`. It owns preflight, safety/collision checks, TF1 compatibility training, paired inference and printed follow-up evaluation commands.
- Workpack protocol and interpretation:
  - `phase6_workpack/docs/README.md`
  - `phase6_workpack/docs/phase6_cv_pers_ablation_protocol.md`
  - `phase6_workpack/docs/phase5_presentation_findings.md`
- Archived Phase 5E reference data in workpack:
  - `phase6_workpack/data/phase5e_macro_from_tag.csv`
  - `phase6_workpack/data/phase5e_stable_radius_from_tag.csv`
- Regenerated spatial figures in `phase6_workpack/figures/` (each available as `.png` and `.svg`): `phase5_gt_coverage`, `phase5_sr_coverage`, `phase5_constrained_cost`, `phase5_stable_advantage`. These are graphical presentations of already-frozen Phase 5E results, **not evidence from new Phase 6 training**.

Exact preparation and static validation transcript:

```bash
cd ~/PhIRE
unzip Phase6_CV_PERS_Spatial_Workpack.zip -d phase6_workpack
cp phase6_workpack/scripts/*.py scripts/
python3 scripts/test_phase6_2x2_static.py
python3 scripts/run_candidateF_cv_pers_ablation.py --print-config
```

Observed static-check output:

```text
PASS: four distinct arms; fixed constraints; no unintended auxiliary losses.
NOTE: this checks configuration logic only; TF1 training/evaluation is NOT run.
...
All per-variant assertions verified: no unexpected loss is active in any variant.
grad_crit confirmed: requires_e2=False, constraints_npz=NONE (no E2 constraint loader or E2 loss ops can execute for this variant).
```

The print-config includes three historical Candidate F variants (`grad_e2_low`, `grad_levelset_e2_low`, `grad_crit`) plus three **new** variants (`grad_pair_zero`, `grad_cv_only`, `grad_pers_only`). The separate `grad_crit` arm is *not* a member of the present fixed-pair 2×2 ablation.

## LXV.2 Factorial objective and fixed experimental controls

The proposed fixed-pair experiment tests whether critical-value supervision and persistence-gap supervision contribute independently or interact, **at the same fixed training schedule and dataset**. Every arm has the original vector reconstruction loss (`lambda_uv=1.0`) and gradient-magnitude loss (`lambda_grad=0.05`); all other scalar proxy weights are zero except the indicated fixed-pair term.

| Arm | `lambda_ttkcv` | `lambda_ttkpers` | Meaning |
|---|---:|---:|---|
| `grad_pair_zero` | 0.000 | 0.000 | Gradient-only control **using the same E2-capable training code path** |
| `grad_cv_only` | 0.004 | 0.000 | Gradient + fixed-pair birth/death critical-value supervision |
| `grad_pers_only` | 0.000 | 0.002 | Gradient + fixed-pair persistence-gap supervision |
| Published `grad_e2_low` / F1 | 0.004 | 0.002 | Gradient + both fixed-pair terms (already completed and frozen) |

In all **three new variants**, `requires_e2=True` and both fixed-pair loss operators may be constructed even when one or both weights are zero; therefore a log label `repaired E2 ACTIVE` must **not** be interpreted as a nonzero loss contribution. It records the code path. This fourth-arm control is useful because the older independently trained gradient-only run took a different plain-training path.

Controls confirmed during dry runs:

```text
training TFRecord:   example_data_topology_expanded_2688/wind_MR-HR.tfrecord
training records:    2688
evaluation TFRecord: example_data_fixed/wind_MR-HR.tfrecord
evaluation fields:   168 (held out; no training)
source checkpoint:   models/wind_mr-hr/trained_cnn/cnn
constraints NPZ:     ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded2688_constraints/ttk_pd_critical_pairs_gtvalues.npz
optimizer:           Adam, learning rate 1e-5
schedule:            3 epochs, batch size 4, save_every 1
normalization mean:  [0.7684, -0.4575]
normalization std:   [5.02455, 5.9017]
other aux weights:   lambda_speed=0, lambda_wpd=0, lambda_levelset=0, lambda_crit=0
```

The constraints validator reported:

```text
stored n_samples:                     2688
sample_idx.shape:                    (2688,)
sample_start.shape:                  (2688,)
sample_count.shape:                  (2688,)
sample_idx range:                    [0, 2687]
constraints/sample index alignment:   exact {0..2687}, each once
pairs per sample:                    min=64, mean=64.00, max=64
total fixed pairs:                   172032
birth vertex ID bounds:              [0, 25599]
death vertex ID bounds:              [1, 25597]
finite birth/death/persistence:       True
persistence nonnegative:             True
TFRecord sample indices == NPZ set:  EXACT match
NPZ metadata validation:             PASSED
```

## LXV.3 Initial failed dry-run: user-site packages disabled

The first attempt was:

```bash
for VARIANT in grad_pair_zero grad_cv_only grad_pers_only; do
    PYTHONNOUSERSITE=1 /usr/bin/python3 \
        scripts/run_candidateF_cv_pers_ablation.py \
        --variant "$VARIANT" --dry-run || break
done
```

It **failed before preflight** on the first arm with:

```text
File ".../scripts/run_candidateF_expanded2688_finetune.py", line 510, in main
    import tensorflow.compat.v1 as tf
ModuleNotFoundError: No module named 'tensorflow'
```

Root cause: `PYTHONNOUSERSITE=1` hides TensorFlow from the user-site directory. This was an inappropriate launcher choice for the existing Spark installation, **not evidence that TensorFlow needed reinstalling**. The original dataset/checkpoints remained untouched.

## LXV.4 Python/TensorFlow environment and import warnings

Corrected environment invocation:

```bash
env -u PYTHONNOUSERSITE /usr/bin/python3 - <<'PY'
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
print("TensorFlow path:", tf.__file__)
print("TensorFlow import: SUCCESS")
PY
```

Recorded successful output:

```text
Python: 3.12.3 (main, Jan  8 2026, 11:30:50) [GCC 13.3.0]
TensorFlow: 2.20.0
TensorFlow path: /home/adadhwal/.local/lib/python3.12/site-packages/tensorflow/_api/v2/compat/v1/__init__.py
TensorFlow import: SUCCESS
```

**Dependency caveat, not silently resolved:** The import printed repeated NumPy 1.x-compiled extension incompatibility diagnostics under NumPy **2.4.2**. The failing optional modules in the shown stack traces were system copies of `numexpr` (`/usr/lib/python3/dist-packages/numexpr`) and `bottleneck` (`/usr/lib/python3/dist-packages/bottleneck`), imported during pandas/Keras startup. Matplotlib also warned about unavailable `Axes3D` from mixed installations; TF emitted v1/deprecation warnings. Despite those stack traces, TF2.20's `tensorflow.compat.v1` import and the subsequent smoke tests **completed successfully**. Do not infer that all training-time operations or the environment are entirely compatible merely from import success. Do not run `pip install --upgrade` or `pip install --user` against the system environment without a separate compatibility plan.

## LXV.5 All three corrected ablation dry-runs: PASS

Corrected invocation (retains user-site packages):

```bash
cd ~/PhIRE
for VARIANT in grad_pair_zero grad_cv_only grad_pers_only; do
    env -u PYTHONNOUSERSITE /usr/bin/python3 \
        scripts/run_candidateF_cv_pers_ablation.py \
        --variant "$VARIANT" --dry-run || break
done
```

| New variant | Dry-run status | Candidate output method | Intended model output |
|---|---|---|---|
| `grad_pair_zero` | PASS | `candidateF_grad_pairzero_expanded2688` | `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688/` |
| `grad_cv_only` | PASS | `candidateF_grad_CV_only_expanded2688` | `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_CV_only_expanded2688/` |
| `grad_pers_only` | PASS | `candidateF_grad_PERS_only_expanded2688` | `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_PERS_only_expanded2688/` |

The confirmed dry-run endpoint for each variant was:

```text
[safety] Output paths do not overlap any protected directory. OK.
[collision-check] No completed-looking output exists yet ... OK to proceed.
--- Constraints NPZ metadata validation: PASSED ---
Training TFRecord index set == constraints NPZ index set EXACTLY
[dry-run] All preflight/collision/NPZ-validation checks passed. No training performed.
[dry-run] No canonical training log or any other file was created by this dry run.
```

These dry-runs do **not** validate actual convergence, scientific correctness of the loss implementation, optimizer execution, paired inference, VTI extraction, or corrected PD distance computation.

### Explicit intended artifact paths (after future successful runs)

```text
model checkpoints (per arm):
  models_fixed/topology_finetuning/wind_finetune_<method>/
paired inference (per arm):
  data_out/wind_finetune_<method>/{idx.npy,dataIN.npy,dataGT.npy,dataSR.npy}
cheap scalar/domain evaluation:
  ttk_runs_fixed/topology_finetuning/<method>_eval/
TTK intermediate VTI:
  ttk_runs_fixed/topology_finetuning/<method>_topology_vti/
TTK output PD / MT:
  ttk_runs_fixed/topology_finetuning/<method>_topology/
canonical training log:
  logs/wind_finetune_<method>.log
```

In these patterns, `<method>` is one of the three exact `candidateF_grad_*_expanded2688` method names in the table, and the checkpoint directory is prefixed with `wind_finetune_` as shown. The runner prints the exact per-arm paths at startup and after inference. The expensive TTK stage does **not** run automatically.

## LXV.6 Actual generator/optimizer/checkpoint smoke test: PASS

Command as provided and tested in the Spark PhIRE working directory (synthetic zero tensors; no training step):

```bash
cd ~/PhIRE
env -u PYTHONNOUSERSITE /usr/bin/python3 - <<'PY'
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PhIREGANs import PhIREGANs
from sr_network import SR_NETWORK, PhysicsLossConfig

MU_SIG = [[0.7684, -0.4575], [5.02455, 5.9017]]
tf.reset_default_graph()

phire = PhIREGANs(
    data_type="phase6_smoke",
    learning_rate=1e-5,
    mu_sig=MU_SIG,
)
phire.set_LR_data_shape(
    "example_data_topology_expanded_2688/wind_MR-HR.tfrecord"
)
h, w, c = phire.LR_data_shape
x_lr = tf.placeholder(tf.float32, [None, h, w, c])
x_hr = tf.placeholder(tf.float32, [None, h*5, w*5, c])

cfg = PhysicsLossConfig(
    use_aux_losses=True,
    mu=MU_SIG[0], sig=MU_SIG[1], lambda_grad=0.05,
)
model = SR_NETWORK(
    x_lr, x_hr, r=[5], status="pretraining", phys_cfg=cfg,
)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
train_op = optimizer.minimize(model.g_loss, var_list=model.g_variables)
saver = tf.train.Saver(var_list=model.g_variables)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "models/wind_mr-hr/trained_cnn/cnn")
    print("CHECKPOINT RESTORE: PASS")
    lr = np.zeros((1, h, w, c), dtype=np.float32)
    hr = np.zeros((1, h*5, w*5, c), dtype=np.float32)
    loss = sess.run(model.g_loss, feed_dict={x_lr: lr, x_hr: hr})
    assert np.isfinite(loss), f"Nonfinite loss: {loss}"
    print("FORWARD LOSS: PASS", float(loss))
    print("OPTIMIZER GRAPH: PASS (no update executed)")
PY
```

Observed end of log (full import stack warnings omitted here but described in LXV.4):

```text
Loading data ... CHECKPOINT RESTORE: PASS
FORWARD LOSS: PASS 3.05885259876959e-05
OPTIMIZER GRAPH: PASS (no update executed)
```

**Interpretation:** The TensorFlow v1-compat generator/loss can be built, PhIRE pretrained CNN weights can be restored into the graph, a finite forward loss can be evaluated on synthetic all-zero inputs, and an optimizer update graph can be constructed. `train_op` was **never executed**; there is no evidence here of a successful backward training step, persistence-pair loss runtime, checkpoint save, or restored evaluation. The numerical loss above is a synthetic compatibility result, **not a scientific experiment metric**.

## LXV.7 Next action: first actual fine-tuning run (NOT yet executed in this record)

Recommendation: run **one arm at a time**, beginning with the new `grad_pair_zero` custom-loop control, and inspect the first batch/epoch and output log before launching the two other arms. Keep the frozen release tag intact and do not overwrite any existing checkpoints/artifacts.

Example tmux session (commands proposed, not verified as executed by this record):

```bash
cd ~/PhIRE
mkdir -p logs
tmux new -s phase6_pairzero
# Within tmux:
set -o pipefail
env -u PYTHONNOUSERSITE /usr/bin/python3 \
    scripts/run_candidateF_cv_pers_ablation.py \
    --variant grad_pair_zero \
    2>&1 | tee logs/phase6_grad_pairzero_console.log
```

The native runner also writes its canonical training log under `logs/wind_finetune_candidateF_grad_pairzero_expanded2688.log`. Only proceed when the full training/inference run succeeds and the output alignment checks pass; do not interpret the smoke test as evidence that this full run has succeeded.

After the run, invoke the printed cheap evaluation command and **only after checking the metrics** consider the expensive TTK PD/MT pipeline. Example cheap evaluation for the first arm (also not yet executed):

```bash
cd ~/PhIRE
env -u PYTHONNOUSERSITE /usr/bin/python3 \
    scripts/evaluate_finetune_candidate.py \
    --candidate-name candidateF_grad_pairzero_expanded2688 \
    --candidate-dir data_out/wind_finetune_candidateF_grad_pairzero_expanded2688 \
    --cnn-dir data_out_fixed/wind_mrhr_cnn \
    --gan-dir data_out_fixed/wind_mrhr_gan \
    --merged-csv ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \
    --out-dir ttk_runs_fixed/topology_finetuning/candidateF_grad_pairzero_expanded2688_eval
```

TTK extraction (also not yet executed) must use the corresponding `--method`, `--data-dir`, `--vti-dir`, and `--out-base` from the runner's printed output; use `--n-samples 168 --threads 1 --skip-viz`. **Use the corrected PD-distance implementation/audited convention from the publication reproducibility guide for scientific PD comparisons, not superseded historical raw PD scores.** Do not assume that the cheap evaluation automatically computes corrected persistence distances.

### Phase 6 status snapshot (as of this recorded session)

| Component | Status |
|---|---|
| Phase 5E spatial-locality analysis and summary | FROZEN, completed previously |
| New Phase 6 workpack extraction and static four-arm assertions | PASS |
| TF2.20 import through TF1 compatibility API | PASS, dependency warnings persist |
| All 3 new-arm preflights, checks and collisions | PASS |
| Generator forward + checkpoint restore + optimizer graph smoke test | PASS (no weight update) |
| Real `grad_pair_zero` custom-loop train / inference | **NOT RUN** |
| Real `grad_cv_only` train / inference | **NOT RUN** |
| Real `grad_pers_only` train / inference | **NOT RUN** |
| New per-arm cheap scalar/domain metrics | **NOT RUN** |
| New corrected PD / merge-tree results | **NOT RUN** |
| Phase 6 scientific findings or published comparison | **NONE YET** |

This section is an append-only run log for reproducibility. Future updates should record environment changes (including pip freeze if needed), first successful optimizer execution, checkpoint paths, sample-index alignment, canonical logs, numerical evaluation output, corrected TTK PD/MT output, and results with exact corresponding script versions and input hashes. Avoid modifying the released v1.1 tag or altering the frozen Phase 5E dataset-wide results.
