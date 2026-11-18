# Phase 8 — One-step Adam optimizer gate (separate from dynamic fine-tuning)

**Prepared:** November 16, 2026. **Status:** Syntax and pure-Python selftest pass locally; **not executed on DGX Spark**. No claim of improved W₂,₂ can be made until Spark's `compare` and `verify` stages pass.

## Why this is the next gate

The verified `phase8_tf1_dynamic_single_batch_gate_20261116_v2.py` established the complete computational chain: pretrained native TF1 PhIRE prediction → TTK full 160×160 speed-field persistence diagrams → canonical Phase 6 H0/H1 GUDHI Euclidean-ground W₂,₂ matching → TF1 `tf.gather_nd` surrogate → nonzero generator-parameter gradient. The latest reported check was `J_TF=427.376465`, `W22²_GUDHI=427.376501`, maximum field-gradient difference `4.77e-7`, generator-parameter gradient norm `12975.8`. Those measurements involved no optimizer step.

This new gate tests the *effect* of a single in-memory update, while keeping its claim narrow. Both arms start from precisely the same original CNN checkpoint, on **training TFRecord sample 0 only**. Each receives exactly one Adam step. No trained weights or optimizer states are saved.

- **Control:** `L_uv + 0.05 L_grad`.
- **Dynamic-PD arm:** `L_uv + 0.05 L_grad + λ_dynamic × J`, where `J=W22²` as encoded by the verified real-field TTK/GUDHI predicted-vertex matching, locally frozen for **one** step.
- Default Adam learning rate `1e-6`; default `λ_dynamic=1e-5`. These are guarded *diagnostic* values, not retuned hyperparameters or final recommendations. The unscaled `J` is a SUM over all positive finite D0/D1 pairs, and it is not the previous 64-pair CV/PERS loss or the original Candidate Dpd lambda.

The two independent TensorFlow sessions initialize their own Adam state, restore the identical pretrained generator checkpoint, and compute the before/after field and losses. The forward and matching artifacts are verified against their existing SHA manifests. `--mode step` writes only two post-update 160×160 **scalar speed arrays** and a manifest, under a new isolated directory. It never writes any model weights/checkpoint, modifies source code, or accesses the 168 evaluation fields.

Crucially, `--mode step` evaluates the **frozen** matching cost after updating weights. Such a cost can fall even when the actual topology distance does not. Therefore **do not interpret the step until** `--mode compare` re-extracts fresh TTK diagrams of the post-update fields and computes canonical GUDHI W₂,₂ for each arm.

## Files and output isolation

Place `phase8_tf1_one_step_optimizer_gate_20261116.py` at the **PhIRE repository root**, beside the original, unchanged v2 bridge script. The gate will read:

- `ttk_runs_fixed/topology_finetuning/phase8_tf1_single_batch_bridge_20261116_v2/{forward,match,tfcheck}/`
- `models/wind_mr-hr/trained_cnn/cnn` (read only)
- Native PhIRE source plus the earlier validated canonical TTK/GUDHI modules

It refuses an existing output directory:

`ttk_runs_fixed/topology_finetuning/phase8_one_step_optimizer_20261116/`

Do not remove the existing v2 bridge directory. Do not use `git add .`.

## Commands on DGX Spark

Run **separately**, stopping if any step fails:

```bash
cd ~/PhIRE

# 1: Independent Python-only numerical selftest; no writes
python3 phase8_tf1_one_step_optimizer_gate_20261116.py --mode selftest

# 2: Read-only preflight; requires verified v2 results
python3 phase8_tf1_one_step_optimizer_gate_20261116.py --mode preflight

# 3: TWO in-memory single-step Adam updates in your native PhIRE TF environment
python3 phase8_tf1_one_step_optimizer_gate_20261116.py --mode step \
  --learning-rate 1e-6 --lambda-dynamic 1e-5

# 4: RECOMPUTE true W22 in gudhi-audit (not the TF environment)
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase8_tf1_one_step_optimizer_gate_20261116.py \
  --mode compare --ttk-backend docker --ttk-timeout 180

# 5: Read-only independent recheck of stored TTK and GUDHI results
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase8_tf1_one_step_optimizer_gate_20261116.py --mode verify
```

**If an assertion fails**, preserve the output. Do not rerun `step` against the same directory, change tolerances, manually remove artifacts, or claim failure/success of the method without first inspecting the error. The script deliberately does not require the dynamic arm to outperform the baseline: a negative or inconclusive one-step result is still valid diagnostic evidence.

## How to read the results

Look for these console lines:

- `[STEP] baseline:` and `[STEP] dynamic:` — after-step change in the *frozen-assignment surrogate*, not true PD loss.
- `[STEP-GATE] PASS` — one Adam update executed per arm without writing weights.
- `[COMPARE] baseline:` and `[COMPARE] dynamic:` — the freshly recomputed canonical **true** W₂,₂.
- `[VERIFY] PASS` — evidence and the canonical metric rechecked.
- `[RESULT] initial_W22=… baseline_after=… dynamic_after=…` — first usable comparison, on **one training field only**.

The result files live under the new output directory's `step/manifest.json` and `compare/result.json`. They record run parameter values, source/checkpoint hashes, source and output checksums, baseline/dynamic objectives, and true post-step W₂,₂. Model and raw TTK artifacts remain untouched outside the new directory.

## Interpretation and next stage

This gate is **not dynamic rematching within a training loop**: it uses one initial diagram matching for one optimizer update, then *recomputes* the diagrams for evaluation. A full Phase 8 training implementation must calculate the current prediction and re-match during each scheduled training step, with proper GT diagram caching, loss normalization/weight selection, timing, and multi-field controls. The 168 held-out fields remain reserved for eventual reporting.

## Known environment caveat

Spark has printed NumPy 2.x versus NumPy-1-built `numexpr`/`bottleneck` warnings when importing TensorFlow's Keras/Pandas stack. The verified v2 forward and TF gradient checks passed despite those messages. This workpack does **not** change or install Python packages. If the environment fails, stop rather than downgrading shared dependencies globally.
