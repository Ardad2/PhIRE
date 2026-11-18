# Phase 8 — Original 168-field benchmark and matched-scale dynamic-PD research plan

**Prepared:** November 18, 2026. **Status:** local source compilation and structural selftest PASS; **NOT run on DGX Spark**. This document prescribes future evaluation; it asserts no new experimental results. No Phase 7 development or locked-test region is used.

## 1. Disentangle benchmark size from training size

The original `example_data_fixed/wind_MR-HR.tfrecord` is a **168-field Mid-Atlantic benchmark**, already reused in model selection and therefore a *development comparison*, not an unbiased external test. It must never be used for optimization of the new dynamic-PD model.

The original PhIRE manuscript experiments trained on `example_data_topology_expanded_2688/wind_MR-HR.tfrecord` (16 × 168 distinct WTK hours from one Mid-Atlantic footprint), with an approximately **three-epoch, batch-size-four** fine-tuning setup for the important candidate arms. The previously successful fresh-rematching pilot used only **four training fields and eight single-sample Adam updates**; even if evaluated on all 168 benchmark fields, it is **not matched to those manuscript-scale models**.

Consequently there are TWO necessary experiments:

**Gate A (ready for Spark now):** evaluate the **existing verified eight-step models** (pretrained CNN, gradient control, gradient + dynamic PD) on all **168 original Mid-Atlantic benchmark fields**, with canonical Phase 6 TTK/GUDHI `dB`, `W2inf`, `W22`, `L_uv`, `L_grad`, physical *top-left-160-crop* wind-speed MAE, and all established Phase 5E spatial radii (4, 8, 16, 32, 64, 96, 128, inf grid cells). Include GT persistence coverage, SR persistence coverage and unmatched features. Report seven consecutive 24-hour blocks and paired per-hour differences; do not treat neighboring hours as independent.

**Gate B (separate future trainer; NOT implemented in this workpack):** train an actual larger model. Use a small, seasonally stratified 168-field *TRAINING subset of the existing 2,688-field TRAINING pool* for a measured development stage; then scale to all 2,688 training fields with the **manuscript-matched batch size, epochs, initial checkpoint, learning rate, normalization, sample order, and optimizer step count**. A genuine paired gradient-only arm must have exactly the same data, batch/order, initialization, optimizer hyperparameters, and number of updates as the dynamic arm. Ideally replicate at multiple seeds once feasibility and cost have been established. The same 168-field benchmark is for evaluation only, for all arms.

For three epochs with batch size four and no dropped samples, the 168-subset stage has 126 model updates / 504 predicted-field rematchings; the full 2,688 stage has 2,016 updates / 8,064 predicted-field rematchings. Each per-example TTK persistence diagram and GUDHI assignment is recomputed at the current predicted field at its step. Group per-example `J_i = W22_i²` as `J_batch = (1/B) sum_i J_i`, not an unnormalized sum whose effective weight changes with batch size. Retain a zero-lambda control and (as resources permit) a fixed-pair control with the same base gradient loss. No hyperparameter choice should use locked regions.

**Loss used in the existing four-field proof-of-concept:**

```
L_control = L_uv + 0.05 * L_grad
L_dynamic = L_uv + 0.05 * L_grad + 1e-5 * J_dynamic
J_dynamic = W2,2(PD(predicted_speed), PD(GT_speed))**2
```

In the existing graph, the assignment and predicted critical indices are frozen within an Adam step. Fresh TTK/GUDHI rematching happens *before every new dynamic step*. The batch-size-four implementation **does not exist yet** and must be validated against the batch-size-one gate before scaling. Preserve proper diagonal assignments and GT-unmatched constant terms; diagonal costs are part of the objective, not an optional heuristic.

## 2. Gate A: auditable script, in original benchmark only

**File:** `phase8_original168_dynamic_pd_benchmark_20261118.py`; put in the PhIRE root next to the verified original Phase8 scripts. The script imports and reuses `phase8_temporal_out_of_pilot_eval_20261117.py` for the **byte-checked eight-step replay**, with the *evaluation* TFRecord replaced by `example_data_fixed/wind_MR-HR.tfrecord`. It does NOT modify the trusted source on disk.

It independently exports all 168 normalized (100×100×2) inputs and matching (500×500×2) GT fields, verifies the embedded TFRecord sample IDs 0–167 exactly, and exports top-left 160×160 physical wind-speed crops. It then replays the original eight pilot updates through two separate Adam sessions, checking archived before/after predictions at every step to the previously frozen 0.005 m/s maximum-absolute gate. It saves predicted speed **crops only**, not new checkpoint files. On the same 168 examples, it scores all GT/CNN/control/dynamic PDs using the existing TTK and canonical Phase 6 GUDHI routines.

Spatial locality uses the **unchanged, SHA-pinned original Phase 5E `SparsePrepared` implementation** (SHA256 `47c73decca4ba151fd8b250cae2911b4c2307471fe9c881fc42bfc968e211c3a`), including both birth and death critical-vertex displacements, positive-persistence handling, diagonal assignments, and eight radii. Each infinite-radius distance must agree with independently computed canonical GUDHI `W22` to 1e-5 absolute and 1e-6 relative tolerance. Source and sample outputs are SHA256-protected. Individual sample `scores/s000` etc. and `locality_samples/s000` etc. are published atomically so interruptions can resume without redoing completed fields.

**The 168 reference source was already used to choose earlier model/loss variants.** Any apparent dynamic-PD improvement here is *development evidence*, not new independent validation. The experiment does not touch the development Gulf/Southwest files or locked Upper Midwest/Pacific inland test files.

## 3. Spark commands, ordered and fail-fast

Run from `~/PhIRE` after transferring the script. Stop at the first failure; do not modify originals or relax fidelity checks silently.

```bash
cd ~/PhIRE
python3 phase8_original168_dynamic_pd_benchmark_20261118.py --mode selftest
python3 phase8_original168_dynamic_pd_benchmark_20261118.py --mode preflight
python3 phase8_original168_dynamic_pd_benchmark_20261118.py --mode export
python3 phase8_original168_dynamic_pd_benchmark_20261118.py --mode replay
```

The next stages require GUDHI + TTK and can be long. To check the first eight fields before processing the entire benchmark:

```bash
export PHASE8_MICROMAMBA="${MAMBA_EXE:-/home/adadhwal/Desktop/bin/micromamba}"
test -x "$PHASE8_MICROMAMBA" || { echo 'STOP: micromamba binary missing'; exit 1; }

PYTHONNOUSERSITE=1 "$PHASE8_MICROMAMBA" run -n gudhi-audit \
  python phase8_original168_dynamic_pd_benchmark_20261118.py \
  --mode score --ttk-backend docker --max-new-samples 8

PYTHONNOUSERSITE=1 "$PHASE8_MICROMAMBA" run -n gudhi-audit \
  python phase8_original168_dynamic_pd_benchmark_20261118.py \
  --mode locality --max-new-samples 8
```

Review those field diagnostics and **check the first locality `r∞` matches canonical GUDHI**, then resume the remainder by omitting `--max-new-samples`:

```bash
PYTHONNOUSERSITE=1 "$PHASE8_MICROMAMBA" run -n gudhi-audit \
  python phase8_original168_dynamic_pd_benchmark_20261118.py \
  --mode score --ttk-backend docker

PYTHONNOUSERSITE=1 "$PHASE8_MICROMAMBA" run -n gudhi-audit \
  python phase8_original168_dynamic_pd_benchmark_20261118.py --mode verify-score

PYTHONNOUSERSITE=1 "$PHASE8_MICROMAMBA" run -n gudhi-audit \
  python phase8_original168_dynamic_pd_benchmark_20261118.py --mode locality

PYTHONNOUSERSITE=1 "$PHASE8_MICROMAMBA" run -n gudhi-audit \
  python phase8_original168_dynamic_pd_benchmark_20261118.py --mode verify-locality

PYTHONNOUSERSITE=1 "$PHASE8_MICROMAMBA" run -n gudhi-audit \
  python phase8_original168_dynamic_pd_benchmark_20261118.py --mode report
```

Output (never overwrite):

```
ttk_runs_fixed/topology_finetuning/phase8_original168_eightstep_benchmark_20261118_r01/
    eval_inputs/manifest.json
    replay/manifest.json
    scores/s000/manifest.json ... scores/s167/manifest.json
    locality_samples/s000/manifest.json ... locality_samples/s167/manifest.json
    report/report.md
    report/per_field.csv
    report/summary.json
    report/manifest.json
```

Local tests cover source compilation, fixed-index/radius assertions and no-write selftest. **Real Spark TF1/TTK/GUDHI runs have not yet executed**. Treat errors as blockers; preserve failed `.staging-*` evidence rather than overwriting previous successful artifacts. Use `tmux` for long-running scoring to survive SSH disconnects. Do not append a Part CXIX success note until actual verified outputs exist.

## 4. Compare with manuscript on a consistent basis

Use *the same 168 benchmark samples, same TTK extraction, same normalization/physical units and exact same top-left 160×160 speed crop* for CNN/gradient/CV/PERS/F1 and any new dynamic model. Check archived baseline numerics for anchor fields before reporting parity. For velocity and wind-power physics metrics, avoid confusing a 160-crop speed MAE with a full-500 field speed MAE; evaluate full-field metrics in the later matched-scale runner. Compare canonical unconstrained `dB`, `W2inf`, `W22`, spatial `r4/r16/r32/r∞ W22`, GT and SR persistence coverages, per-region/per-week summaries, UV RMSE, speed MAE, WPD/PSD/threshold errors and later MT if required.

Do NOT claim that Gate A's original-168 scores are a fair replacement for the manuscript 2,688-sample / three-epoch model comparisons. An appropriately trained Gate B model is needed for that comparison. It may improve global PD while worsening location-constrained metrics; both outcomes matter. Continue Phase7 Gulf/Southwest development validation as previously frozen; keep Upper Midwest/Pacific inland untouched until final protocol and models are frozen. Pretrained PhIRE training-data exposure remains unverified, so locked-region geographic claims need that caveat.
