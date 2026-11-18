# Phase 8 — Final 168-training-field model evaluation workpack (November 18, 2026)

## Status of inputs

Spark completed three arms to 504/504 updates each (22 verified checkpoint segments per arm): `grad`, `fixed`, `dynamic`. The 168 initial fixed-pair feeds are sealed. The existing original-168 eight-step benchmark has also been independently verified and reported. This new evaluation has **NOT** yet run on Spark. The script was syntactically compiled and its static self-test passed locally; it has not been runtime-tested with Spark's TensorFlow, TTK or GUDHI.

Source scripts are immutable and SHA-pinned in the new evaluator:

- `phase8_train168_threearm_dynamic_pd_20261118.py` SHA256 `6dccfd25f10a85ac09dcde6425c5de2ee6cf2bf622da9e12bec800adf8301fa2`.
- `phase8_original168_dynamic_pd_benchmark_20261118.py` SHA256 `c28b2ed444083668472f5527d53cdd4a7cd04e3c2611868062668091f64192d0`.

Existing outputs are read-only:

- `ttk_runs_fixed/topology_finetuning/phase8_train168_threearm_20261118_r01` (training checkpoints)
- `ttk_runs_fixed/topology_finetuning/phase8_original168_eightstep_benchmark_20261118_r01` (original benchmark inputs and reference scores)

Only writes to new:

- `ttk_runs_fixed/topology_finetuning/phase8_train168_final_original168_eval_20261118_r01`

No Phase 7 development regions, Upper Midwest, or Pacific inland outputs are loaded. The original 168-field Mid-Atlantic benchmark is a repeatedly used **development comparator**, not a geographically held-out test. The new 168-field training pilot used batch size 1 and three epochs; it is not a matched train-scale comparison to a 2688-field/batch-size-4 manuscript run.

## Execution: start with read-only checks

Copy `phase8_train168_final_original168_eval_20261118.py` to the PhIRE repo root without editing the frozen previous drivers.

```bash
cd ~/PhIRE
python3 phase8_train168_final_original168_eval_20261118.py --mode selftest
python3 phase8_train168_final_original168_eval_20261118.py --mode preflight
python3 phase8_train168_final_original168_eval_20261118.py --mode training-audit
```

`training-audit` reads only the three 504-step ledgers and summarizes mean training losses/gradient norm contribution for each epoch. These are not evaluation metrics.

## Export four final-state predictions

```bash
cd ~/PhIRE
set -o pipefail
mkdir -p logs
python3 phase8_train168_final_original168_eval_20261118.py --mode predict \
  2>&1 | tee logs/phase8_train168_final_eval_predict_20261118.log
```

This restores the complete final TF1 graph including stored model weights/Adam slots for each arm, but runs **zero optimizer updates**. It also predicts pretrained CNN using original weights, and checks every CNN crop against the archived original-benchmark initial CNN output (absolute tolerance <0.005 m/s). One atomic directory per arm; resume with the same `--mode predict` if interrupted. If this regression check fails, stop and diagnose, do not relax the tolerance without reviewing provenance.

## Three-field numerical and locality smoke test

```bash
export PHASE8_MICROMAMBA="${MAMBA_EXE:-/home/adadhwal/Desktop/bin/micromamba}"
PYTHONNOUSERSITE=1 "$PHASE8_MICROMAMBA" run -n gudhi-audit \
  python phase8_train168_final_original168_eval_20261118.py \
  --mode score --ttk-backend docker --max-new-samples 3
PYTHONNOUSERSITE=1 "$PHASE8_MICROMAMBA" run -n gudhi-audit \
  python phase8_train168_final_original168_eval_20261118.py \
  --mode locality --max-new-samples 3
```

The score stage re-extracts GT, CNN, gradient-control, fixed-pair, and dynamic-pair diagrams from the original 160x160 speed crop and evaluates numerical GUDHI distances. CNN's numerical scores are cross-checked against the existing 168-field benchmark. The locality stage uses unchanged SHA-pinned Phase5E `SparsePrepared` for radii 4, 8, 16, 32, 64, 96, 128, infinity. It reports both GT and SR persistence coverage, GT-to-diagonal and diagonal-to-SR counts. Partial stages are resumable without replacing sealed outputs.

## Full benchmark

Only after the three-field smoke test passes:

```bash
set -euo pipefail
PYTHONNOUSERSITE=1 "$PHASE8_MICROMAMBA" run -n gudhi-audit \
  python phase8_train168_final_original168_eval_20261118.py --mode score --ttk-backend docker \
  2>&1 | tee logs/phase8_train168_final_eval_score_20261118.log
PYTHONNOUSERSITE=1 "$PHASE8_MICROMAMBA" run -n gudhi-audit \
  python phase8_train168_final_original168_eval_20261118.py --mode verify-score \
  2>&1 | tee logs/phase8_train168_final_eval_verify_score_20261118.log
PYTHONNOUSERSITE=1 "$PHASE8_MICROMAMBA" run -n gudhi-audit \
  python phase8_train168_final_original168_eval_20261118.py --mode locality \
  2>&1 | tee logs/phase8_train168_final_eval_locality_20261118.log
PYTHONNOUSERSITE=1 "$PHASE8_MICROMAMBA" run -n gudhi-audit \
  python phase8_train168_final_original168_eval_20261118.py --mode verify-locality \
  2>&1 | tee logs/phase8_train168_final_eval_verify_locality_20261118.log
PYTHONNOUSERSITE=1 "$PHASE8_MICROMAMBA" run -n gudhi-audit \
  python phase8_train168_final_original168_eval_20261118.py --mode report \
  2>&1 | tee logs/phase8_train168_final_eval_report_20261118.log
```

If running remotely for extended periods, start a `tmux` session first. Inspect exact PASS markers after each stage. Do not infer success solely from a shell prompt or a Python process terminating.

Expected final report:

`ttk_runs_fixed/topology_finetuning/phase8_train168_final_original168_eval_20261118_r01/report/report.md`

Accompanying `summary.json` and `per_field.csv` contain 168-field means and paired deltas. The report is atomically published **only after** independent numerical and locality verification seals exist; no report overwrites.

## Scientific comparison

- Pretrained CNN, gradient-only baseline, frozen initial-pair PD, dynamic rematched PD.
- All three fine-tuned arms: *identical pretrained starting weights*, same 168 training samples, same order, 504 Adam updates, batch size 1, three epochs.
- Compare unconstrained canonical `W22`, `W2inf`, `dB`; normalized `L_uv`,`L_grad`; speed crop MAE; spatially constrained `W22` and persistence coverage at r4/r16/r32/r∞.
- Paired deltas for fixed vs grad, dynamic vs grad, dynamic vs fixed; seven 24-hour blocks rather than invalid treating 168 correlated hours as independent significance trials.
- Fixed-pair *initial-critical-vertex* objective and dynamic rematching isolate correspondence policy, but do not automatically establish the better training outcome.
- Larger 2688-field balanced training and region/2012 development replication remain separate follow-on experiments; locked geographic tests reserved for a frozen final protocol.
