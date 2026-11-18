# Phase 8 — Three-arm 168-field training experiment (November 18, 2026)

## Scope and objectives

This is an **expanded-training-pool method-development pilot**, not the final manuscript-scale comparison and not a geographic held-out test. The original Mid-Atlantic 168-field benchmark remains evaluation-only. The two geographically locked Phase 7 regions — Upper Midwest and Pacific inland — must remain unexamined.

A new deterministic selection chooses **168 distinct hourly WTK training records** from `example_data_topology_expanded_2688/wind_MR-HR.tfrecord` and its `manifest.csv`. It spans **16 independent source window identities** (4 source windows per season), **42 selected records in each season**, and excludes the four indices `0,1,2,3` used in the earlier eight-step pilot. It is not a claim of 168 statistically independent weather episodes. The WTK provenance mapping is verified at preflight.

Three *separate* TensorFlow 1 sessions, each begun from the original frozen `models/wind_mr-hr/trained_cnn/cnn` generator weights with a newly initialized Adam state, optimize the **same index sequence** for three epochs. Each arm takes 504 single-example updates at learning rate `1e-6`, with a common normalized base loss `L_uv + 0.05 * L_grad`.

| Arm | Objective and correspondence policy |
|---|---|
| `grad` | Base loss only |
| `fixed` | Base + `1e-5 * W22²` local surrogate, using the **initial pretrained CNN's TTK critical vertices and GUDHI assignment**, frozen for all epochs |
| `dynamic` | Base + `1e-5 * W22²` local surrogate, with **fresh predicted TTK critical vertices and fresh GUDHI assignment at every training step** |

The same original Phase 8 TensorFlow loss graph and original TTK/GUDHI correspondence/gradient worker are reused; the new runner changes the example inventory, temporal sample order, staged state persistence, and fixed/dynamic scheduling, not the underlying matching solver. There is no spatial radius constraint in either PD training term; spatial locality is an **evaluation-only measure** at this stage. GT diagrams are computed once on the top-left 160×160 denormalized scalar-speed crop.

This is intentionally **batch size 1**, because the verified original dynamic-matching gradient implementation is single-batch. Accordingly, it is not yet training-protocol-equivalent to the manuscript's batch-size-4, 2,688-example, 3-epoch runs. The next workpack will evaluate all new models on exactly the original 168 benchmark fields with the same loss, canonical PD and Phase 5E locality scoring protocol as the recently verified eight-step benchmark.

## Installation

Copy `phase8_train168_threearm_dynamic_pd_20261118.py` to the **top level of `~/PhIRE`**. Do not edit the old Phase 8 scripts, old benchmark files, master research notes, original model checkpoints, or any Phase 7 regional assets. The output destination is **new** and immutable:

`ttk_runs_fixed/topology_finetuning/phase8_train168_threearm_20261118_r01`

The script refuses existing `plan`, exported input, or GT cache destinations. It refuses to overwrite any completed fixed-pair or training segment. An interrupted segment's hidden `.staging-*` folder remains for forensic inspection, while resumption uses the last **sealed** checkpoint.

### 1. Read-only gates and immutable plan

```bash
cd ~/PhIRE
python3 phase8_train168_threearm_dynamic_pd_20261118.py --mode selftest
python3 phase8_train168_threearm_dynamic_pd_20261118.py --mode preflight
python3 phase8_train168_threearm_dynamic_pd_20261118.py --mode plan
```

The plan pins original pretrained checkpoint hashes, source-script hashes, the expanded training TFRecord bytes and 2,688-row `manifest.csv`, and the existing development-benchmark report. `--mode plan` reads no geographic locked datasets and performs no updates.

### 2. Export exactly 168 train-pool inputs (native TensorFlow)

```bash
python3 phase8_train168_threearm_dynamic_pd_20261118.py --mode export
```

This reads the **entire 2,688-record expanded training TFRecord**, proves selected records have the expected record IDs and 500×500 / 100×100 shapes, and stores only the selected 168 records and 160×160 speed crops. This is *not* the benchmark's `example_data_fixed` file. On the previously configured system, optional NumPy-2 ABI tracebacks for `numexpr`/`bottleneck` appeared but were nonfatal when TensorFlow continued to print successful stage markers; do not change Python dependencies unless a stage actually fails.

### 3. Prepare 168 GT diagrams (GUDHI audit environment)

```bash
export PHASE8_MICROMAMBA="${MAMBA_EXE:-/home/adadhwal/Desktop/bin/micromamba}"
mkdir -p logs
set -o pipefail

PYTHONNOUSERSITE=1 "$PHASE8_MICROMAMBA" run -n gudhi-audit \
  python phase8_train168_threearm_dynamic_pd_20261118.py \
  --mode prepare --ttk-backend docker \
  2>&1 | tee logs/phase8_train168_prepare_20261118.log
```

The GT cache is atomically published only after all 168 diagrams are finished; it cannot be restarted from an unfinished GT-cache stage without reviewing the preserved `.staging-*` directory. This stage does not use the GPU training optimizer.

### 4. Two-sample fixed-pair cache and **two-update smoke test per arm**

```bash
# Native TensorFlow parent starts the TTK/GUDHI child worker by micromamba.
python3 phase8_train168_threearm_dynamic_pd_20261118.py \
  --mode fixed-cache --max-new-samples 2 --ttk-backend docker

python3 phase8_train168_threearm_dynamic_pd_20261118.py \
  --mode train --arm grad --max-new-steps 2

python3 phase8_train168_threearm_dynamic_pd_20261118.py \
  --mode train --arm fixed --max-new-steps 2

python3 phase8_train168_threearm_dynamic_pd_20261118.py \
  --mode train --arm dynamic --max-new-steps 2 --ttk-backend docker

python3 phase8_train168_threearm_dynamic_pd_20261118.py \
  --mode verify --arm all
```

The first two steps of each arm must report `[SEGMENT] PASS`, with `2/504` updates sealed. The fixed cache computes the *pretrained initial* correspondence for exactly those two selected train-pool fields. Dynamic matching must report `[MATCH-WORKER]` / `[WORKER] PASS` for **every** dynamic training update. The worker independently compares freshly matched canonical squared `W22` with its differentiable TensorFlow surrogate before applying that update. The smoke tests create *actual* optimizer updates but only in new self-contained checkpoints, never in the original pretrained weights.

**Do not start the three full runs until this smoke test has passed and the checkpoint/gradient logs have been reviewed.** Training with `--max-new-steps 2` creates real stateful segments that will be reused during the later continuation, not discarded or repeated.

### 5. Full continuation (only after review)

```bash
# Complete the initial-pretrained fixed correspondence cache on all 168 records.
python3 phase8_train168_threearm_dynamic_pd_20261118.py \
  --mode fixed-cache --ttk-backend docker

# Extend each pilot arm from step 2 to 504 without resetting Adam.
python3 phase8_train168_threearm_dynamic_pd_20261118.py --mode train --arm grad
python3 phase8_train168_threearm_dynamic_pd_20261118.py --mode train --arm fixed
python3 phase8_train168_threearm_dynamic_pd_20261118.py --mode train --arm dynamic \
  --ttk-backend docker

# Verify all chained segment hashes and optimizer checkpoints.
python3 phase8_train168_threearm_dynamic_pd_20261118.py \
  --mode verify --arm all --full-data-hash
```

Training checkpoints are saved **every 24 steps** as immutable segments, including **all TF1 optimizer slot variables and model weights**. Thus a new call to `train` resumes the same exact Adam states at the last sealed boundary. The initial two-update smoke segment has size 2, and the continuation follows immediately afterward. The full-run checkpoint artifacts will consume disk space. Before long training, inspect `df -h` and use `tmux` or another durable session. Do not delete any sealed or previous source artifacts.

## Audit gates / interpretation

- `[SELFTEST] PASS`: season-balanced selection and 504-step schedule (no writes).
- `[PLAN] PASS`: new immutable input/checkpoint/sources contract.
- `[EXPORT] PASS`: 168 selected train-pool normalized records, no updates.
- `[PREPARE] PASS`: 168 frozen GT speed PDs.
- `[FIXED-CACHE] PASS`: one initial pretrained CNN correspondence per selected field.
- `[TRAIN-GRAD|FIXED|DYNAMIC]`: actual optimizer update and diagnostics.
- `[SEGMENT] PASS`: model+Adam snapshot and immutable step ledger published.
- `[VERIFY-GRAD|FIXED|DYNAMIC] PASS`: full 504-step arm complete (otherwise `PARTIAL`).

For each update, save normalized `L_uv`, `L_grad`, 160-crop speed MAE, squared PD surrogate `J` (fixed/dynamic), PD gradient norm, base gradient norm, total gradient norm, and `lambda * PD_grad_norm / base_grad_norm`. **Those logged training losses are not independent evaluation metrics.** No new manuscript claim should be made until model states from these three arms are scored using the frozen 168-field evaluation protocol, with checks on contamination, temporal autocorrelation, geography, and pretraining exposure.

**Known limits:** checkpoint original pretraining exposure not independently established; training sample hours are correlated; “fixed” means the particular frozen initial vertex/correspondence policy defined above (not necessarily the same as manuscript PERS). Existing original 168 benchmark has been used for development/hyperparameter selection. Neither the additional 2012 development replication nor the Upper Midwest/Pacific inland locked geographic evaluation is part of this runner.
