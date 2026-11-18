# Phase 8 — Single-sample native-TensorFlow dynamic TTK→GUDHI→gradient bridge

**Prepared:** November 16, 2026. **Status:** Script syntax and pure NumPy selftest pass locally. **Not executed on Spark**. All proposed scientific results remain unverified until Spark runs.

## 1. Goal and scope

The previous Phase 8 tests verified TTK extraction, GUDHI Euclidean-ground H0/H1 W₂,₂ assignment, unmatched-to-diagonal handling, unique predicted TTK critical vertices for a simple synthetic field, and finite differences of a local *frozen-assignment* squared-distance surrogate. This stage closes the TensorFlow bridge on **one actual training example**.

The experiment is intentionally **not training**: no optimizer is instantiated, no `train_op` runs, no parameter is assigned, no checkpoint is saved, and the 168-sample held-out benchmark is not read. We instantiate the same `SR_NETWORK` generator/pretraining graph, restore the frozen pretrained CNN, and use TensorFlow's `tf.gather_nd` with dynamically matched vertex positions. We measure only gradient flow.

**Data:** `example_data_topology_expanded_2688/wind_MR-HR.tfrecord`, embedded sample index 0 by default. Generator checkpoint: `models/wind_mr-hr/trained_cnn/cnn`. Mean/std: `mu=[0.7684,-0.4575]`, `sigma=[5.02455,5.9017]`. Crop: top-left 160×160 physical scalar wind speed. TTK command: `ttkPersistenceDiagramCmd -t 1 -i ... -a wind_speed -o ...` through the existing `phire-ttk:latest` Docker image.

**Reference metric:** canonical Phase 6 H0 and H1 finite positive-persistence pairs; `gudhi.wasserstein_distance(order=2., internal_p=2., matching=True, keep_essential_parts=False)` applied separately to H0/H1, squared costs summed and square-rooted to form final W₂,₂. Do not substitute internal_p=∞ or the 64-pair fixed-index subset. No arbitrary threshold or top-K truncation.

## 2. Why two Python environments

The standard PhIRE TensorFlow 1-compat environment may differ from the separate `gudhi-audit` micromamba environment. Therefore, the standalone script has three incremental stages, plus local checks:

- `forward`: **PhIRE TensorFlow environment** — load precisely one training TFRecord example by its `index`, construct/restore `SR_NETWORK`, produce the current CNN predicted speed crop and its GT crop, write only to a new `forward/` stage.
- `match`: **gudhi-audit environment** — write fresh C-order VTIs, run actual TTK on both crops, decode H0/H1 with the canonical Phase 6 reader, uniquely recover predicted critical vertex IDs, run GUDHI matchings, save positional target feeds plus full squared-loss and analytical field-gradient reference.
- `tfcheck`: **PhIRE TensorFlow environment** — reconstruct the same generator and restore the original checkpoint; feed predicted birth/death `[batch, pairs, (y,x)]` positions and matching targets to the `tf.gather_nd` graph; check graph loss and field gradients against the independently computed NumPy values; check nonzero finite gradients to generator variables; never update variables.

The output root is new and unique: `ttk_runs_fixed/topology_finetuning/phase8_tf1_single_batch_bridge_20261116/`. Each stage is written atomically and refuses an already existing destination. It is not a permanent training entrypoint.

## 3. Execute on DGX Spark

Transfer `phase8_tf1_dynamic_single_batch_gate_20261116.py` to the root of `~/PhIRE` alongside the three previous Phase 8 scripts. Do not modify the original released Candidate F runner or the cached training arrays.

From the repository root:

```bash
cd ~/PhIRE

# A. Pure NumPy selftest (any Python with NumPy)
python3 phase8_tf1_dynamic_single_batch_gate_20261116.py --mode selftest

# B. Read-only source, checkpoint and training-split preflight
python3 phase8_tf1_dynamic_single_batch_gate_20261116.py --mode preflight

# C. Pretrained CNN forward only (use the same Python that runs native PhIRE TF1)
python3 -c 'import tensorflow.compat.v1 as tf; print("PhIRE TF", tf.__version__)'
python3 phase8_tf1_dynamic_single_batch_gate_20261116.py --mode forward --sample-index 0

# D. Actual TTK / full canonical H0/H1 matching (use GUDHI environment)
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase8_tf1_dynamic_single_batch_gate_20261116.py \
  --mode match --ttk-backend docker --ttk-timeout 180

# E. Rebuild native TensorFlow CNN and validate nonzero parameter gradients
python3 phase8_tf1_dynamic_single_batch_gate_20261116.py --mode tfcheck

# F. Read-only manifest and hash verification (NumPy only)
python3 phase8_tf1_dynamic_single_batch_gate_20261116.py --mode verify
```

If TensorFlow is unavailable to the default `python3`, run stages C and E using whichever Python environment you normally use for Candidate F. Do **not** automatically install packages in the audit environment or modify the historical training stack. Confirm with your own environment before proceeding.

If you see `ambiguous critical-vertex pair` or `cannot be mapped to any TTK critical-vertex pair` during D, **stop** and report the full error. That identifies a real numerical correspondence problem with scalar-value-based pair alignment on actual fields. Do not arbitrarily select one tied TTK pair or discard any diagram points to force passage. A stronger identifier-based correspondence method may be needed before training.

If the matching count exceeds the default safety cap, similarly stop and inspect the per-dimension sizes; do not silently truncate the full evaluation distance. The previous training sample 0 took ~0.37–0.75 seconds for individual TTK calls, but this is not a runtime guarantee for a new TF forward field.

## 4. Mathematical quantity and precision checks

For each matched GT/predicted pair `q=(b_g,d_g)` and `p=(b_p,d_p)` use `(b_p-b_g)^2 + (d_p-d_g)^2`. For unmatched predicted `p`, use `(d_p-b_p)^2/2`. For unmatched GT `q`, include `(d_g-b_g)^2/2` as a **constant** (no gradient with respect to predictions). Sum across both H0 and H1, with no per-pair average. The graph loss is therefore locally consistent with `W22**2` under the frozen matching/critical-vertex set.

The TF1 graph's `tf.gradients(J, speed_crop)` is compared to the independent predicted-vertex analytical NumPy gradient. We also compute `tf.gradients(J, model.g_variables)` and require a finite, nonzero global norm. This does **not** prove that a gradient step improves W₂,₂, and is **not** differentiation through TTK or GUDHI. Discrete diagram extraction/matching must be redone after weight changes during an actual training loop.

For the smoke gate we permit small float32/VTI serialization tolerances for forward reconstruction and matching costs. Any failure requires diagnosis, not a relaxed tolerance by default.

## 5. Audit and interpretation

Upon a verified Spark PASS, append a **new** dated master-notes section documenting actual commands, environment, source hash, model checkpoint hash, TF normalized reconstruction loss, full canonical W₂,₂, J, field gradient discrepancy, parameter gradient norm, timings, pass/fail messages, and unresolved caveats. Until then, Part CXIV's TF gate is marked planned.

Do not commit generated checkpoints or stage files simply because tests pass. They are deliberately isolated measurements, and no weights should change. Your previous Part CXIII audit records timing/field gates; Part CXIV records the newly verified synthetic critical-vertex gradient gate.
