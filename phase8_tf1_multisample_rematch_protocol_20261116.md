# Phase 8 — Four-sample / two-pass dynamic TTK matching mini-pilot

**Status:** CODE PREPARED; NOT RUN ON DGX SPARK. Date: November 16, 2026.

## Aim and hypothesis

The preceding one-step experiment reduced true W₂,₂ on training sample 0, but used a single frozen matching. The new mini-pilot asks whether a small sequence of updates with **newly computed predicted TTK diagrams, vertex IDs, and GUDHI matchings before *every* dynamic optimizer step** can run reliably and affect actual topology errors, relative to a matched non-topological control.

The result is a training diagnostic, **not** test-set or cross-sample generalization evidence.

## Frozen experimental design

- **Dataset:** PhIRE `example_data_topology_expanded_2688/wind_MR-HR.tfrecord`. Four unique predetermined TRAINING indices `(0, 1, 2, 3)`; no access to `example_data_fixed` or to the 168-field benchmark. Normalize using pretrained PhIRE `mu=(0.7684,-0.4575)` and `sigma=(5.02455,5.9017)`. Compute scalar wind speed in physical units, cropping the top-left 160×160.
- **Initial weights:** Same frozen `models/wind_mr-hr/trained_cnn/cnn` checkpoint for both arms; identical sample order; separate TensorFlow sessions and Adam optimizer states. The original model checkpoint is never saved to or overwritten.
- **Schedule:** 2 passes × 4 samples = 8 optimizer steps *per arm*. Each step uses batch size 1 on the same index for both arms, Adam LR `1e-6`, with no shuffled order. One can set `--rounds 1` for a shorter diagnostic but the default frozen protocol is 2.
- **Baseline:** `L_uv + 0.05 L_grad` (the original normalized vector loss plus gradient-magnitude loss on denormalized speed).
- **Dynamic:** `L_uv + 0.05 L_grad + 1e-5 J_PD`, where `J_PD` is the current, **unaveraged squared** diagram W₂,₂ matching cost, with Euclidean internal norm (`internal_p=2`), Wasserstein order 2, H0/H1 combined in quadrature. The `1e-5` coefficient is diagnostic and must not be treated as tuned or optimal.
- **Critical-vertex gradient:** The surrogate gathers current SR wind-speed values at the predicted TTK birth/death critical vertices. Matched pairs target GT birth/death values; unmatched predicted pairs use Euclidean diagonal cost `(death-birth)^2/2`; unmatched GT pairs contribute a constant with no gradient to the predicted field. No claim is made that TTK or GUDHI matching is differentiable.
- **Pair selection:** All positive-persistence finite H0/H1 points from the canonical Phase 6 decoder, with no top-64 truncation or positive persistence threshold. If any GT+predicted per-dimension diagram exceeds the conservative pair cap or critical-vertex mapping is ambiguous, the experiment stops rather than pruning/relabeling silently.
- **Matching refresh:** Recompute new predicted VTU diagrams and GUDHI assignments for **every** dynamic update. GT VTUs are cached once per index. Recompute *true* canonical W₂,₂ on both arms' post-update scalar fields at each step, via independent TTK extraction. A decrease in the frozen surrogate alone is not considered a success.
- **Evaluation:** Per-step true W₂,₂ on the current training field for both arms, plus `L_uv`, `L_grad`, gradients norms, and external worker times. Results do not describe a held-out benchmark.

## Source dependencies

- `phase8_tf1_dynamic_single_batch_gate_20261116_v2.py` (the verified successful TF gather bridge)
- `phase8_tf1_one_step_optimizer_gate_20261116.py` (the verified two-independent-Adam graph)
- `phase8_matching_vertex_gradient_gate_20261116.py` (validated raw TTK critical-vertex mapping)
- `phase8_ttk_w22_timing_pilot_20261116.py` (canonical C-order VTI writer and Docker TTK command)
- `phase6_lxxvii_update/scripts/phase6_pilot_pd_gudhi_gate.py`
- `scripts/extract_ttk_pd_critical_pairs.py`
- Frozen canonical runtime parser in `~/phire_runtime_audit_20260809_221548/recompute_pd/canonical_pd_pilot.py`
- PhIRE generator and data parser (`sr_network.py`, `PhIREGANs.py`)

The new script hashes its own source and upstream source files and records/checks the original checkpoint members. It does not modify those sources.

## Execute on DGX Spark

Transfer `phase8_tf1_multisample_rematch_pilot_20261116.py` into the repository root, `~/PhIRE`. Do **not** change the published Candidate F runner. Run the following commands **one at a time, stopping at any error**:

```bash
cd ~/PhIRE

# Native Python: numerical and source checks; no writes
python3 phase8_tf1_multisample_rematch_pilot_20261116.py --mode selftest
python3 phase8_tf1_multisample_rematch_pilot_20261116.py --mode preflight

# Native TensorFlow/PhIRE Python: extract only 4 normalized training samples
python3 phase8_tf1_multisample_rematch_pilot_20261116.py --mode export

# Separate GUDHI environment: compute/cache GT TTK diagrams once
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase8_tf1_multisample_rematch_pilot_20261116.py \
  --mode prepare --ttk-backend docker

# Native TensorFlow/PhIRE Python: two independent in-memory Adam sequences.
# It launches GUDHI/TTK workers using micromamba for every dynamic rematch
# and for post-update true-W22 checks. No model checkpoint is written.
python3 phase8_tf1_multisample_rematch_pilot_20261116.py \
  --mode train --rounds 2 --learning-rate 1e-6 \
  --lambda-dynamic 1e-5 --ttk-backend docker

# Independent read-only recomputation/manifest validation using GUDHI
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase8_tf1_multisample_rematch_pilot_20261116.py --mode verify
```

### Output and failure handling

The destination is:

`ttk_runs_fixed/topology_finetuning/phase8_multisample_rematch_20261116_r01/`

An `inputs/` stage contains normalized LR/HR fields and physical GT crops; `gt_cache/` contains the four canonical GT VTUs and their PDs; `train/` is published *only on successful completion* of the entire mini-pilot, with per-step before/after fields, VTI/VTU diagrams, fresh matching feeds, loss values, per-step controls and SHA256 manifests.

The script refuses to overwrite any completed stage. If an error occurs, the incomplete stage is retained under a distinct hidden `.stage.staging-*` directory with `FAILED.json`. This is diagnostic evidence, not a completed result and not a resumable checkpoint. Do not delete or repurpose historical artifacts automatically.

If the same experiment is rerun intentionally, choose a different new `--out-dir`; do not overwrite the original.

## Interpretation gates

1. All eight updates use new predicted diagram extraction and new GUDHI matching, with a feed input hash in each step record.
2. The TensorFlow frozen-matching `J` before each update agrees with the independently computed true W₂,₂ **squared**, within the established FP32 tolerance.
3. Both arms begin with identical parameter hashes, take the same eight sample indices in the same order, and retain separate Adam state.
4. The pretraining checkpoint member hashes are unchanged.
5. All per-step output diagram values and matching feeds are finite; no silent pruning or ambiguous vertex mapping.
6. The final read-only `verify` stage re-decodes archived VTUs and recomputes the actual W₂,₂ for the dynamic prediction before each step, and for both arms after each step.
7. Improvements and regressions are both recorded. Differences are descriptive on these four training fields; no significance claim is warranted.

## Audit policy

Do not append a new successful-results section to the master notes before the Spark `--mode verify` gate passes. On success, add a dated **Part CXVI** entry recording the actual eight-step results and timings, source/checkpoint hashes, exact commands, runtime version, known limitations, and any failures/retries. On failure, record the failure and recovery with clear separation between incomplete and verified evidence. Preserve Part CXV and all earlier entries without rewriting history.

## Implementation checks completed locally (not Spark)

- Python compilation and `--mode selftest` passed.
- Atomic stage creation, artifact hashing, corrupt-path rejection primitives, and preservation of explicitly failed staging artifacts passed local synthetic tests.
- Full native TensorFlow / TTK / GUDHI model execution **has not** been run here. This protocol must not be characterized as an executed mini-pilot until you provide Spark results.
