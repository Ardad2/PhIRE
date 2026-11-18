# Phase 8 TF1 single-batch bridge: staging-path correction (v2)

**Date:** November 16, 2026. **Status:** Local source-level tests pass; actual Spark v2 run not yet executed.

## Observed Spark evidence

- Pure numerical selftest and checkpoint/training-data preflight passed.
- TensorFlow `forward` restored the frozen CNN on training sample index 0, returning normalized `L_uv=0.013026411` in `0.600 s`.
- `match` failed before any new TTK extraction with `FileNotFoundError` for `.../.forward.staging-wt5bhjar/gt_crop.npy`.
- The original `forward/` output remains present; this failure is due to a stale absolute pathname in `forward/manifest.json`, not a failed model forward pass.
- TensorFlow printed NumPy-1-compiled `numexpr`/`bottleneck` compatibility warnings, but the forward stage still passed. These are not the observed fatal error and should not prompt changing the established training environment.
- Separately, master notes Part CXIV was appended successfully; its changes were not shown committed in the provided transcript.

## Root cause and fix

`atomic_stage` writes to a temporary staging directory and renames the folder after the writer returns. Version 1 recorded the full temporary paths as dictionary keys in `output_hashes`. `load_forward` / `load_match` subsequently tried to use those nonexistent staging locations.

Version 2 records stage-**relative** keys (e.g. `gt_crop.npy`, `vti/gt.vti`) while hashing the same staged bytes, and independently verifies those paths after the rename. It rejects absolute or parent-traversal paths, missing files, and changed bytes. The corrected source includes its own path in the list of hash-checked source inputs.

The versioned script has a distinct output root to preserve the v1 evidence:

- v1: `ttk_runs_fixed/topology_finetuning/phase8_tf1_single_batch_bridge_20261116/` (forward only, with stale-path manifest)
- v2: `ttk_runs_fixed/topology_finetuning/phase8_tf1_single_batch_bridge_20261116_v2/` (new, independently staged outputs)

**Do not edit or delete the v1 result or overwrite the original script.** No training step or model-weight update was added. Do not reinterpret this fix as a successfully validated dynamic loss.

## Local checks performed

- Python syntax compile: PASS.
- Unchanged mathematical selftest: PASS.
- End-to-end local `atomic_stage` rename with stage-relative hash manifest: PASS.
- Rejected stale absolute artifact paths and `../` traversal: PASS.
- Detected modified artifact bytes: PASS.

## Spark procedure

Download and copy `phase8_tf1_dynamic_single_batch_gate_20261116_v2.py` into the PhIRE repository root. Run these commands individually; **stop at the first failure**:

```bash
cd ~/PhIRE
python3 phase8_tf1_dynamic_single_batch_gate_20261116_v2.py --mode selftest
python3 phase8_tf1_dynamic_single_batch_gate_20261116_v2.py --mode preflight
python3 phase8_tf1_dynamic_single_batch_gate_20261116_v2.py --mode forward --sample-index 0
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python phase8_tf1_dynamic_single_batch_gate_20261116_v2.py --mode match --ttk-backend docker --ttk-timeout 180
python3 phase8_tf1_dynamic_single_batch_gate_20261116_v2.py --mode tfcheck
python3 phase8_tf1_dynamic_single_batch_gate_20261116_v2.py --mode verify
```

The `forward` and `tfcheck` stages require the same TensorFlow environment that was able to execute the original forward stage. The `match` stage uses the separate GUDHI environment and the existing TTK Docker image.

If v2 `match` fails with an ambiguous real-field critical-vertex correspondence, or with a large point-count guard, preserve the error log and do not bypass the check. Those would be independent scientific/numerical questions, not the path fix.

## Future audit requirement

Once v2 succeeds or identifies a new blocker, add a dated *new* notes entry recording the failure of the original v1 bridge, the staging-path root cause, the v2 script SHA-256, the v2 artifact paths and hashes, the observed TF gradient/loss results or diagnostic failure, and caveats. Avoid marking full native TF integration verified until **both `tfcheck` and `verify` pass on Spark**.
