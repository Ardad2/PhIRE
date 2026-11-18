# Phase 8 — Matching-to-vertex and local-gradient gate

**Status (before Spark execution):** source prepared and Python syntax + pure NumPy analytical/finite-difference selftest checked locally. No actual TTK/GT/SR endpoint mapping or TF1 gradient training has been executed by this gate.

## Motivation

The accepted synthetic-field gate shows that recomputed TTK diagrams detect a spurious predicted peak that is invisible at the fixed GT anchor vertices. However, this is a *numerical distance*, not a training loss with gradients flowing into the generated field. The next research risk is matching a canonical diagram point, indexed by GUDHI, to the birth/death **vertex IDs of the predicted TTK diagram**. Only then can the model use differentiable `tf.gather_nd` to supervise those locations.

## Scope

`phase8_matching_vertex_gradient_gate_20261116.py` reopens the existing four synthetic-field `.npy` and `.vtu` files, checks their hashes against the accepted synthetic manifest, decodes the canonical H0/H1 persistence coordinates, parses TTK critical pairs from port-0 VTUs, and maps canonical diagram coordinates to predicted birth/death vertex IDs. It requires the corrected `field[y, x] -> pointId = x + y*160` convention. It fails closed if a canonical point cannot be mapped or competing raw TTK critical pairs have indistinguishable coordinates.

For each predicted field and each H0/H1 matching, it computes:

- **Predicted-to-GT feature** `(b_p,d_p) -> (b_g,d_g)`: `(b_p-b_g)^2 + (d_p-d_g)^2`.
- **Predicted-to-diagonal feature:** `(d_p-b_p)^2/2`, with midpoint target `((b_p+d_p)/2, (b_p+d_p)/2)` when later fed into TensorFlow.
- **GT-to-diagonal feature:** `(d_g-b_g)^2/2` as a constant. This contributes to the reported distance, **not** the gradient with respect to predicted field values under the fixed matching.

It checks that their sum equals the numerical canonical `W22²` and verifies analytic gradients at every involved predicted critical vertex against central finite differences, while holding the *matching and TTK vertex locations fixed*. This is the correct local surrogate derivative, **not** a derivative through the discrete TTK or GUDHI solver.

## DGX Spark instructions

Transfer the new Python file into `~/PhIRE` alongside the two previously accepted Phase 8 scripts. Do **not** rerun the earlier immutable benchmark jobs.

```bash
cd ~/PhIRE

# Pure NumPy gradient math only; no model or TTK execution
python3 phase8_matching_vertex_gradient_gate_20261116.py --mode selftest

# Validate synthetic artifacts, canonical parser, and matching environment
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase8_matching_vertex_gradient_gate_20261116.py --mode preflight

# Run prediction-dependent critical-vertex correspondence + gradient checks
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase8_matching_vertex_gradient_gate_20261116.py --mode benchmark

# Independently recompute and check the stored output + source hashes
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase8_matching_vertex_gradient_gate_20261116.py --mode verify
```

If correspondence is ambiguous or raw TTK pairs do not match the canonical reader, **preserve the error output and stop**. Do not loosen matching tolerances or fabricate vertex IDs to get a green gate. We can inspect the `ttkVertexScalarField`, `PairIdentifier`, `PairType`, `Birth`, and `Persistence` arrays and implement an identifier-preserving mapper instead. No changes to the original Candidate F code or existing result directories are required.

Successful output is created once in:

`ttk_runs_fixed/topology_finetuning/phase8_matching_vertex_gradient_gate_20261116/result.json`

The script refuses to overwrite the output directory.

## After a successful gate

The next *separate* implementation gate will use the current-model TF1 forward pass, TTK diagram reconstruction, GUDHI matchings, prediction-critical-vertex placeholders and `tf.gather_nd` to verify gradient flow to `model.x_SR` and to trainable generator weights on **one training batch**. Match-to-GT and predicted-to-diagonal losses must use the same explicit **sum-of-squares / W22²** scaling, with logged normalization and weights. Unmatched GT features must be addressed explicitly, not silently attributed to exact W₂,₂ gradient. Only then benchmark training step throughput and plan the number of epochs and batch size.
