# Part CXIV — Phase 8: TTK critical-vertex surrogate gradient verified (November 16, 2026)

## CXIV.1 Evidence and scope

**Evidence:** Actual DGX Spark stdout shared November 16, 2026 for `phase8_matching_vertex_gradient_gate_20261116.py`, together with the independent prior Phase 8 field/matching gate. This entry is **observed and verified**, not a prediction of the planned TensorFlow experiment. All results used the existing 160×160 synthetic-field TTK VTUs, canonical H0/H1 decoder, Euclidean-ground GUDHI W₂,₂, and the corrected C-order point-to-vertex mapping. The held-out 168-field benchmark and network checkpoints were not accessed for training.

The script's `selftest`, `preflight`, `benchmark` and `verify` stages all reported PASS. Synthetic selftest included matched predicted pairs, predicted-to-diagonal, GT-to-diagonal constant cost, gradients accumulating at shared vertices, and fixed-assignment finite differences with maximum absolute discrepancy **2.11e-12**. No TF1 graph or optimizer ran.

## CXIV.2 Per-field verified values

| Predicted field | Canonical W₂,₂ | Local squared W₂,₂ surrogate J | Maximum absolute scalar-field gradient | Nonzero gradient entries |
|---|---:|---:|---:|---:|
| Identity | 4.76837158e-07 | 2.27373675e-13 | 9.53674e-07 | 1 |
| Spurious peak | 6.34677624 | 40.2815687 | 8.9757 | 5 |
| Displaced peak | 0.126036278 | 0.0158851433 | 0.17824 | 6 |

The identity's tiny positive distance and gradient are numerical roundoff, not a substantive topological disagreement. These values are for the **unweighted sum of squared Euclidean-ground W₂,₂ costs** across H0 and H1, not the fixed-CV/PERS proxy scales.

**Recorded outputs:** `ttk_runs_fixed/topology_finetuning/phase8_matching_vertex_gradient_gate_20261116/result.json`. Final `--mode verify` reports that upstream hashes, canonical W₂,₂, predicted TTK vertex correspondences, and fixed-assignment finite differences agree. The console transcript does not expose a full recorded manifest or its cryptographic hashes; do not invent them.

## CXIV.3 Valid conclusions and limitations

- Recomputed diagram matching can be represented as a local scalar loss with gradients at **prediction-dependent TTK critical vertex IDs**; the synthetic spurious feature gives a clear nonzero direction that fixed GT anchors do not provide.
- The derivative verified is conditional on **fixed current critical-vertex IDs and matching assignment**. This does not differentiate through TTK or the discrete matching optimization, and a future model step must recompute them on updated predictions.
- An unmatched *predicted* pair is penalized toward the diagonal with squared cost `(d-b)^2/2`. An unmatched *ground-truth* pair has its standard diagonal cost but contributes **zero gradient to the predicted field under frozen matching**; any extra feature-recreation/GT-location constraint is a separately defined training term.
- Real wind fields may exhibit diagram coordinate ties or ambiguous correspondence to TTK vertex pairs. The current mapping intentionally fails closed; passing simple synthetic fields does not resolve general critical-pair identity and H0/H1 metadata problems.
- Nothing here establishes end-to-end TF1 generator backpropagation, successful optimization, improved prediction metrics, generalization, or spatial alignment by a diagram-only loss.

## CXIV.4 Next milestone — planned, not yet executed

Use a **single training example from the 2,688-field training TFRecord**, restore the frozen pretrained CNN checkpoint using the same `SR_NETWORK` graph as Candidate F, compute the current predicted physical-speed 160×160 crop, extract its TTK diagram, and match to the corresponding GT diagram with the validated GUDHI convention. Feed prediction-derived vertex indices and matching targets to `tf.gather_nd`. The first TF1 gate must compare the graph's squared matching loss and speed-field gradients against the validated NumPy surrogate and demonstrate nonzero gradient flow to generator parameters. **Do not instantiate an optimizer or update checkpoint weights** in this gate. Run it in a fresh isolated artifact directory and retain full provenance.

This is a plan, not an observed result. Preserve all previous Parts and historical checkpoints unchanged.
