# Part CXV — Phase 8: native TensorFlow dynamic-PD bridge and one-step optimizer comparison (November 16, 2026)

**Evidence status:** DGX Spark console logs supplied November 16, 2026. The native TensorFlow frozen-matching bridge and a controlled one-step comparison both passed their independent verification stages. **Scope:** *training* sample index 0 (from the 2,688-field training dataset); physical wind speed on the top-left 160×160 crop; canonical TTK/GUDHI persistence diagrams combining finite H0/H1 with Euclidean-ground W₂,₂. No 168-field benchmark or held-out fields used. No checkpoint was saved or modified.

## CXV.1 — Bridge v1 failure and v2 repair

The original standalone `phase8_tf1_dynamic_single_batch_gate_20261116.py` successfully restored the pretrained CNN and produced sample-0 forward predictions, but its downstream matching/verification stages failed: the forward manifest recorded an absolute `.forward.staging-*` path that no longer existed after atomic rename. The issue was the stage manifest, not TTK/GUDHI or the generator.

The versioned correction `phase8_tf1_dynamic_single_batch_gate_20261116_v2.py` records and validates paths at their final destination, preserving the original v1 output. The corrected script passed `selftest`, `preflight`, `forward`, `match`, `tfcheck`, and `verify` on Spark. Results and independent checks:

- Forward: normalized `L_uv = 0.013026411`; elapsed 0.783 s.
- TTK extraction: ground truth 0.521 s; CNN prediction 0.355 s.
- Canonical H0/H1 `W22 = 20.6730864`; matching-derived `J = W22² = 427.376501` with 1,187 predicted diagram pairs.
- TensorFlow gathered-vertex loss `J_TF = 427.376465` (absolute squared-cost discrepancy approximately `3.6e-5`); maximum TF-versus-independent field-gradient discrepancy `4.77e-7`.
- Generator-parameter gradient global norm `12975.8` (nonzero). The bridge used **no optimizer** and saved no trained weights.
- Final verification passed source/checkpoint hashes and the independent canonical matching and gradient checks.

**Interpretation:** prediction-dependent TTK vertex indices and GUDHI match targets can drive a numerically consistent **fixed-matching surrogate** gradient through the native PhIRE TensorFlow graph. This is not differentiation through TTK/GUDHI itself and does not yet imply successful training.

## CXV.2 — One-step independent optimizer comparison (verified)

Script `phase8_tf1_one_step_optimizer_gate_20261116.py`, outputs at `ttk_runs_fixed/topology_finetuning/phase8_one_step_optimizer_20261116/`. The `selftest`, `preflight`, `step`, `compare`, and `verify` stages all passed on Spark.

Both arms started from exactly the same immutable pretrained generator checkpoint and had separately initialized Adam optimizer state. Each received **one in-memory update** on the same training sample, with identical learning rate `1e-6` and background loss. The two objectives were:

- **Control:** normalized `L_uv + 0.05 L_grad`.
- **Dynamic-PD arm:** `L_uv + 0.05 L_grad + 1e-5 J_PD`, where `J_PD = W22²`, computed from the **pre-update** TTK critical vertices and GUDHI matching (held fixed for this one step).

The coefficient `1e-5` was a **diagnostic choice**, not a tuned or validated full-training hyperparameter. No truncation to the top-64 pairs or pair-count normalization was applied to the full canonical matched squared cost.

| Quantity | Original frozen CNN | Control after 1 step | Dynamic-PD after 1 step |
|---|---:|---:|---:|
| True independently recomputed H0/H1 `W22` | 20.6730864 | 20.6505469 | **20.6020992** |
| Decrease from original | — | 0.0225395 | **0.0709872** |
| Initial frozen-matching `J` | 427.376465 | 427.376465 | 427.376465 |
| Post-step frozen-matching `J` | — | 426.475952 | **424.499146** |
| Maximum absolute scalar-speed change (m/s) | — | 0.01812 | 0.03227 |

After the optimizer step, TTK extracted new persistence diagrams from **both** predicted fields (TTK durations 0.377 s and 0.356 s); GUDHI recomputed the actual diagram matching and canonical W₂,₂, and `verify` recomputed these values from immutable VTU artifacts. The comparison gate was explicitly **not** conditioned on either arm improving.

Observed on this one training sample, the dynamic arm reduced true W₂,₂ by 0.0484477 more than the control after one step (approximately 0.235% of the control's post-step W₂,₂). The dynamic arm's decrease from the original was approximately 3.15× the control's absolute decrease, but both changes are **single-sample, single-step observations**. This is positive local feasibility evidence, not evidence of statistically meaningful, held-out, or full-run gains.

## CXV.3 — Caveats and next experimental gate

1. Matching and critical vertex indices were recomputed **after** the step for evaluation, but **not** between multiple optimizer steps. A fully dynamically rematched training loop is still unimplemented.
2. This is one training field only; repeat on multiple preselected training fields with heterogeneous topology before drawing broad conclusions. Preserve the 168-field evaluation benchmark for later reporting and avoid selecting hyperparameters by repeated tuning on that benchmark.
3. Track **both** normalized reconstruction loss and physically meaningful field/topological metrics. A smaller diagram distance alone need not imply improved wind-field fidelity or spatial correspondence; ordinary PD matching omits geographic locations.
4. The raw full-diagram squared cost is scale-dependent (1,187 predicted pairs in the initial matching). Measure gradient norms and loss-term ratios before choosing a multi-step coefficient; compare to control with identical optimizer and compute budgets.
5. Recompute prediction diagrams and matching for each planned update (or a declared, tested cadence), with stale-match protection; cache ground-truth diagrams on the training set. Confirm loss reductions and changes to the **true** recomputed W₂,₂ on development training fields.
6. The TensorFlow environment emitted NumPy-2.x ABI import diagnostics for optional `numexpr`/`bottleneck` dependencies. Native TensorFlow stages nevertheless completed; avoid modifying the established runtime solely to suppress those warnings.
7. Preserve run manifests with source and checkpoint hashes, exact training sample IDs, TTK/GUDHI conventions, H0/H1 and diagonal costs, timing, and output hashes. Keep every experiment in an independent, collision-protected directory; do not overwrite historical scripts or checkpoints.

**Next status:** a limited **multi-sample, multi-step, prediction-rematched training-development pilot** is *planned, not executed*. No full 2,688-field training result or held-out evaluation result is claimed here. The v2 bridge, one-step script, and verified result artifacts exist on Spark according to the supplied logs; their recorded manifest hashes remain the authoritative provenance.
