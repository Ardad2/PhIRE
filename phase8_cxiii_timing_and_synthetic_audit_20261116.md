# Part CXIII — Phase 8: TTK/GUDHI runtime and synthetic-field behavioral gates (November 16, 2026)

## CXIII.1 Scope, provenance, and scientific status

**Evidence:** DGX Spark console transcripts provided on November 16, 2026, for `phase8_ttk_w22_timing_pilot_20261116.py` and `phase8_synthetic_field_gate_20261116.py`. Both are **non-training** gates. They do not update the PhIRE model, touch the 168-sample held-out benchmark, rerun the incomplete Candidate Dpd experiment, or establish improvement from a trained dynamic-PD model. Preserve Parts VIII, XIII, CXI, and CXII unchanged. This CXIII entry supersedes the earlier *draft* CXIII addendum that listed the synthetic experiment as not yet executed; that draft was not a verified GitHub commit in the supplied transcript.

Both gates use 160×160 scalar wind speed, the repaired x-fastest/C-order VTI convention, `ttkPersistenceDiagramCmd` in Docker (`phire-ttk:latest`), the existing canonical Phase 6 H0/H1 TTK decoder and corrected GUDHI Euclidean-ground W₂,₂: `order=2`, `internal_p=2`, essential features excluded, zero-persistence features ignored, and H0/H1 combined by Euclidean norm. The final numerical distance is **not** based on only the top-64 GT persistence pairs. Top-64/1% GT pairs are used solely for the *fixed-pair control* in the synthetic-field test.

## CXIII.2 Actual 2,688-training-sample timing pilot — observed

- `selftest`: PASS for hand-constructed PD identity, displaced point, extra feature, diagonal projection and C-order orientation.
- `preflight`: PASS; `data_out/wind_mrhr_cnn_expanded2688/dataGT.npy` and `dataSR.npy`, each `(2688, 500, 500, 2)`, sample ordinal 0, existing GUDHI decoder and Docker backend.
- Sample 0 GT TTK extraction: **0.748 s**; pretrained-CNN prediction: **0.367 s**.
- H0/H1 matching-only time: **0.115 s**; canonical and reconstructed `W22=20.673087`.
- End-to-end benchmark: **2.997 s**. The total includes operations other than the separately timed TTK and matching calls.
- Result: `ttk_runs_fixed/topology_finetuning/phase8_timing_s000_20261116/benchmark.json`.
- `--mode verify`: PASS; six output artifacts and canonical decoder SHA verified.

**Limitations:** One actual training field is insufficient to estimate full-field runtime distributions, batch/epoch cost, or GPU/CPU training overhead. GT diagrams can be cached, but predicted diagrams must be updated. No checkpoint was modified.

## CXIII.3 Real TTK synthetic-field gate — observed

The second script created four 160×160 physical scalar-speed fields, extracted genuine TTK PDs and recomputed canonical GUDHI distances: `gt`, `identity`, `spurious` (original peak plus a separate compact peak) and `displaced` (main peak moved).

- `--mode selftest`: PASS for scalar field shape, spurious bump and C-order VTI readback, **without** running TTK.
- `--mode preflight`: PASS for Docker backend, canonical H0/H1 decoder and GUDHI; archived inputs left unchanged.
- GT/identity/spurious/displaced TTK extraction times: **0.377 / 0.362 / 0.351 / 0.359 s**, respectively.
- `identity`: `W22=0`, predicted-to-diagonal assignments `0`, fixed-CV `0`, fixed-PERS `0`.
- `spurious`: `W22=6.3467759`, predicted-to-diagonal assignments `2`, fixed-CV `0`, fixed-PERS `0`. Extra predicted topological features are detected by the dynamic diagram comparison, while the original GT fixed-pair anchor values remain unchanged.
- `displaced`: `W22=0.12603661`, predicted-to-diagonal assignments `2`, fixed-CV `72`, fixed-PERS `142.513`. These CV/PERS values are **unweighted control-loss values**, not directly comparable in scale to W₂,₂. This demonstrates that ordinary birth/death PD distances do not encode spatial location and may be small even when an important feature moves.
- `--mode benchmark`: PASS for the spurious-feature discriminator. Output: `ttk_runs_fixed/topology_finetuning/phase8_synthetic_fields_20261116/`.
- `--mode verify`: PASS; 12 field/VTI/PD artifacts and canonical W₂,₂/matching counts were rechecked against their stored hashes.

**Interpretation:** The synthetic experiment provides actual *discrete extraction and matching* evidence that dynamically recomputed diagrams can penalize predicted spurious features missed by top-64 fixed GT-vertex CV/PERS losses. It does **not** demonstrate a usable gradient, improved reconstruction, or preservation of geographic location. The displaced result is an important counterexample to any unqualified claim that PD W₂,₂ enforces spatial alignment.

## CXIII.4 Next gate — planned, not yet executed

A separately versioned `phase8_matching_vertex_gradient_gate_20261116.py` has been prepared. It will use the **existing synthetic-field TTK artifacts** without rerunning them and without accessing the benchmark. It must map each positive finite canonical H0/H1 diagram point to predicted TTK critical vertex IDs in the *current predicted field*, reconstruct the full summed squared W₂,₂ matching cost, and verify local fixed-assignment analytical field gradients with finite differences.

For matched predicted/GT diagram points `(b_p,d_p) -> (b_g,d_g)`, the squared contribution is `(b_p-b_g)^2+(d_p-d_g)^2`. For a predicted unmatched pair, the diagonal squared contribution is `(d_p-b_p)^2/2`; for a GT unmatched pair it is a constant `(d_g-b_g)^2/2` and supplies **no** gradient to the current predicted field under fixed matching. Extra GT-anchor/reconstruction terms, if needed, must be separately labeled and ablated. Avoid claiming that differentiation passes through TTK or the discrete matching solver: the intended method differentiates a *local frozen-assignment surrogate* evaluated at the current field's critical vertices. Recompute TTK and matching after predictions change.

**Important risk:** Matching a diagram coordinate to a raw TTK pair via scalar value is ambiguous when feature values coincide. The prepared gate refuses ambiguous correspondences. Before implementing a robust 2,688-example native TF1 training runner, verify stable per-pair identifiers and dimension mapping rather than relying on value ties broken arbitrarily. Keep original Candidate F/Phase 6 outputs and pretrained checkpoints immutable.

## CXIII.5 Audit follow-ups

Before claiming the gradient gate has passed or making a new experiment claim, record its actual Spark stdout/stderr, pass/fail status, per-H0/H1 counts and matching types, exact source and result SHA256 values, execution environment, and any refinements required. The supplied transcripts did not include file-content hashes or complete per-dimension pair counts; none are invented here. Append a new dated section for future completed gates instead of silently rewriting prior entries.
