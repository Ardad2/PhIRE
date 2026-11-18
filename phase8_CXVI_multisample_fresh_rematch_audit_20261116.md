# Part CXVI — Phase 8: four-sample, eight-step fresh dynamic-PD rematching pilot (November 16, 2026)

**Status: VERIFIED DEVELOPMENT EXPERIMENT on DGX Spark; not a held-out/generalization result.** This is an additive record following Part CXV. The evidence consists of the Spark terminal transcript supplied on November 16, including the `--mode verify` PASS and its eight recomputed TTK/GUDHI comparisons. Original program code, model weights, earlier experiment outputs, and historical notes were not changed by this entry. The reported metrics are transcript observations, not independently downloaded artifact contents.

## Protocol and provenance

- New standalone pilot source: `phase8_tf1_multisample_rematch_pilot_20261116.py` in `~/PhIRE`; source and checkpoint dependencies are checked by the pilot. It reused the independently verified v2 TF1 bridge and one-step graph. **No modification to published Candidate F training scripts.** GitHub notes HEAD before this addendum was reported at commit `c751ef89` (`Audit Phase 8 TensorFlow bridge and one-step optimizer`). A standalone source-file SHA256 was **not quoted** in the supplied transcript; consult the immutable output manifest rather than inventing one.
- Data: indices **0, 1, 2, 3** from the 2,688-example *training* TFRecord, not the frozen 168-field evaluation benchmark. Four normalized samples were exported; four GT persistence diagrams were computed once and cached. GT TTK finite-pair dimensions in the preflight/prepare transcript: s0 D0=979/D1=896; s1 990/909; s2 977/928; s3 967/918.
- Initialization: two separate TensorFlow 1-compatible Adam sessions restored the same pretrained wind CNN checkpoint, with independent optimizer state. Eight updates per arm in order `(0,1,2,3,0,1,2,3)`; batch size 1; learning rate `1e-6`; gradient coefficient `0.05`.
- Control objective: normalized `L_uv + 0.05 L_grad`. Dynamic objective adds `1e-5 * J_PD`, where `J_PD` is the **unaveraged squared** Euclidean-ground `W_{2,2}^2` matching surrogate computed on denormalized scalar speed over the top-left 160×160 crop. The diagnostic coefficient is not optimized or validated as a recommended value.
- At every dynamic step, the *current predicted* speed field was re-extracted with TTK, H0/H1 pairs re-matched to the cached GT diagram via GUDHI, and predicted birth/death critical vertices were fed to `tf.gather_nd`. Matching and critical-vertex selection are held fixed **during** each differentiable optimization step; TTK/GUDHI are not differentiated. Independently recomputed true TTK/GUDHI `W_{2,2}` was measured for **both arms on the current training field after each update**.
- Immutable output root: `ttk_runs_fixed/topology_finetuning/phase8_multisample_rematch_20261116_r01/` (`inputs/`, `gt_cache/`, `train/`). The final verifier reported no checkpoint writes and no held-out data access.

## Execution issue and recovery (preserve for audit)

1. Initial `--mode train` failed before launching an optimizer step: `RuntimeError: micromamba not available`. This was an executable discovery mismatch: the interactive shell exposed `micromamba` as a function, `MAMBA_EXE=/home/adadhwal/Desktop/bin/micromamba`, and Python `shutil.which('micromamba')` initially returned `None`.
2. The environment-only correction added `$(dirname "$MAMBA_EXE")` to the shell `PATH`. Python then located `/home/adadhwal/Desktop/bin/micromamba`; `PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python -c 'import gudhi'` passed. **No pilot source edit or environment reinstall was necessary.**
3. The resumed eight-step run completed and independently verified. TensorFlow imports emitted noisy NumPy 2.4.2 versus NumPy-1-compiled optional dependency warnings (`numexpr`, `bottleneck`) and TF deprecation messages, but the job proceeded to explicit PASS outputs. The warnings remain an environment hygiene concern, not an observed training-stage failure.

## Verified per-step TRUE `W_{2,2}` after optimizer update

| Step | Training index | Control (lower is better) | Dynamic PD (lower is better) | Dynamic − control |
|---:|---:|---:|---:|---:|
| 1 | 0 | 20.650547 | 20.602099 | −0.048448 |
| 2 | 1 | 21.037756 | 20.970895 | −0.066860 |
| 3 | 2 | 21.444351 | 21.349161 | −0.095190 |
| 4 | 3 | 22.799111 | 22.672562 | −0.126549 |
| 5 | 0 | 20.575978 | 20.380917 | −0.195061 |
| 6 | 1 | 20.960697 | 20.747571 | −0.213126 |
| 7 | 2 | 21.367292 | 21.122726 | −0.244566 |
| 8 | 3 | 22.712298 | 22.433847 | −0.278452 |

The verifier returned `dynamic_better_steps=8`, `dynamic_worse_steps=0`, `mean_delta=-0.15853146389738182` (dynamic minus control across the eight *paired*, correlated comparisons), and `train_steps=8`. Mean distances computed from the displayed pairs are approximately control **21.443504** and dynamic **21.284972**, a **0.739%** lower mean for dynamic relative to the control mean. First pass mean delta ≈ **−0.084262**; second pass ≈ **−0.232801**. Because fields differ by step and indices are reused, neither the mean nor the improving gap constitutes an independent-sample statistical estimate.

Fresh predicted pair counts for the eight pre-update dynamic diagrams: **1187, 1159, 1206, 1171, 1179, 1165, 1221, 1173**. Logged prediction TTK extraction times ranged approximately **0.341–0.405 seconds** and post-update field TTK times approximately **0.365–0.426 seconds**. These are **not** end-to-end per-step worker runtimes; worker startup, GUDHI matching, TensorFlow forward/backward, evaluation, I/O, and hashing add overhead. Consult `train` step JSON fields `seconds_match_worker` and `seconds_eval_worker` for complete worker timing analysis.

Explicit verification: `[TRAIN] PASS: 8 fresh dynamic rematchings, two independent TF1 Adam sessions, frozen checkpoint unchanged` followed by `[VERIFY] PASS: 8 fresh W22 matchings and 2x8 independent true TTK/GUDHI checks; no held-out data, no checkpoint writes`.

## Interpretation, limits, and next gate

- **Supported:** Prediction-dependent fresh PD matching can be used in repeated native TF1 optimizer steps with stable gradients on this four-field training mini-pilot; the dynamic arm achieved a lower *recomputed* topology distance than a matched control in all eight within-training post-update comparisons.
- **Not established:** performance on new training examples or the sealed 168-field test benchmark; robustness across seeds, field regions, crop positions, or sample ordering; whether normalized `L_uv`, speed error, physical gradients, or other field-quality metrics deteriorate; runtime feasibility for thousands of examples; statistical significance or an optimal PD-loss coefficient. Numerical confidence is conditional on the existing canonical TTK/GUDHI conventions.
- **Next decision gate:** Inspect actual `L_uv` and `L_grad` before/after values and worker runtimes from the *stored per-step JSON*, then run an isolated **no-update evaluation** on predetermined, never-updated training-pool indices (for example 4–7) for both final in-memory models; additionally consider an independent seed/order repeat. Do not promote or tune against the sealed 168-field benchmark. Since the pilot intentionally saved no trained weights, an extension needing end-of-run model predictions must be implemented in a **new versioned runner** and rerun; the published immutable outputs must not be altered.

**Audit status:** This section records a completed development experiment. The suggested follow-up is **PLANNED / NOT RUN** and is not represented as an experimental outcome.
