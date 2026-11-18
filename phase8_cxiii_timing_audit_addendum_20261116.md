# Part CXIII — Phase 8: first native TTK/GUDHI timing and matching gate (November 16, 2026)

## CXIII.1 Scope and evidence

**Evidence received:** Actual terminal transcript from DGX Spark on November 16, 2026, for `phase8_ttk_w22_timing_pilot_20261116.py`. This was a **non-training** numerical/runtime feasibility test using sample ordinal 0 from the original **2,688-field training arrays**, not the held-out 168-field benchmark. No TensorFlow weight update, model training, Phase 7 external-region evaluation, checkpoint modification, or retuning occurred.

The source fields were `data_out/wind_mrhr_cnn_expanded2688/dataGT.npy` and `data_out/wind_mrhr_cnn_expanded2688/dataSR.npy`, each reported shape `(2688,500,500,2)`. Both were converted to physical scalar speed on the historical top-left 160×160 crop, written with validated C-order VTK point indexing, and processed with `phire-ttk:latest` / `ttkPersistenceDiagramCmd -t 1`. Numerical comparison used the existing corrected Phase 6 parser and GUDHI **W₂,₂** conventions: finite H0 and H1, q=2, internal_p=2, diagonal matching, essential points excluded, zero-persistence pairs ignored, dimensions combined by Euclidean norm. This is not the 64-pair top-K fixed E2 training objective.

## CXIII.2 Observed results

- `selftest` **PASS**: synthetic *diagram* identity/displacement/extra-point/diagonal cases and C-order orientation checks.
- `preflight` **PASS**: full training-array shapes, canonical parser, GUDHI environment and Docker TTK backend.
- TTK extraction: **0.748 seconds GT**, **0.367 seconds pretrained-CNN prediction** for the sample-0 160×160 speed fields.
- GUDHI matching: **0.115 seconds** reported for H0/H1 combined; manual match decomposition reproduced corrected **W₂,₂ = 20.673087** for this single sample.
- End-to-end benchmark: **2.997 seconds**, including more than just the sum of the individually logged TTK/matching times.
- Immutable output: `ttk_runs_fixed/topology_finetuning/phase8_timing_s000_20261116/benchmark.json`.
- Independent script `--mode verify` reported **PASS**, verifying six output artifacts and the canonical decoder SHA. Actual full JSON content/file SHA has not been pasted into this audit entry, so do not invent dimension-specific pair counts or checksums.

**Interpretation:** The installed TTK+GUDHI matching machinery works on this training sample, and the manually reconstructed matching cost agrees with the established corrected W₂,₂ calculation. One sample does **not** establish the runtime distribution for other 160×160 fields, minibatches, or integrated TensorFlow training. Precomputed GT diagrams may be cached; prediction diagrams must be refreshed at the chosen update frequency. No training improvement, spatial localization, or topological robustness is demonstrated here.

## CXIII.3 Next planned independent gate

A new, separate, collision-safe `phase8_synthetic_field_gate_20261116.py` with protocol `phase8_synthetic_field_protocol_20261116.md` has been locally syntax-checked and passed a NumPy/VTI construction selftest (no TTK/GUDHI runtime available in the preparation environment). It is **NOT YET EXECUTED ON SPARK**. It will test `identity`, `spurious` compact high-speed peak, and `displaced` peak scalar fields with real TTK diagrams, comparing corrected GUDHI matching with fixed GT-critical-pair CV/PERS proxies. In particular, the spurious case must produce an extra predicted feature and diagonal penalty while leaving selected GT fixed-pair endpoint values unchanged; the displaced case illustrates that ordinary PD W₂,₂ ignores spatial positions. A separate later stage must check the matching-derived gradient and its integration into the native TF1 training loop; this numerical gate is not a gradient check.

Before adding any later Part CXIV or committing Phase 8 scripts, record the exact new script SHA256, generated output SHA256, Spark command logs, per-H0/H1 match counts, failures/corrections and verification status. Preserve immutable Part CXII and all historical Phase 6/7 records.
