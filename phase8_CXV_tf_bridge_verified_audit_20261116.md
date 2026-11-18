# Part CXV — Phase 8 native-TensorFlow dynamic PD bridge: stage-path repair and verified generator gradient (November 16, 2026)

**Audit status:** Verified on DGX Spark from the recorded terminal execution, before any optimizer update. **Scope:** training TFRecord sample index 0 only, top-left 160×160 physical-speed crop, full canonical H0/H1 Euclidean-ground W₂,₂; no evaluation/held-out inference and no saved weights.

## Earlier failed bridge and repair

The initial standalone `phase8_tf1_dynamic_single_batch_gate_20261116.py` successfully restored the frozen pretrained CNN and performed a forward pass (`L_uv=0.013026411`, forward about 0.600 s), but its `match` and `verify` stages failed after the forward stage: the output manifest referenced temporary `.forward.staging-*` absolute file paths that ceased to exist after an atomic rename. This was an artifact manifest/path bug, not evidence of a failed persistence loss. The `v2` script records stage-relative paths, verifies their SHA-256 hashes at their finished location, and preserves the original failed output directory without overwriting it. The repair was locally syntax- and stage-path-tested and subsequently exercised on Spark.

## Successful Spark v2 results

Script: `phase8_tf1_dynamic_single_batch_gate_20261116_v2.py`, output:
`ttk_runs_fixed/topology_finetuning/phase8_tf1_single_batch_bridge_20261116_v2/`.

Stages reported PASS for: `selftest`, `preflight`, `forward`, `match`, `tfcheck`, `verify`. The forwarded training sample was index 0 from the 2,688-sample training TFRecord; the pretrained CNN checkpoint was restored read-only. Observed values:

- Forward: normalized `L_uv=0.013026411`, elapsed `0.783 s`.
- TTK: GT `0.521 s`, predicted field `0.355 s`.
- Canonical matched diagram distance: `W22=20.6730864` (H0 and H1 combined), matching-derived squared `J=427.376501`; `1187` predicted finite pairs used by the matching feed.
- Native TensorFlow gathered-vertex loss: `J_TF=427.376465`. Absolute squared-loss difference about `3.6e-5`.
- Maximum TF versus independent field-gradient discrepancy: `4.77e-7`; generator-parameter gradient global norm: `12975.8`.
- Final verification: PASS for frozen source/checkpoint hashes, canonical matching, TensorFlow gather gradient; **no optimizer, no model training and no checkpoint written**.

The values demonstrate a numerically consistent local **fixed-matching** loss and nonzero gradients through the *native PhIRE generator*. They do **not** show that taking a step will improve the true W₂,₂, do not establish gradients through TTK/GUDHI, and do not provide held-out improvement evidence. Re-matching at every scheduled training step remains unimplemented at this stage.

## Remaining issues and planned gate

- NumPy 2.4.2 generated import warnings for some packages built against NumPy 1.x (`numexpr` and `bottleneck` in the mixed Python environment); both native TensorFlow stages nevertheless completed. This remains an environment hygiene concern, **not** a reason to silently alter installed packages.
- The v2 run's precommit `git HEAD` was `c287be40a216fabf90f3fd18a2ac181b4334d0ac`. A subsequent documentation commit, `4f120e12`, was successfully pushed to `master` and added Part CXIV. Preserve the source hashes recorded by each stage manifest; do not assume every artifact was built after that later commit.
- Next **planned, not executed** gate: two independent single-sample, one-step Adam updates (normalized `L_uv+0.05 L_grad` control versus the same plus a small weighted frozen-matching `J=W22²` term). Re-extract predicted diagrams **after** both steps and compare true W₂,₂ using the same canonical H0/H1 conventions. Keep pretrained checkpoint immutable, use a fresh output directory, and avoid held-out data.

**Provenance:** Spark console transcript supplied November 16, 2026; independent artifact/checkpoint hash checks and archived numerical measurements are recorded in the v2 output manifests. No extra source SHA or nonexistent output is asserted here.
