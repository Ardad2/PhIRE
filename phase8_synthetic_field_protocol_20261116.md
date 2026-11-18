# Phase 8: accepted sample-0 timing gate and next synthetic-field gate

**Date:** November 16, 2026  
**Context:** PhIRE native TensorFlow fine-tuning; TTK persistence diagrams on the top-left 160×160 scalar wind-speed crop.  
**Status:** Real training sample-0 TTK/GUDHI timing **executed and verified on Spark**. New synthetic-field gate **locally syntax/selftest checked; not yet executed on Spark**.

## 1. Recorded Spark evidence (not inferred)

User supplied the Spark console output for `phase8_ttk_w22_timing_pilot_20261116.py`:

- `--mode selftest`: PASS for diagram identity, displaced-diagram example, spurious-diagram example, unmatched-diagonal cost, and C-order VTI orientation.
- `--mode preflight`: PASS; training arrays `dataGT.npy` and `dataSR.npy` have shape `(2688, 500, 500, 2)`, sample ordinal `0`, Docker TTK and canonical GUDHI parser available.
- `--mode benchmark`: PASS; GT TTK 0.748 s, pretrained CNN SR TTK 0.367 s, GUDHI matching 0.115 s, computed corrected `W22=20.673087`, end-to-end 2.997 s, immutable report `ttk_runs_fixed/topology_finetuning/phase8_timing_s000_20261116/benchmark.json`.
- `--mode verify`: PASS, six output artifacts and canonical-decoder SHA verified.

**Interpretation:** The GT and pretrained prediction diagrams can be extracted and matched in the installed Spark environment, and the manual matching-cost decomposition reproduces the project’s validated corrected GUDHI W₂,₂ convention. These numbers are one sample on one machine: they are not a per-step benchmark for a full TensorFlow training session, not evidence of learned improvement, and not a verified runtime estimate for 2,688×3 training. A dynamic training step needs the current **predicted** diagram recomputed; GT diagrams can be cached. For batching, matching costs may change with sample complexity and execution environment.

## 2. What the next gate tests

The previous selftest used **hand-constructed persistence diagrams**, not diagrams extracted from deliberately changed scalar fields. The next gate writes real synthetic physical wind-speed fields and runs canonical TTK + corrected GUDHI:

1. `identity`: same field as GT, should have W₂,₂ ≈ 0.
2. `spurious`: identical GT feature plus a compact, geographically separated **extra high-speed peak**. Its value is unchanged at GT's selected fixed-pair vertices; the original Candidate F fixed-pair CV/PERS proxy should be exactly zero, but dynamically matched predicted PD should contain an extra pair matched to the diagonal and positive W₂,₂.
3. `displaced`: move the main high-speed peak. The fixed GT-vertex CV proxy should detect a change. Pure birth/death PD Wasserstein is **location-blind** and is not guaranteed to penalize spatial displacement; the gate reports its true numerical response instead of requiring it to be large.

This is a **behavioral differentiation** test, not yet a differentiable-training/gradient test. It validates the *discrete matching* half of Kissi-style dynamic supervision, not the TensorFlow training half.

Conventions: 160×160 float32 physical speed; C-order VTI writer; TTK CLI `-t 1 -i ... -a wind_speed -o ...`; canonical Phase 6 H0/H1 reader and GUDHI order=2/internal_p=2 matching with essential features excluded, zero-persistence points ignored; diagonal squared cost `(death-birth)^2/2`; H0/H1 costs combined by Euclidean norm. **Do not** silently substitute the 64-pair GT training constraints for all finite matching pairs. Fixed-pair proxy is separately computed using GT's top-64 finite persistence pairs above 1% of maximum persistence, strictly for methodological comparison.

## 3. Running the synthetic-field gate on Spark

Transfer `phase8_synthetic_field_gate_20261116.py` into `~/PhIRE` alongside `phase8_ttk_w22_timing_pilot_20261116.py`. From repo root:

```bash
cd ~/PhIRE

# Test field geometry and VTI orientation, without TTK or writing research artifacts
python3 phase8_synthetic_field_gate_20261116.py --mode selftest

# Verify the installed TTK/GUDHI environment and existing source scripts
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase8_synthetic_field_gate_20261116.py --mode preflight \
  --ttk-backend docker

# Only after both checks pass, run synthetic field extraction/matching
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase8_synthetic_field_gate_20261116.py --mode benchmark \
  --ttk-backend docker --ttk-timeout 180

# Recompute and verify stored numerical results and input file hashes
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase8_synthetic_field_gate_20261116.py --mode verify
```

New results are written once under:

`ttk_runs_fixed/topology_finetuning/phase8_synthetic_fields_20261116/`

The script does not access the 168-hour benchmark or checkpoint files. It does not edit archived Candidate F source or any existing results, and refuses to overwrite its output directory. If a scientific assertion fails (e.g., the extra peak fails to create a PD feature), it exits with a failing diagnostic rather than manufacturing a PASS; revise the **new** synthetic field construction in a versioned file and record that iteration.

## 4. Criteria to proceed to TensorFlow dynamic feeds

- Synthetic-field extraction and matching must pass and independently verify.
- Actual predicted persistence-pair vertex indices must be associated with the **current prediction** on each optimization step, not cached GT vertex IDs.
- Prototype a small matching-to-vertex-target routine; verify signed birth/death values and handling of matched, SR-unmatched, GT-unmatched, H0/H1 and essential pairs. An unmatched predicted pair’s midpoint is its exact Euclidean diagonal projection. An unmatched GT pair has a constant diagonal cost under the *fixed assignment*, so any extra GT-critical-value supervision is a separate objective and must not be described as exact Wasserstein gradient.
- Distinguish loss definition (raw squared W₂,₂, W₂,₂, pairwise mean, optional weights) and its gradient scale; log every setting. Verify loss and gradient numerically against an independent small example.
- Only then implement a separate, collision-safe native PhIRE Candidate F-derived runner, with a GT PD cache built strictly from training fields. Preserve the original 168-hour benchmark for final reporting.

## 5. Suggested additive notes entry (after Spark gate)

Once `--mode benchmark` and `--mode verify` pass, append a new Part CXIII only after checking the current final section numbers. Record script SHA-256, output JSON and SHA-256, GT/CNN baseline, TTK timings, per-H0/H1 match counts, fixed-pair CV/PERS on the spurious and displaced cases, verification transcript, and any failed/revised synthetic field definitions. Do not revise Part CXII retrospectively or label the experiment trained.
