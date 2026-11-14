# Phase 6 — LXXIV sample-0 corrected-PD gate

**Evidence status:** The gradient-only sample-0 VTI + TTK PD + MT extraction was reported successful by the researcher on Spark (exit status 0). The attached `phase6_s0_corrected_pd_gudhi.py` has been syntax-checked and its local read-only preflight tested, but the GUDHI calculation has **not** been executed on Spark here. Do not report corrected candidate topology numbers until the Spark run itself passes.

## Contents

- `patches/phase6_notes_LXXIII_to_LXXIV.patch`: additive master-notes update; apply only when the Spark file already ends at Part LXXIII.
- `scripts/phase6_s0_corrected_pd_gudhi.py`: standalone sample-0 corrected persistence-distance calculator, using the **archived** `canonical_pd_pilot.read_pd` parser and the existing `gudhi-audit` environment.

## Install on Spark

Transfer this ZIP to `~/PhIRE` and unzip it as `phase6_lxxiv_update`.

```bash
cd ~/PhIRE
unzip Phase6_S0_Corrected_PD_Audit_LXXIV_20261014.zip -d phase6_lxxiv_update

grep '^# Part LX' dataset_generation_and_repair_notes.md | tail -4
PATCH=phase6_lxxiv_update/patches/phase6_notes_LXXIII_to_LXXIV.patch
git apply --check "$PATCH" && git apply "$PATCH"
```

If `git apply --check` fails, **do not force the patch**. Confirm which Part heading is in your current notes; it may be ahead of or behind the supplied LXXIII base.

## Step 1 — read-only preflight (no GUDHI)

```bash
cd ~/PhIRE
/usr/bin/python3 phase6_lxxiv_update/scripts/phase6_s0_corrected_pd_gudhi.py --preflight-only
```

Checks 2 VTI, 2 PD, 6 MT non-empty artifacts under the sample-0 pilot, the preserved canonical parser, and the historical CNN sample-0 PDs. **This preflight does not validate new diagrams numerically.**

## Step 2 — corrected-distance regression and new model sample 0

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail

PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase6_lxxiv_update/scripts/phase6_s0_corrected_pd_gudhi.py \
  2>&1 | tee logs/phase6_s0_corrected_pd_gudhi_20261014.log

echo "Corrected-PD audit exit status: $?"
```

The script checks the **historical CNN sample-0** finite diagram counts and frozen corrected aggregate values before analyzing the new candidate:

- `GT D0 = 967`, `SR D0 = 703`; `GT D1 = 871`, `SR D1 = 632`
- corrected `d_B = 3.0914716720581055`
- corrected `W_{2,∞} = 17.467813434330861`

It then computes the gradient-only sample-0 `d_B`, `W_{2,∞}`, and **separately labeled** `W_{2,2}` with the canonical finite-pair semantics, aggregating D0/D1 identically to the archived audit. Results and input-file SHA-256s are written to a **new** isolated folder:

`ttk_runs_fixed/topology_finetuning/candidateF_grad_pairzero_expanded2688_s0_corrected_pd_audit_20261014/phase6_s0_corrected_pd.json`

The script refuses to overwrite an existing JSON. If a rerun is necessary, preserve the first run and provide a new `--out-dir`.

## What the result does **not** establish

A successful result validates the *distance-computation path* for this one candidate/sample (conditioned on the extracted TTK PDs), including a known reference regression. It **does not** validate the sample-0 merge-tree distance, GT/SR feature-locality correspondence, or full-168 sample trend. It also does not independently reconstruct the persistence pairs from raw scalar fields; GUDHI computes distances on the TTK-extracted pairs. Full extraction of all variants should wait until the regression gate passes.
