# Phase 6 — LXXVI: GT diagonal-equivalence and frozen Phase-5E reuse gate

Status: **prepared locally; not run against Spark**. The supplied LXXV Spark logs have established identical benchmark arrays, canonical new VTI parity, and one extra exact zero-persistence GT D0 pair. This bundle adds an explicit numerical GUDHI invariance test and read-only discovery/validation of the already completed Phase 5E 672-file canonical PD archive.

## Install

Transfer `Phase6_LXXVI_GT_Equivalent_And_Phase5E_Reuse_20261014.zip` into `~/PhIRE` on Spark:

```bash
cd ~/PhIRE
unzip Phase6_LXXVI_GT_Equivalent_And_Phase5E_Reuse_20261014.zip -d phase6_lxxvi_update
( cd phase6_lxxvi_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase6_lxxvi_update/scripts/phase6_s0_diagonal_equivalence_and_phase5e_reuse.py
```

**Manifest paths** are relative to the extracted `phase6_lxxvi_update` directory; the parenthesized checksum command above checks them from that directory without changing your working shell.

## Apply the master-notes patch

The current Spark notes should end in Part LXXV, with a single Part LXXV heading. The patch is an append to the LXXV version that results from applying LXXIV and LXXV update patches to the user-provided LXXIII master. No existing paragraphs are rewritten.

```bash
cd ~/PhIRE
PATCH=phase6_lxxvi_update/patches/phase6_notes_LXXV_to_LXXVI.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part LX' dataset_generation_and_repair_notes.md | tail -5
```

If `git apply --check` fails, stop and compare the local end of the document; do not run `git reset --hard`, `--reject`, or overwrite your notes wholesale.

## Run read-only numerical gates

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python \
  phase6_lxxvi_update/scripts/phase6_s0_diagonal_equivalence_and_phase5e_reuse.py \
  --mode metric 2>&1 | tee logs/phase6_lxxvi_metric_20261014.log
printf 'Metric gate exit: %s\n' "$?"

PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python \
  phase6_lxxvi_update/scripts/phase6_s0_diagonal_equivalence_and_phase5e_reuse.py \
  --mode phase5e 2>&1 | tee logs/phase6_lxxvi_phase5e_reuse_20261014.log
printf 'Phase5E reuse gate exit: %s\n' "$?"
```

Modes do not generate or modify any VTI, VTU, model arrays, or audit CSV. A Phase-5E gate failure due to unexpected filename structure is a stop/inspect result, not a claim that data are corrupt. If needed, use `find spatial_pd/phase5e_all168_canonical_pd -name '*_pd_port_0.vtu' | head -20` and share the output.

## Gate outcomes

- `[HIST-NEW-GT] PASS`: positive-persistence D0/D1 pairs exactly equal across old and new GT. Zero-persistence diagonal points may differ.
- `[METRIC-GATE] PASS`: GUDHI reproduced historical CNN and gradient-only distances were unchanged by substituting historical, canonical or zero-filtered GT.
- `[PHASE5E-GATE] PASS`: 672 expected canonical diagrams located with every index, and Phase 5E GT sample-0 nonzero-persistence pair multiset equals the new Phase 6 GT sample-0 multiset.

If both pass, the next stage is a separately validated, isolated 168-field extraction for the **three new** Phase 6 reconstructions, followed by corrected distances, MT, and Phase 5E locality. Do not rerun historical corrected-PD audits or overwrite frozen Phase 5E artifacts.

## Local validation scope

The bundled Python script passed `py_compile`. The archive-inventory logic and nonzero persistence multiset check were additionally exercised on synthetic files. The container does not have GUDHI or Spark's experimental artifacts, so the real numerical recomputation remains explicitly unexecuted.
