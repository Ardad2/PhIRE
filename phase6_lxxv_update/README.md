# Phase 6 — Part LXXV GT sample-0 parity diagnostic

**Status:** Prepared *after* the researcher observed a successful corrected-GUDHI sample-0 pilot. The script has not been run on Spark. It does **not** alter any historical or current experiment files.

## Motivation

The historical CNN GT diagram has D0=967, D1=871; the new gradient-only GT has D0=968, D1=871. A separate corrected-distance regression passed, and the new single-sample metrics were `dB_all=2.58057403564`, `W2inf_all=15.6249480697`, `W22_all=18.6702768797`. These are not yet a matched GT comparison. Historically, the TTK command used 20 threads; the isolated pilot used 1, but **no causality has been established**. Diagnose before scaling to all 168.

## Run on Spark

Transfer this ZIP into `~/PhIRE`, unzip it, and review the `README.md`. Run these **read-only** commands from the repository root:

```bash
cd ~/PhIRE
unzip Phase6_S0_GT_Parity_Gate_LXXV_20261014.zip -d phase6_lxxv_update

# Optional, applies only if LXXIV is the last current heading:
PATCH=phase6_lxxv_update/patches/phase6_notes_LXXIV_to_LXXV.patch
git apply --check "$PATCH" && git apply "$PATCH"

mkdir -p logs
set -o pipefail

# 1. File discovery + exact hashes; stdlib, no Docker
/usr/bin/python3 phase6_lxxv_update/scripts/phase6_s0_gt_parity.py \
  --mode inventory 2>&1 | tee logs/phase6_s0_gt_parity_inventory_20261014.log
printf 'Inventory exit: %s\n' "$?"

# 2. Compare sample-0 GT source vectors, VTI wind_speed, and spatial orientation.
#    Mount repo read-only; Docker image already contains VTK.
docker run --rm -v "$PWD:/work:ro" -w /work phire-ttk:latest \
  python3 phase6_lxxv_update/scripts/phase6_s0_gt_parity.py \
  --repo-root /work --mode arrays-vti \
  2>&1 | tee logs/phase6_s0_gt_parity_fields_20261014.log
printf 'Fields exit: %s\n' "$?"

# 3. Compare full canonical finite D0/D1 birth/death multisets.
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit python \
  phase6_lxxv_update/scripts/phase6_s0_gt_parity.py \
  --mode pd \
  2>&1 | tee logs/phase6_s0_gt_parity_pd_20261014.log
printf 'PD exit: %s\n' "$?"
```

If inventory cannot find the historic CNN VTI, **stop and share its error/inventory**, rather than assuming a path or regenerating a historical file. The diagnostic does not overwrite any VTI, PD, model, or report. A process exit of 0 means the *diagnostic ran*, not that parity was established: look for `[GT FIELD] MATCH` and `[GT DIAGRAM] MATCH` separately.

## How to interpret

- Same GT NumPy vector **and** same VTI speed field **and** same PD multiset: matched GT representation; sample-0 metric differences can be compared cautiously.
- Same GT vectors and VTI scalars but differing PD multiset: extraction settings/numerical/tie behavior may differ. Historical `-t 20` versus pilot `-t 1` is a candidate explanation, not a proven cause. Preserve existing files and run a controlled extraction with matched settings in a separate folder if needed.
- Different GT VTI scalar values: diagnose crop, orientation, scalar construction, output precision and source alignment before comparing distances.
- Different GT source vectors: resolve input/idx provenance first.

Do **not** run `scripts/run_candidate_topology_pipeline.sh` end-to-end solely to fix the problem; it includes superseded Stage 3 persistence distances. Do not launch the three-arm 168-sample extraction until the GT diagnostic is understood. Neither MT distance nor Phase 5E locality is evaluated here.
