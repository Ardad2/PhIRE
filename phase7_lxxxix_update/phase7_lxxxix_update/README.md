# Phase 7 Part LXXXIX — split freeze before model inference

**Scientific guard:** only covariate/footprint data from LXXXVII/LXXXVIII may be used to assign geographic roles. Inference on locked test regions must not be used to choose the six models, checkpoints, losses, radii, or analysis conventions. This package does *not* run inference.

## Proposed locked split

- Development: `gulf_inland`, `southwest_borderland`.
- Locked geographic external test: `upper_midwest`, `pacific_inland`.
- Primary windows per region: 2010-02-01, 2010-05-01, 2010-08-01, 2010-11-01, each 168 hours.
- Predeclared seasonal replication: 2012 equivalent weeks (not used to adapt the plan).
- First six-model inference pilot: **four development previews only** (Feb/Aug 2010).
- The phrase "external test" here means external to the **user's fine-tuning** footprint. The original PhIRE CNN/GAN pretraining coverage is unknown and may include these WTK years and locations.

## Installation from repo root

```bash
unzip Phase7_LXXXIX_Geographic_Split_Freeze_20261015.zip -d phase7_lxxxix_update
( cd phase7_lxxxix_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_lxxxix_update/scripts/*.py
PATCH=phase7_lxxxix_update/patches/phase7_notes_LXXXVIII_to_LXXXIX.patch
git apply --check "$PATCH" && git apply "$PATCH"
```

**Review `phase7_lxxxix_update/split_plan.json` before freezing.** The new master notes merely document the plan; only `--mode freeze` records the binding assignment.

## Three separate commands

```bash
/usr/bin/python3 phase7_lxxxix_update/scripts/phase7_freeze_geographic_split.py --mode preflight
/usr/bin/python3 phase7_lxxxix_update/scripts/phase7_freeze_geographic_split.py --mode freeze
/usr/bin/python3 phase7_lxxxix_update/scripts/phase7_freeze_geographic_split.py --mode verify
OUT=ttk_runs_fixed/topology_finetuning/phase7_lxxxix_locked_region_split_20261015
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/locked_region_split.md"
```

Use `set -o pipefail` if teeing to logs. Run the full path from `~/PhIRE`.

**No overwrites:** if the output already exists, do not delete it or repeat `--mode freeze`; use `--mode verify`.

This script reads only the locally archived source reports and eight preview inputs through LXXXVIII's read-only source verifier. It never contacts HSDS or runs TensorFlow or model inference. Source SHA checks and role assignments are strict.
