# Phase 6 topology preflight documentation (LXXIII)

From Spark `~/PhIRE` (transfer this ZIP first):

```bash
unzip Phase6_Topology_Preflight_and_S0_Pilot_Notes_LXXIII_20261014.zip -d phase6_lxxiii_update
PATCH=phase6_lxxiii_update/patches/phase6_notes_LXXII_to_LXXIII.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part LX' dataset_generation_and_repair_notes.md | tail -5

# Isolated sample-0 extraction, no legacy distance computation:
bash -n phase6_lxxiii_update/scripts/phase6_topology_s0_extraction_pilot.sh
mkdir -p logs
set -o pipefail
bash phase6_lxxiii_update/scripts/phase6_topology_s0_extraction_pilot.sh 2>&1 | tee logs/phase6_topology_s0_extraction_pilot_20261014.log
echo "Pilot exit status: $?"
```

The preflight and patch are validated; the actual Docker pilot has NOT run on Spark yet.
Do not run the full legacy wrapper for corrected PD metrics.
