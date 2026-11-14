# Phase 7 — Part LXXXVI compact WTK datetime compatibility hotfix (2026-10-15)

Observed Spark probe:

- `[GRID] WTK wind shape=(61368,1602,2976)` — remote HSDS connection and coordinate inventory worked.
- `[LEGACY] PASS old reference near grid=(970,2352)` — original 500x500 field location matched.
- `[SELECTION] PROVISIONAL ...` — geographic candidates successfully screened.
- `[LXXXVI] ERROR ValueError: Unparseable WTK datetime item '20070101000000'` — compact timestamp not supported by the original ISO parser.

This **adapter**, not a modification, imports the original verified script by exact source SHA-256 and overrides only its `to_datetime` function in memory. Fourteen-digit `YYYYMMDDHHMMSS` values such as `20070101000000` parse through `datetime.strptime`; original ISO timestamp behavior is unchanged. Both the script and geographic candidate list are pinned by SHA256. The wrapper prints its own SHA256 into the captured log to document the executed adapter. All original project scripts, notes, geographic candidates, and outputs stay untouched.

## Install (from the repo root)

```
cd ~/PhIRE
unzip Phase7_LXXXVI_WTK_Datetime_Compat_20261015.zip -d phase7_lxxxvi_datetime_hotfix
( cd phase7_lxxxvi_datetime_hotfix && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_lxxxvi_datetime_hotfix/scripts/phase7_lxxxvi_datetime_compat.py
```

**Do not** rerun the original LXXXVI `git apply` patch: Part LXXXVI is already committed in the on-host notes. This hotfix has no master-notes patch because it makes no scientific or experimental changes. Its SHA and the probe log provide the compatibility provenance.

## Probe (real HSDS connection required)

```
cd ~/PhIRE
export HS_ENDPOINT="https://developer.nlr.gov/api/hsds"
export HSDS_ENDPOINT="$HS_ENDPOINT"
mkdir -p logs
set -o pipefail
python3 phase7_lxxxvi_datetime_hotfix/scripts/phase7_lxxxvi_datetime_compat.py --mode probe \
  2>&1 | tee logs/phase7_lxxxvi_datetime_compat_probe_20261015.log
echo "Probe exit: $?"
```

Expect `[TIME] 8/8`, `[PROBE] PASS`, and exit 0. The WTK metadata is consulted remotely, but no wind-speed or wind-direction slices are read. The script still refuses to overwrite the original isolated output directory.

## Verify (the original verifier is unchanged)

```
cd ~/PhIRE
set -o pipefail
python3 phase7_lxxxvi_update/scripts/phase7_region_inventory.py --mode verify \
  2>&1 | tee logs/phase7_lxxxvi_verify_20261015.log
echo "Verify exit: $?"
OUT=ttk_runs_fixed/topology_finetuning/phase7_lxxxvi_region_inventory_20261015
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/inventory.md"
```

If the probe fails again, do not remove files or change dates; send its complete output. The four proposed regions are **provisional** and do not yet constitute locked train/validation/test splits. Time-index validation does not establish disjointness from the archived fine-tuning dates or the original pretrained model's training data.
