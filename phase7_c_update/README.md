# Part C | full-week development source readiness and guarded acquisition

**Scope:** one complete frozen 168-hour *development* week only: gulf_inland, February 1–7, 2010 UTC, WTK indices 27048–27215; rectangle `[205,705,1471,1971]` in WTK rows/columns. Reuse the existing XCV hour-0 and hour-84 source files bit-for-bit; 166 additional HSDS wind slices if the live probes pass. **No model inference, GUDHI, TTK, training or locked external-region data.**

## Package installation from PhIRE root

```
unzip Phase7_C_Development_Fullweek_Streaming_20261018.zip -d phase7_c_update
( cd phase7_c_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_c_update/scripts/*.py
PATCH=phase7_c_update/patches/phase7_notes_XCIX_to_C.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part ' dataset_generation_and_repair_notes.md | tail -3
```

This ZIP has **flat** members and can be extracted exactly as above; do not extract over an existing `phase7_c_update` directory. Notes patch applies only after Part XCIX is present in the master notes. Do not reapply it.

## 1. Local selftest and offline, read-only preflight

```
cd ~/PhIRE
mkdir -p logs
set -o pipefail
SCRIPT=phase7_c_update/scripts/phase7_c_dev_fullweek_source.py
/usr/bin/python3 "$SCRIPT" --mode selftest --repo-root "$PWD"
/usr/bin/python3 "$SCRIPT" --mode preflight --repo-root "$PWD" \
  2>&1 | tee logs/phase7_c_preflight_20261018.log
echo "Preflight exit: $?"
```

Require `[SELFTEST] PASS`, `[FROZEN] PASS`, `[SCOPE] PASS`, `[BUDGET] PASS`, `[PREFLIGHT] PASS`, and exit 0. Source-only budget requires at least 2 GiB free disk and 2 GiB Linux MemAvailable. This is intentionally conservative; the report provides raw array byte counts but **does not estimate runtime**. Offline mode does not access HSDS, model checkpoints or locked external field arrays.

## 2. Explicit online HSDS metadata probe (no wind fields)

```
export HS_ENDPOINT="https://developer.nlr.gov/api/hsds"
export HSDS_ENDPOINT="$HS_ENDPOINT"
/usr/bin/python3 "$SCRIPT" --mode probe --repo-root "$PWD" \
  2>&1 | tee logs/phase7_c_probe_20261018.log
echo "Probe exit: $?"
```

Require `[HSDS-PROBE] PASS` and exit 0, confirming every one of 168 source UTC indices matches frozen week timestamps; only HSDS dataset metadata and `datetime` are accessed. No new training, model inference, or geographic test reads.

## 3. Guarded first HSDS batch (six new hours), then resume

```
/usr/bin/python3 "$SCRIPT" --mode extract --repo-root "$PWD" --max-new-hours 6 \
  2>&1 | tee logs/phase7_c_firstbatch_20261018.log
echo "First batch exit: $?"
```

Require a `[PARTIAL-GATE] PASS` and exit 0; per-hour SHA index is persisted under the protected `.staging` directory. Initial batch reuses hour 0 and fetches hours 1–6; hour 84 is reused when reached in later runs. Do **not** remove or rename `.staging` and do not overwrite an existing completed output.

If all progress checks and source index timings look correct, resume:

```
/usr/bin/python3 "$SCRIPT" --mode extract --repo-root "$PWD" \
  2>&1 | tee logs/phase7_c_extract_20261018.log
echo "Extraction exit: $?"
```

Expect `[SOURCE-WEEK-GATE] PASS` on completion; any interruption preserves checksum-indexed completed hourly samples. Rerun the same `extract` command to resume. A changed source/config, altered file, orphan or corrupt hour triggers a **fail-closed** error instead of silently dropping an hour.

## 4. Verify complete week without network reads

```
/usr/bin/python3 "$SCRIPT" --mode verify --repo-root "$PWD" \
  2>&1 | tee logs/phase7_c_verify_20261018.log
echo "Verify exit: $?"
OUT=ttk_runs_fixed/topology_finetuning/phase7_c_dev_gulf_inland_feb2010_week168_20261018
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/fullweek_report.md"
```

Expect `[VERIFY-GATE] PASS`, exit 0, **173 `OK` checksums** (168 NPZ + config + progress + manifest + CSV + report). Mode `verify` has no HSDS, TensorFlow, or TTK calls. Passing this source-only gate does **not** mean inference or topology on the 168-hour week is complete.

## Resource and statistical warnings

Each HR `[u,v]` float64 field is 4,000,000 bytes and each MR `[u,v]` field 160,000 bytes. Entire week source arrays: 698,880,000 bytes (~666.50 MiB) uncompressed. Six full500 model outputs for this week would require another 4,032,000,000 bytes (~3.75 GiB) **before** TFRecords, intermediate files, TTK diagrams and model runtime memory. HSDS throughput is unknown until measured. The first full-week field week is a **development engineering pilot**; no independent hourly significance or locked external generalization claim is warranted.
