# Phase 6 — Part LXXXV: F1 historical lineage evidence audit

This is a targeted **read-only provenance check**, not a newly trained experiment, automatic lineage certification, or new topology metric. The LXXXIV report has been independently verified on Spark. The next step is to review actual F1 and new-arm training logs and model artifacts, as the full four-arm spatial contrasts are already complete.

## Install

Place `Phase6_LXXXV_F1_Lineage_20261015.zip` in `~/PhIRE`:

```bash
cd ~/PhIRE
unzip Phase6_LXXXV_F1_Lineage_20261015.zip -d phase6_lxxxv_update
( cd phase6_lxxxv_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase6_lxxxv_update/scripts/phase6_lxxxv_f1_lineage_audit.py
PATCH=phase6_lxxxv_update/patches/phase6_notes_LXXXIV_to_LXXXV.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part LX' dataset_generation_and_repair_notes.md | tail -4
```

Last heading should be Part LXXXV. If the patch fails, do not force it or reapply older patches.

## Preflight (read-only)

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
/usr/bin/python3 phase6_lxxxv_update/scripts/phase6_lxxxv_f1_lineage_audit.py \
 --mode preflight 2>&1 | tee logs/phase6_lxxxv_f1_preflight_20261015.log
echo "F1 preflight exit: $?"
```

This checks the frozen LXXXIV source hashes and the exact benchmark `idx.npy` / `dataGT.npy` SHA256 parity of F1 and gradient-only. It also fingerprints historical logs and current source/checkpoint metadata. **Preflight success does not mean optimizer or seed provenance is closed.** Large checkpoint data/TFRecord hashes are skipped by default; `--deep-hash` enables them, but hashing their current bytes still does not establish July-time bytes.

## Evidence report (new isolated directory)

Only after successful preflight:

```bash
cd ~/PhIRE
set -o pipefail
/usr/bin/python3 phase6_lxxxv_update/scripts/phase6_lxxxv_f1_lineage_audit.py \
 --mode report 2>&1 | tee logs/phase6_lxxxv_f1_report_20261015.log
echo "F1 report exit: $?"
```

Output: `ttk_runs_fixed/topology_finetuning/phase6_lxxxv_f1_lineage_audit_20261015/`.

## Verify and inspect

```bash
cd ~/PhIRE
set -o pipefail
/usr/bin/python3 phase6_lxxxv_update/scripts/phase6_lxxxv_f1_lineage_audit.py \
 --mode verify 2>&1 | tee logs/phase6_lxxxv_f1_verify_20261015.log
echo "F1 verify exit: $?"

OUT=ttk_runs_fixed/topology_finetuning/phase6_lxxxv_f1_lineage_audit_20261015
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/f1_training_lineage.md"
```

**Please provide** the report (or its relevant sections) and the exit codes. Compare the exact original F1 objective `L_uv + 0.05 L_grad + 0.004 L_TTKCV + 0.002 L_TTKpers` with the matched new controls, and look for training-time source revision, initial pretrained checkpoint, 2,688 training fields, 64 pair constraints, optimizer path and random seed evidence. Missing log text means **unverified**, not **different**. No automatic certification is asserted.

`--mode report` refuses overwrite. `--mode verify` checks that source evidence and reports remain unchanged. No saved model, TTK output, Phase5E, LXXIX, LXXXIII or LXXXIV artifact is altered. The frozen publication tag is left alone.
