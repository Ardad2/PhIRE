# Part LXXX: validated full corrected PD results — paired-field audit

**Project status:** LXXIX preflight, all 840 GUDHI comparisons and post-run integrity verification passed on Spark; source means are copied into `lxxx_append.md`. This workpack is **prepared but not executed on Spark**.

The audit reads the existing LXXIX `per_sample_corrected_pd.csv`, `summary.json`, `manifest.json` and `results_sha256.txt` and verifies exact 840-cell pairing, source and CSV checksums, and consistency with the full-precision summary means and main effects. It calculates field-wise winners, paired wins/losses, medians, extreme example indices, and seven contiguous 24-hour block descriptive means. This is **not** a re-execution of GUDHI, a new TTK extraction, or a significance test.

## 1 — Install in ~/PhIRE and update the notes

```bash
cd ~/PhIRE
unzip Phase6_LXXX_Paired_PD_Audit_20261014.zip -d phase6_lxxx_update
( cd phase6_lxxx_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase6_lxxx_update/scripts/phase6_lxxx_paired_effect_audit.py
PATCH=phase6_lxxx_update/patches/phase6_notes_LXXIX_to_LXXX.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part LX' dataset_generation_and_repair_notes.md | tail -4
```

If the patch check fails, do not apply with `--reject`, use `git reset --hard`, or replace the notes wholesale. Diagnose differences first. The patch assumes that the master notes end with the unchanged LXXIX workpack note.

## 2 — Read-only paired-source preflight

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
/usr/bin/python3 phase6_lxxx_update/scripts/phase6_lxxx_paired_effect_audit.py --mode preflight \
  2>&1 | tee logs/phase6_lxxx_paired_preflight_20261014.log
printf 'LXXX preflight exit: %s\n' "$?"
```

Require `[SOURCE] PASS`, `[LXXX] PREFLIGHT PASS`, exit `0`. This stage writes no report to the experimental directories.

## 3 — Generate isolated descriptive report

```bash
cd ~/PhIRE
set -o pipefail
/usr/bin/python3 phase6_lxxx_update/scripts/phase6_lxxx_paired_effect_audit.py --mode report \
  2>&1 | tee logs/phase6_lxxx_paired_report_20261014.log
printf 'LXXX report exit: %s\n' "$?"

OUT=ttk_runs_fixed/topology_finetuning/phase6_lxxx_paired_effect_audit_20261014
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/paired_audit.md"
```

Outputs only under `ttk_runs_fixed/topology_finetuning/phase6_lxxx_paired_effect_audit_20261014/`: `paired_audit.md`, `paired_audit.json`, a source-CSV digest and output checksums. Re-running against the same input replaces only these derived reports; input hash drift is rejected.

## What to inspect next

1. For each PD metric, compare paired *counts* and *medians* (especially F1 vs PERS-only, CV-only vs gradient-only), not just five method means. Inspect the five largest regressions and seven non-overlapping 24-hour block means before calling an effect consistent across the week.
2. Confirm archived F1 lineage by checking model checkpoint identity, exact training-source file and loss configuration in the historical logs and Phase 6 matched protocol. The verified PD archive by itself does not prove identical training data and preprocessing.
3. Follow with Phase 5E spatial-locality reuse for new arms; preserve GT coverage vs constrained matching-cost separation and the validated radius grid `4,8,16,32,64,96,128,infinity`. Merge-tree distances remain separate.

**Important caveats:** No seeds or independent climate realizations; temporal dependence across 168 successive hours; the original F1 model may have been chosen after examining this benchmark. No hypothesis-test p-values, statistical significance or causal claims.
