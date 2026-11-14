# Phase 6 documentation update — Part LXXII (2026-10-14)

**Correct base is Part LXX.** Your most recent `grep '^# Part LX' dataset_generation_and_repair_notes.md` output ended at Part LXX. The failed `phase6_notes_LXVII_to_LXXI.patch` was for the wrong base and did not change your file.

This package contains a cumulative **LXX→LXXII** patch adding the historical Part LXXI training note and the completed Part LXXII full quantitative comparison. It also has the full 1.1 MB updated Markdown notes as a fallback reference. Do not replace the repo file with it unless you have verified your version; prefer the patch.

Transfer this ZIP to Spark repository root, then:

```bash
cd ~/PhIRE
unzip Phase6_PERS_Only_FourArm_Notes_LXXII_20261014.zip -d phase6_lxxii_update

grep '^# Part LX' dataset_generation_and_repair_notes.md | tail -6
PATCH=phase6_lxxii_update/patches/phase6_notes_LXX_to_LXXII_cumulative.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part LX' dataset_generation_and_repair_notes.md | tail -7
git diff --stat -- dataset_generation_and_repair_notes.md
```

Afterward, the last headings should be LXXI and LXXII. No manuscript figures, models, existing report files or frozen Git tag are changed by this patch. If `git apply --check` fails, **stop** and share `git status --short` and the last headings; do not use `--reject`, `git reset --hard` or overwrite the original notes blindly.

To reproduce descriptive four-arm contrasts **after confirming all four Markdown reports exist in `~/PhIRE/docs`**:

```bash
cd ~/PhIRE
/usr/bin/python3 phase6_lxxii_update/scripts/compare_phase6_2x2_reports.py --repo-root "$PWD"
```

The comparison script uses only the standard library, reads the actual four Markdown reports, and leaves repository files unchanged. Full results are in `docs/phase6_pers_only_four_arm_findings_LXXII_20261014.md`. New corrected PD/MT and spatial-locality numbers are **not yet available**.
