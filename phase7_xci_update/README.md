# Phase 7 XCI — freeze geographic evaluation protocol before any locked-region inference

This **prepares** a test protocol; it does not run new metrics, extract WTK meteorology, or use test predictions. It expects real XC results to have passed source and output verification (including 11 SHA entries). The original LXXXIX split remains immutable.

Installation (from `~/PhIRE`; flat ZIP):

```bash
unzip Phase7_XCI_Evaluation_Protocol_Freeze_20261015.zip -d phase7_xci_update
( cd phase7_xci_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_xci_update/scripts/*.py
PATCH=phase7_xci_update/patches/phase7_notes_XC_to_XCI.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part ' dataset_generation_and_repair_notes.md | tail -3
```

Run with `/usr/bin/python3` (no TensorFlow needed for XCI) and `set -o pipefail`:

```bash
cd ~/PhIRE
S=phase7_xci_update/scripts/phase7_freeze_evaluation_protocol.py
/usr/bin/python3 "$S" --mode selftest
/usr/bin/python3 "$S" --mode preflight --repo-root "$PWD"
/usr/bin/python3 "$S" --mode freeze --repo-root "$PWD"
/usr/bin/python3 "$S" --mode verify --repo-root "$PWD"
OUT=ttk_runs_fixed/topology_finetuning/phase7_xci_frozen_evaluation_protocol_20261015
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/evaluation_protocol.md"
```

Every step must exit 0 before the next. Source checks rerun XC verification of input arrays and all six predictions on **development** only. The protocol is immutable; if a technical correction is necessary before locked inference, make a new amendment/version, do not overwrite it. The protocol specifies the original 160×160 top-left speed crop for **primary historical comparability**, a 3×3 set of nonoverlapping 160×160 tiles as **secondary**, and a pilot of complete 500×500 topology on dev only. Full topology implementation will be validated separately before any external run; this package does not certify topological extraction or solver behavior on the new data.
