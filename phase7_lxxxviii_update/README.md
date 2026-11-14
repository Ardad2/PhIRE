# Phase 7 — Part LXXXVIII: archived preview parity & geographic footprint review

This is the next isolated, **no-model-scores** checkpoint following successful live LXXXVII: 64/64 full native WTK 500² snapshots valid, 8 `hr_uv/mr_uv` previews saved, SHA256 verification passed. The screening means by candidate are Gulf 6.494 m/s; Upper Midwest 6.830; Pacific inland 6.043; Southwest borderland 5.277. Mean field p95 absolute pixel gradients vary markedly, Pacific 2.011 vs Midwest 0.783 m/s per grid-cell difference. These are pilot descriptors, not terrain classification or statistical generalization.

### Contents

- `scripts/phase7_preview_parity.py`: `--mode preflight|parity|verify`.
- `lxxxviii_append.md` and Git patch to **append** Part LXXXVIII to the existing master notes (requires Part LXXXVII).
- Checksums of the package files.

The script imports the unchanged LXXXVII source validator, verifies all 11 screening output files, reads 8 preview arrays with `allow_pickle=False`, checks each preview's time/index/500x500 WTK rectangle against screened metadata, displays all available geographic centers and corners, and compares each preview's 100x100 `[u,v]` against the **original PhIRE `utils.downscale_image(hr_uv, K=5)`** with absolute tolerance 1e-8. This step uses the locally installed TensorFlow environment (via `utils.py`) but does not load model checkpoints or call HSDS. It performs no model inference, score calculation, training, TFRecord mutation, or locked-region designation. The report is checksum-indexed, atomic and overwrite-protected.

**Run from `~/PhIRE`:**

```bash
unzip Phase7_LXXXVIII_Preview_Parity_20261015.zip -d phase7_lxxxviii_update
( cd phase7_lxxxviii_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_lxxxviii_update/scripts/*.py
PATCH=phase7_lxxxviii_update/patches/phase7_notes_LXXXVII_to_LXXXVIII.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part LX' dataset_generation_and_repair_notes.md | tail -4
```

**Offline preflight** (requires NumPy; does not require TensorFlow):

```bash
mkdir -p logs; set -o pipefail
python3 phase7_lxxxviii_update/scripts/phase7_preview_parity.py --mode preflight \
  2>&1 | tee logs/phase7_lxxxviii_preflight_20261015.log
echo "Preflight exit: $?"
```

**TF downscale parity** (use your same Python interpreter/environment used for `build_example_data_extension_500.py`):

```bash
python3 -c 'import tensorflow as tf; import utils; print("TensorFlow",tf.__version__)'
python3 phase7_lxxxviii_update/scripts/phase7_preview_parity.py --mode parity \
  2>&1 | tee logs/phase7_lxxxviii_parity_20261015.log
echo "Parity exit: $?"
```

If TensorFlow fails to import, stop and report the error before modifying environments. No inference can yet be considered comparable. A successful parity prints all eight `[PARITY] PASS` lines, followed by `[PARITY-GATE] PASS` and exit 0.

**Offline verify:**

```bash
python3 phase7_lxxxviii_update/scripts/phase7_preview_parity.py --mode verify \
  2>&1 | tee logs/phase7_lxxxviii_verify_20261015.log
echo "Verify exit: $?"
OUT=ttk_runs_fixed/topology_finetuning/phase7_lxxxviii_preview_parity_20261015
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/parity_report.md"
```

### Next decision

Inspect the actual geographic corner coordinates, non-overlap, terrain/land attributes and temporal regime suitability, then **predeclare** development vs held-out geographic test regions *before viewing model scores*. Provisional split proposal for future discussion only: development `gulf_inland`/`southwest_borderland`, locked external `upper_midwest`/`pacific_inland`. Don't treat this as an already-frozen split. Subsequent checkpoint-inference pilot should only expose development-region metrics; the other regions must remain unexamined until the final evaluation plan is fixed.

Historical caveat: original PhIRE CNN/GAN trained on broader WTK patches; no proof of source pretraining novelty. LXXXVIII validates native 500x500 MR construction, not full cascade LR→MR→HR or TTK spatial orientation.
