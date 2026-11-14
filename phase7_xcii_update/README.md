# Phase 7 Part XCII — Four-field development-only canonical TTK PD extraction

**Prepared locally; not run on Spark.** This is an extraction-only continuation of verified XCI evaluation-protocol freeze, not a new inference stage, score report, full-field topology run or locked external-test evaluation.

## Scope

Source directory: `ttk_runs_fixed/topology_finetuning/phase7_xc_development_inference_pilot_20261015`.
Input arrays: `reference_gt.npy` and six `pred_{cnn,gan,grad,cv,pers,f1}.npy`. Field order is frozen by XC (Gulf Feb/Aug 2010, Southwest Feb/Aug 2010). Crop coordinates `(y0=0,x0=0,height=160,width=160)`. Scalar speed `sqrt(float32(u)^2 + float32(v)^2)` computed **after** full-500 inference.

Output: `ttk_runs_fixed/topology_finetuning/phase7_xcii_dev_canonical_pd_20261015` containing 28 `.vti` in `vti/`, 28 `_pd_port_0.vtu` in `pd/`, extraction manifest, report, and SHA256SUMS. It never edits XC/XCI, original datasets, model checkpoints or historical topology artifacts.

Extraction requires the *unchanged* original `scripts/convert_phire_to_vti.py`, with its SHA validated against the verified Phase 6 LXXVII input manifest. All generated VTI scalar arrays undergo a strict equality check against the source float32 speed crop. A TTK port-0 diagram must exist and be nonempty, but **numeric PD distances are deferred** to the corrected GUDHI gate in the next phase. Strict checksum checks ensure immutable output. No overwriting, continuation or silent regeneration.

## Install from ZIP

```bash
cd ~/PhIRE
unzip Phase7_XCII_Development_Canonical_PD_20261015.zip -d phase7_xcii_update
( cd phase7_xcii_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase7_xcii_update/scripts/*.py
PATCH=phase7_xcii_update/patches/phase7_notes_XCI_to_XCII.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part ' dataset_generation_and_repair_notes.md | tail -3
```

If the ZIP extraction asks whether to replace any preexisting file, **stop and inspect** rather than overwriting it. The final notes heading should be Part XCII. Do not reapply older notes patches.

## Selftest and read-only preflight (Docker)

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work:ro" -w /work phire-ttk:latest \
  python3 phase7_xcii_update/scripts/phase7_xcii_dev_canonical_pd.py \
  --mode selftest --repo-root /work

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work:ro" -w /work phire-ttk:latest \
  python3 phase7_xcii_update/scripts/phase7_xcii_dev_canonical_pd.py \
  --mode preflight --repo-root /work \
  2>&1 | tee logs/phase7_xcii_preflight_20261015.log
echo "Preflight exit: $?"
```

Require `[SELFTEST] PASS`, `[SOURCE] PASS`, `[PREFLIGHT] PASS` and exit 0. No TTK extraction is performed.

## Run canonical extraction

```bash
cd ~/PhIRE
set -o pipefail
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" -w /work phire-ttk:latest \
  python3 phase7_xcii_update/scripts/phase7_xcii_dev_canonical_pd.py \
  --mode extract --repo-root /work \
  2>&1 | tee logs/phase7_xcii_extract_20261015.log
echo "Extraction exit: $?"
```

Require **28** `[PD]` records and `[EXTRACT-GATE] PASS`; zero locked-region samples. This action runs the existing TTK extraction on 28 small crops and creates only the isolated output directory. On a failed run, read the error and stop; do not overwrite or change the legacy workflow.

## Independent verification (read-only mount)

```bash
cd ~/PhIRE
set -o pipefail
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work:ro" -w /work phire-ttk:latest \
  python3 phase7_xcii_update/scripts/phase7_xcii_dev_canonical_pd.py \
  --mode verify --repo-root /work \
  2>&1 | tee logs/phase7_xcii_verify_20261015.log
echo "Verify exit: $?"
OUT=ttk_runs_fixed/topology_finetuning/phase7_xcii_dev_canonical_pd_20261015
( cd "$OUT" && sha256sum -c SHA256SUMS )
cat "$OUT/extraction_report.md"
```

Require `[VERIFY-GATE] PASS`, exit 0, 58 SHA-256 entries `OK` (28 VTI + 28 PD + 2 reports). These are extraction validations, **not** GUDHI or spatial-locality findings.

## Implementation test status

Python syntax and VTK orientation selftest passed locally. A reduced mock integration run completed extraction, independent validation and deliberate file-tampering rejection. Live Spark input hashes and TTK CLI are not available locally and must pass on the Spark host. A larger mock test was stopped by a runtime limit; no claim of full 28-file local integration success is made.
