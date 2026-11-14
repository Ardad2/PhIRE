# Phase 6 Part LXXVII — isolated all-168 canonical SR persistence extraction

**Prepared locally; NOT executed on Spark.** The user's LXXVI Spark evidence showed `[METRIC-GATE] PASS` and `[PHASE5E-GATE] PASS`. This workpack does **not** edit any archived Phase 5E, historical CNN/F1, checkpoints, or earlier experimental reports. It generates only **SR** VTI and persistence port-0 files for the three newly trained variants in an isolated directory. The authoritative Phase 5E GT, CNN, and F1 diagrams remain read-only references.

## Design and safeguards

* Three arms: `grad` = `candidateF_grad_pairzero_expanded2688` (CV=0,PERS=0); `cv` = `candidateF_grad_CV_only_expanded2688`; `pers` = `candidateF_grad_PERS_only_expanded2688`.
* Source paths: `data_out/wind_finetune_<method>/{dataGT.npy,dataSR.npy,idx.npy}`. Verifies all 168 ordered indices 0..167 and whole-file equality of each arm's GT and idx to the fixed CNN benchmark. Scans reconstructed arrays for finiteness.
* Requires all 168 canonical Phase5E GT PD port-0 VTUs and pinned sample-0 SHA-256. Reuses, never regenerates GT.
* Writer is the existing `scripts/convert_phire_to_vti.py`, explicit `--samples`, speed, 160×160 top-left crop, `wind_speed`, C-order, `x0=y0=0`.
* Executes TTK `ttkPersistenceDiagramCmd -t 1 -i <vti> -a wind_speed -o <prefix>` **inside** the existing `phire-ttk:latest` image, never invoking the historical wrapper's superseded distance stage.
* `pilot` (sample 0 all 3) -> canonical parser/GUDHI continuity gate -> `full` (0..167 for all 3, resumable) -> read-only `verify`.
* Each arm receives a source/config/code SHA-256 manifest that must match for a rerun. Existing VTIs are rechecked against source values; existing PD VTUs are opened/checked instead of silently overwritten. Full extraction requires `pilot_verified.json`, and verifies its sample-0 PD hashes.
* Output root (new only): `ttk_runs_fixed/topology_finetuning/phase6_lxxvii_canonical_pd_newarms_20261014/` containing subfolders `grad`, `cv`, `pers` with `vti/`, `pd/`, and `input_manifest.json`. The separate `pilot_verified.json` stores sample-0 corrected-GUDHI metrics and checksums.
* Docker uses the **current host user ID**, avoiding root-owned outputs. The read-only preflight mounts the repository read-only. No new GT VTI, merge tree, locality metric or 168-sample distance CSV is produced in this stage.

**Important:** the scripts were syntax-checked and their path/manifest logic tested locally. The real Spark extraction and batch outputs have not been run here. If the existing TTK image or converter CLI has drifted, stop at the first failed pilot rather than modifying frozen source scripts.

## 1. Installation, checksums, documentation

After transferring the ZIP into `~/PhIRE`:

```bash
cd ~/PhIRE
unzip Phase6_LXXVII_NewArms_Canonical_PD_20261014.zip -d phase6_lxxvii_update
( cd phase6_lxxvii_update && sha256sum -c SHA256SUMS )
python3 -m py_compile phase6_lxxvii_update/scripts/*.py
PATCH=phase6_lxxvii_update/patches/phase6_notes_LXXVI_to_LXXVII.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part LX' dataset_generation_and_repair_notes.md | tail -4
```

Apply the Git patch **once** only if notes currently end with Part LXXVI; do not apply with `--reject`, force-reset or overwrite the original notes. No `git add -A`.

## 2. Read-only preflight

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work:ro" -w /work phire-ttk:latest \
  python3 phase6_lxxvii_update/scripts/phase6_canonical_newarms_pd.py \
  --repo-root /work --mode preflight --arm all \
  2>&1 | tee logs/phase6_lxxvii_preflight_20261014.log
printf 'Preflight exit: %s\n' "$?"
```

Expect `[REFERENCE] PASS`, three `[INPUT] PASS`, then `[LXXVII] PASS mode=preflight` and exit 0. `preflight` writes nothing except the host `tee` log.

## 3. Pilot extract: sample 0 for all new SR arms

```bash
cd ~/PhIRE
set -o pipefail
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" -w /work phire-ttk:latest \
  python3 phase6_lxxvii_update/scripts/phase6_canonical_newarms_pd.py \
  --repo-root /work --mode pilot --arm all \
  2>&1 | tee logs/phase6_lxxvii_pilot_20261014.log
printf 'Pilot exit: %s\n' "$?"
```

Expect `[PILOT] PASS` for grad/cv/pers, then exit 0. **Next run the GUDHI gate before full:**

```bash
cd ~/PhIRE
set -o pipefail
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase6_lxxvii_update/scripts/phase6_pilot_pd_gudhi_gate.py \
  2>&1 | tee logs/phase6_lxxvii_pilot_gudhi_20261014.log
printf 'GUDHI pilot exit: %s\n' "$?"
```

This reproduces the existing gradient-only sample-0 PD nonzero-pair multiset and three corrected GUDHI metrics against the frozen Phase 5E GT. It computes analogous **preliminary** metrics for CV/PERS sample 0. It writes `pilot_verified.json` in the new output root only on PASS.

## 4. Full extraction (only after all pilot checks PASS)

Recommended `tmux` because processing 504 SR fields may take substantial time:

```bash
cd ~/PhIRE
tmux new -s phase6_lxxvii_pd
# Inside tmux:
cd ~/PhIRE
mkdir -p logs
set -o pipefail
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" -w /work phire-ttk:latest \
  python3 phase6_lxxvii_update/scripts/phase6_canonical_newarms_pd.py \
  --repo-root /work --mode full --arm all \
  2>&1 | tee logs/phase6_lxxvii_full_20261014.log
printf 'Full extraction exit: %s\n' "$?"
```

For an interrupted run, use the **same** `--mode full --arm all` command. It validates and skips already complete outputs; it stops instead of overwriting malformed existing artifacts or resuming with changed inputs or scripts.

## 5. Final inventory check

```bash
cd ~/PhIRE
set -o pipefail
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work:ro" -w /work phire-ttk:latest \
  python3 phase6_lxxvii_update/scripts/phase6_canonical_newarms_pd.py \
  --repo-root /work --mode verify --arm all \
  2>&1 | tee logs/phase6_lxxvii_verify_20261014.log
printf 'Verify exit: %s\n' "$?"
```

Expected `168` VTIs + `168` PD port-0 VTUs per new arm = **504 each**, and `[LXXVII] PASS mode=verify`. The frozen Phase5E archive should remain 672 PDs. These counts and checks are *targets* until Spark execution supplies evidence.

After full pass, the next stage is a separate, independently audited **504-pair corrected GUDHI distance sweep** against the frozen canonical GT, followed by matched MT and Phase5E locality checks. Do not interpret the old `pd_distance` as corrected bottleneck, or infer new 168-field metrics from the existing 8,568 historical comparison rows.
