# Phase 6: verified training + cheap eval documentation update

This ZIP is for the Spark PhIRE checkout, after Part LXVI has been applied.

1. From the PhIRE root, unzip this package into a new directory, e.g.:
   `unzip Phase6_GradPairZero_Verified_Notes_LXVII_20261013.zip -d phase6_lxvii_update`
2. `git apply --check phase6_lxvii_update/patches/phase6_grad_pairzero_verified_LXVII_incremental.patch`
3. `git apply phase6_lxvii_update/patches/phase6_grad_pairzero_verified_LXVII_incremental.patch`
4. Inspect `git diff -- dataset_generation_and_repair_notes.md`.

The patch appends Part LXVII only. It does not overwrite old sections or publish/push anything.
The standalone doc is included as a convenient separate record.
The new model vs baseline numerical tables have not been supplied; they must be extracted locally from the generated CSV.
