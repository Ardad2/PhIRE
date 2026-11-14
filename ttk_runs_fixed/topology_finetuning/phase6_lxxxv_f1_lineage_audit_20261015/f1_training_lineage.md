# Phase 6 LXXXV — F1 historical training-lineage evidence

**Status: NOT CLOSED.** This is an evidence inventory and a targeted excerpting of logs. It does not certify identical random seeds, optimizer state or code versions.

## Proven input files and current source fingerprints

| Role | Exists | Bytes | SHA256 or reason |
|---|---|---:|---|
| training TFRecord (`example_data_topology_expanded_2688/wind_MR-HR.tfrecord`) | True | 11182526080 | `SKIPPED (large file; --deep-hash to include)` |
| evaluation TFRecord (`example_data_fixed/wind_MR-HR.tfrecord`) | True | 698907760 | `SKIPPED (large file; --deep-hash to include)` |
| fixed E2 targets (`ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded2688_constraints/ttk_pd_critical_pairs_gtvalues.npz`) | True | 4883706 | `7909fdfefb2dd2c3574feffacd4e316045904776eeaa70cdb5ec72941a47a43d` |
| original CNN init index (`models/wind_mr-hr/trained_cnn/cnn.index`) | True | 2830 | `e5ee08bd24450fb95987867bae76c00e53077de6dbd7f76de579e5316bd20ce2` |
| current base runner (`scripts/run_candidateF_expanded2688_finetune.py`) | True | 71003 | `7ac9453799044985a0d79a62729036979eb4b7e12e0033fdb5b0ec4d91ba726a` |
| current ablation wrapper (`scripts/run_candidateF_cv_pers_ablation.py`) | True | 4390 | `78666dad7d608f622fd564798205fcd3515ad2c44e1c17dac17947a02e653412` |
| F1 evaluation report (`docs/topology_finetuning_candidateF_grad_E2_low_expanded2688_eval.md`) | True | 9102 | `2e71512450eec15f5057d700b656e313a50066441c5bca599ba1c8411a252b07` |

## Full benchmark F1 versus gradient-only array identity

- `idx.npy`: exact SHA256 equality = **True**.
- `dataGT.npy`: exact SHA256 equality = **True**.

## Checkpoint inventory

### f1
- `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_E2_low_expanded2688/cnn00001/cnn.index`: PRESENT, index SHA256=`ee7404ebfc83b7042cd1161bc4ec587c3e4871e619082d9dfc115fd25a828c76`; data shards=1
  - `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_E2_low_expanded2688/cnn00001/cnn.data-00000-of-00001` bytes=8576776 SHA256=`4767e116c3cb76ae40a5cc757084747602651e3ce743019f3fa73bdf552a0d9f`
- `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_E2_low_expanded2688/cnn00002/cnn.index`: PRESENT, index SHA256=`59c1fe7075f390eaba47fc5f0367533a2cf8731da3de052844964302cd20cd97`; data shards=1
  - `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_E2_low_expanded2688/cnn00002/cnn.data-00000-of-00001` bytes=8576776 SHA256=`b4777ba2634a40e5411ba9fc08e9fb9cea9f9cf5ef2aab748b54c38961ddbd37`
- `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_E2_low_expanded2688/cnn00003/cnn.index`: PRESENT, index SHA256=`5c0e9bc6f0243df50edc0cab4e953f1e1b5438001ba00e5a8180a9708d825d91`; data shards=1
  - `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_E2_low_expanded2688/cnn00003/cnn.data-00000-of-00001` bytes=8576776 SHA256=`dc4a58d28535c63e21147b0033859d757f29d403d7dd7d059348a2b893d065f0`
- `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_E2_low_expanded2688/cnn/cnn.index`: PRESENT, index SHA256=`5c0e9bc6f0243df50edc0cab4e953f1e1b5438001ba00e5a8180a9708d825d91`; data shards=1
  - `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_E2_low_expanded2688/cnn/cnn.data-00000-of-00001` bytes=8576776 SHA256=`dc4a58d28535c63e21147b0033859d757f29d403d7dd7d059348a2b893d065f0`
### grad
- `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688/cnn00001/cnn.index`: PRESENT, index SHA256=`beced36199eb7a9a7be721cc89b62e86a85c9b7389dc75fe0c8d0006011c84eb`; data shards=1
  - `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688/cnn00001/cnn.data-00000-of-00001` bytes=8576776 SHA256=`75b06253c8db2d32f7636618139d5bd47b6acfb1f508d0b3cca5f159603c8ae4`
- `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688/cnn00002/cnn.index`: PRESENT, index SHA256=`fc83c8a329c86c926c4aab871552b4060f09970a7259ef2b093491d80b49d3a4`; data shards=1
  - `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688/cnn00002/cnn.data-00000-of-00001` bytes=8576776 SHA256=`43f55d599bc7686cf821134cc475f378ba53af6e3bc927f9fa6bb60f592b1786`
- `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688/cnn00003/cnn.index`: PRESENT, index SHA256=`7dbbbb6bcfe7f75d4551f42c70994e05f1d3acdbc2912cc1d99e06c4b12c3b40`; data shards=1
  - `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688/cnn00003/cnn.data-00000-of-00001` bytes=8576776 SHA256=`454d09cf32e409e1f8f6037046327f54f7930ffb7db2b7fac831d7acefd3118a`
- `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688/cnn/cnn.index`: PRESENT, index SHA256=`7dbbbb6bcfe7f75d4551f42c70994e05f1d3acdbc2912cc1d99e06c4b12c3b40`; data shards=1
  - `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688/cnn/cnn.data-00000-of-00001` bytes=8576776 SHA256=`454d09cf32e409e1f8f6037046327f54f7930ffb7db2b7fac831d7acefd3118a`
### cv
- `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_CV_only_expanded2688/cnn00001/cnn.index`: PRESENT, index SHA256=`520a9f8ad37f12f73b00bcd841e967efceea567c8c6844650f19703d06e7a1c1`; data shards=1
  - `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_CV_only_expanded2688/cnn00001/cnn.data-00000-of-00001` bytes=8576776 SHA256=`98fee207d8ea8bebd6ec581112e150164b9c32d4d5d727671171b6d8921885cb`
- `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_CV_only_expanded2688/cnn00002/cnn.index`: PRESENT, index SHA256=`9c3a7c8e54a80a39827a3c4e9c50f397ce6f4c5ad1c1ce0f029d7ab5be67fd05`; data shards=1
  - `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_CV_only_expanded2688/cnn00002/cnn.data-00000-of-00001` bytes=8576776 SHA256=`f12ba8fa158e95535c63bbb203b4bd51d17d1c5c29bfdd9f5d975fd484f34592`
- `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_CV_only_expanded2688/cnn00003/cnn.index`: PRESENT, index SHA256=`2747e8e49767771ce32b3e6f5283b65eb337052e1f72ca9f40a838047b465f58`; data shards=1
  - `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_CV_only_expanded2688/cnn00003/cnn.data-00000-of-00001` bytes=8576776 SHA256=`d45cc6cc75c3a53170eccd0459b2017e1b42b5d7cbfea4681fff1a282624d561`
- `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_CV_only_expanded2688/cnn/cnn.index`: PRESENT, index SHA256=`2747e8e49767771ce32b3e6f5283b65eb337052e1f72ca9f40a838047b465f58`; data shards=1
  - `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_CV_only_expanded2688/cnn/cnn.data-00000-of-00001` bytes=8576776 SHA256=`d45cc6cc75c3a53170eccd0459b2017e1b42b5d7cbfea4681fff1a282624d561`
### pers
- `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_PERS_only_expanded2688/cnn00001/cnn.index`: PRESENT, index SHA256=`bc0986883349b7247f9f4460e9ea0025b81d254328992b0b3d57b7aac3e8bf32`; data shards=1
  - `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_PERS_only_expanded2688/cnn00001/cnn.data-00000-of-00001` bytes=8576776 SHA256=`e7ad8cf993859eb4be5984eff2bd863762e941855c8128013f40030c60160618`
- `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_PERS_only_expanded2688/cnn00002/cnn.index`: PRESENT, index SHA256=`b320c5d8e708320df8bcdd1e536c98663c73039f08824dc24758a1e72fcc3431`; data shards=1
  - `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_PERS_only_expanded2688/cnn00002/cnn.data-00000-of-00001` bytes=8576776 SHA256=`9d797ced7b60e617086935bb1ff5f16e1487da635d35e1702b023cbd9cacc992`
- `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_PERS_only_expanded2688/cnn00003/cnn.index`: PRESENT, index SHA256=`83664c3f784b1caa5d2f74575ff294f088d0e3eb2b181a498be1a1e29c1f38a5`; data shards=1
  - `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_PERS_only_expanded2688/cnn00003/cnn.data-00000-of-00001` bytes=8576776 SHA256=`2630af4c32c95e71d13457b98fca4e1afb3d9c1dd50f30215b0e2f75e7a48829`
- `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_PERS_only_expanded2688/cnn/cnn.index`: PRESENT, index SHA256=`83664c3f784b1caa5d2f74575ff294f088d0e3eb2b181a498be1a1e29c1f38a5`; data shards=1
  - `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_PERS_only_expanded2688/cnn/cnn.data-00000-of-00001` bytes=8576776 SHA256=`2630af4c32c95e71d13457b98fca4e1afb3d9c1dd50f30215b0e2f75e7a48829`

- F1 final-vs-epoch3 checkpoint **index** SHA256 match: `True` (index alone is not the full checkpoint).

## Targeted saved-log evidence

Excerpt lines are for manual validation; a missing excerpt does not establish a contrary fact. Each file SHA256 is stored in the JSON manifest.

### f1_train: `logs/wind_finetune_candidateF_grad_E2_low_expanded2688.log` (present)
**launch / branch / environment:** 3 matching lines (first 3 shown)
- L11171: `--candidate-name candidateF_grad_E2_low_expanded2688 \`
- L11172: `--candidate-dir  data_out/wind_finetune_candidateF_grad_E2_low_expanded2688 \`
- L11183: `--method     candidateF_grad_E2_low_expanded2688 \`
**loss coefficients:** 2040 matching lines (first 12 shown)
- L1: `[log] Canonical training log opened: /home/adadhwal/PhIRE/logs/wind_finetune_candidateF_grad_E2_low_expanded2688.log`
- L15: `L_ttkcv                      16.579762`
- L16: `L_ttkpers                    17.197647`
- L31: `L_ttkcv:               11.230244`
- L32: `L_ttkpers:             10.550603`
- L42: `L_ttkcv:               10.974796`
- L43: `L_ttkpers:             12.556776`
- L53: `L_ttkcv:               9.432949`
- L54: `L_ttkpers:             10.251405`
- L64: `L_ttkcv:               9.387091`
- L65: `L_ttkpers:             8.929781`
- L75: `L_ttkcv:               12.040817`
**training hyperparameters:** 6 matching lines (first 6 shown)
- L22: `Epoch: 1`
- L3720: `Epoch total loss (with TTK) = 0.08181`
- L3722: `Epoch: 2`
- L7420: `Epoch total loss (with TTK) = 0.07055`
- L7422: `Epoch: 3`
- L11120: `Epoch total loss (with TTK) = 0.06582`
**input paths / constraints / checkpoint restore:** 24 matching lines (first 12 shown)
- L1: `[log] Canonical training log opened: /home/adadhwal/PhIRE/logs/wind_finetune_candidateF_grad_E2_low_expanded2688.log`
- L4: `Loading pretrained checkpoint … Done.`
- L21: `Phase 1: Training on 2688 expanded samples …`
- L3719: `Checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_E2_low_expanded2688/cnn00001/cnn`
- L7419: `Checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_E2_low_expanded2688/cnn00002/cnn`
- L11119: `Checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_E2_low_expanded2688/cnn00003/cnn`
- L11122: `Final checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_E2_low_expanded2688/cnn/cnn`
- L11126: `Fine-tuning complete. Final checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_E2_low_expanded2688/cnn/cnn`
- L11131: `Checkpoint : models_fixed/topology_finetuning/wind_finetune_candidateF_grad_E2_low_expanded2688/cnn/cnn`
- L11132: `Eval data  : example_data_fixed/wind_MR-HR.tfrecord`
- L11133: `Output dir : data_out/wind_finetune_candidateF_grad_E2_low_expanded2688`
- L11160: `Final checkpoint : models_fixed/topology_finetuning/wind_finetune_candidateF_grad_E2_low_expanded2688/cnn/cnn`
**completion / validation:** 2 matching lines (first 2 shown)
- L11155: `[validate] Post-inference validation complete: shapes, exact idx order, finiteness, and (where the baseline was available) GT/IN alignment all passed.`
- L11158: `Run complete.`

### f1_batch: `logs/candidateF_grad_E2_low_expanded2688_batch.log` (present)
**launch / branch / environment:** 11 matching lines (first 11 shown)
- L23: `Traceback (most recent call last):  File "/home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py", line 1453, in <module>`
- L25: `File "/home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py", line 510, in main`
- L99: `Traceback (most recent call last):  File "/home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py", line 1453, in <module>`
- L101: `File "/home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py", line 510, in main`
- L166: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:591: tf_record_iterator (from tensorflow.python.lib.io.tf_record) is deprecated and will be removed in a future version.`
- L176: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:1138: DatasetV1.output_types (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.`
- L179: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:1138: DatasetV1.output_shapes (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.`
- L11438: `--candidate-name candidateF_grad_E2_low_expanded2688 \`
- L11439: `--candidate-dir  data_out/wind_finetune_candidateF_grad_E2_low_expanded2688 \`
- L11450: `--method     candidateF_grad_E2_low_expanded2688 \`
- L11502: `--method     candidateF_grad_E2_low_expanded2688 \`
**loss coefficients:** 2075 matching lines (first 12 shown)
- L5: `Variants to run    : grad_e2_low`
- L11: `# Variant: grad_e2_low`
- L192: `Candidate F expanded-2688 fine-tuning: variant='grad_e2_low' (method='candidateF_grad_E2_low_expanded2688')`
- L197: `Output model dir  : models_fixed/topology_finetuning/wind_finetune_candidateF_grad_E2_low_expanded2688`
- L198: `Inference output  : data_out/wind_finetune_candidateF_grad_E2_low_expanded2688`
- L199: `Cheap-eval dir    : ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_eval`
- L200: `Topology VTI dir  : ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology_vti`
- L201: `Topology out dir  : ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology`
- L202: `Log file          : /home/adadhwal/PhIRE/logs/wind_finetune_candidateF_grad_E2_low_expanded2688.log`
- L205: `lambda_uv       = 1.0  (always on; normalized [u, v])`
- L206: `lambda_speed    = 0.0`
- L207: `lambda_grad     = 0.05`
**training hyperparameters:** 11 matching lines (first 11 shown)
- L28: `importlib.import_module("keras.src.optimizers")`
- L104: `importlib.import_module("keras.src.optimizers")`
- L217: `learning_rate   = 1e-05`
- L218: `N_epochs        = 3`
- L221: `batch_size      = 4`
- L283: `Epoch: 1`
- L3983: `Epoch total loss (with TTK) = 0.08181`
- L3985: `Epoch: 2`
- L7685: `Epoch total loss (with TTK) = 0.07055`
- L7687: `Epoch: 3`
- L11387: `Epoch total loss (with TTK) = 0.06582`
**input paths / constraints / checkpoint restore:** 67 matching lines (first 12 shown)
- L2: `Candidate F expanded-2688 batch`
- L23: `Traceback (most recent call last):  File "/home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py", line 1453, in <module>`
- L25: `File "/home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py", line 510, in main`
- L99: `Traceback (most recent call last):  File "/home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py", line 1453, in <module>`
- L101: `File "/home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py", line 510, in main`
- L166: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:591: tf_record_iterator (from tensorflow.python.lib.io.tf_record) is deprecated and will be removed in a future version.`
- L176: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:1138: DatasetV1.output_types (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.`
- L179: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:1138: DatasetV1.output_shapes (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.`
- L192: `Candidate F expanded-2688 fine-tuning: variant='grad_e2_low' (method='candidateF_grad_E2_low_expanded2688')`
- L194: `Training data     : example_data_topology_expanded_2688/wind_MR-HR.tfrecord`
- L195: `Evaluation data   : example_data_fixed/wind_MR-HR.tfrecord`
- L196: `Source checkpoint : models/wind_mr-hr/trained_cnn/cnn`
**completion / validation:** 4 matching lines (first 4 shown)
- L235: `Confirmed: 2688-sample training set matches candidateC_expanded2688 / candidateB_expanded2688 / the completed E2-low-2688 scripts.`
- L256: `--- Constraints NPZ metadata validation: PASSED ---`
- L11422: `[validate] Post-inference validation complete: shapes, exact idx order, finiteness, and (where the baseline was available) GT/IN alignment all passed.`
- L11425: `Run complete.`

### grad_train: `logs/wind_finetune_candidateF_grad_pairzero_expanded2688.log` (present)
**launch / branch / environment:** 3 matching lines (first 3 shown)
- L11171: `--candidate-name candidateF_grad_pairzero_expanded2688 \`
- L11172: `--candidate-dir  data_out/wind_finetune_candidateF_grad_pairzero_expanded2688 \`
- L11183: `--method     candidateF_grad_pairzero_expanded2688 \`
**loss coefficients:** 2039 matching lines (first 12 shown)
- L1: `[log] Canonical training log opened: /home/adadhwal/PhIRE/logs/wind_finetune_candidateF_grad_pairzero_expanded2688.log`
- L15: `L_ttkcv                      18.656094`
- L16: `L_ttkpers                    18.093243`
- L31: `L_ttkcv:               11.574254`
- L32: `L_ttkpers:             12.626558`
- L42: `L_ttkcv:               11.169500`
- L43: `L_ttkpers:             13.127578`
- L53: `L_ttkcv:               14.601936`
- L54: `L_ttkpers:             15.260389`
- L64: `L_ttkcv:               15.547695`
- L65: `L_ttkpers:             16.009600`
- L75: `L_ttkcv:               13.997806`
**training hyperparameters:** 6 matching lines (first 6 shown)
- L22: `Epoch: 1`
- L3720: `Epoch total loss (with TTK) = 0.03337`
- L3722: `Epoch: 2`
- L7420: `Epoch total loss (with TTK) = 0.02937`
- L7422: `Epoch: 3`
- L11120: `Epoch total loss (with TTK) = 0.02825`
**input paths / constraints / checkpoint restore:** 25 matching lines (first 12 shown)
- L1: `[log] Canonical training log opened: /home/adadhwal/PhIRE/logs/wind_finetune_candidateF_grad_pairzero_expanded2688.log`
- L4: `Loading pretrained checkpoint … Done.`
- L21: `Phase 1: Training on 2688 expanded samples …`
- L2769: `L_crit:                3.268838`
- L3719: `Checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688/cnn00001/cnn`
- L7419: `Checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688/cnn00002/cnn`
- L11119: `Checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688/cnn00003/cnn`
- L11122: `Final checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688/cnn/cnn`
- L11126: `Fine-tuning complete. Final checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688/cnn/cnn`
- L11131: `Checkpoint : models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688/cnn/cnn`
- L11132: `Eval data  : example_data_fixed/wind_MR-HR.tfrecord`
- L11133: `Output dir : data_out/wind_finetune_candidateF_grad_pairzero_expanded2688`
**completion / validation:** 2 matching lines (first 2 shown)
- L11155: `[validate] Post-inference validation complete: shapes, exact idx order, finiteness, and (where the baseline was available) GT/IN alignment all passed.`
- L11158: `Run complete.`

### cv_train: `logs/wind_finetune_candidateF_grad_CV_only_expanded2688.log` (present)
**launch / branch / environment:** 3 matching lines (first 3 shown)
- L11171: `--candidate-name candidateF_grad_CV_only_expanded2688 \`
- L11172: `--candidate-dir  data_out/wind_finetune_candidateF_grad_CV_only_expanded2688 \`
- L11183: `--method     candidateF_grad_CV_only_expanded2688 \`
**loss coefficients:** 2040 matching lines (first 12 shown)
- L1: `[log] Canonical training log opened: /home/adadhwal/PhIRE/logs/wind_finetune_candidateF_grad_CV_only_expanded2688.log`
- L15: `L_ttkcv                      12.328762`
- L16: `L_ttkpers                    12.232632`
- L31: `L_ttkcv:               10.320076`
- L32: `L_ttkpers:             12.619616`
- L42: `L_ttkcv:               17.011951`
- L43: `L_ttkpers:             19.364162`
- L53: `L_ttkcv:               15.494966`
- L54: `L_ttkpers:             15.877012`
- L64: `L_ttkcv:               12.184373`
- L65: `L_ttkpers:             12.190518`
- L75: `L_ttkcv:               10.615253`
**training hyperparameters:** 6 matching lines (first 6 shown)
- L22: `Epoch: 1`
- L3720: `Epoch total loss (with TTK) = 0.06436`
- L3722: `Epoch: 2`
- L7420: `Epoch total loss (with TTK) = 0.05508`
- L7422: `Epoch: 3`
- L11120: `Epoch total loss (with TTK) = 0.05134`
**input paths / constraints / checkpoint restore:** 28 matching lines (first 12 shown)
- L1: `[log] Canonical training log opened: /home/adadhwal/PhIRE/logs/wind_finetune_candidateF_grad_CV_only_expanded2688.log`
- L4: `Loading pretrained checkpoint … Done.`
- L21: `Phase 1: Training on 2688 expanded samples …`
- L2098: `L_crit:                2.662688`
- L3719: `Checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_CV_only_expanded2688/cnn00001/cnn`
- L7419: `Checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_CV_only_expanded2688/cnn00002/cnn`
- L7966: `L_grad:                0.352688`
- L7978: `L_wpd:                 158.026886`
- L9332: `L_levelset:            0.026882`
- L11119: `Checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_CV_only_expanded2688/cnn00003/cnn`
- L11122: `Final checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_CV_only_expanded2688/cnn/cnn`
- L11126: `Fine-tuning complete. Final checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_CV_only_expanded2688/cnn/cnn`
**completion / validation:** 2 matching lines (first 2 shown)
- L11155: `[validate] Post-inference validation complete: shapes, exact idx order, finiteness, and (where the baseline was available) GT/IN alignment all passed.`
- L11158: `Run complete.`

### pers_train: `logs/wind_finetune_candidateF_grad_PERS_only_expanded2688.log` (present)
**launch / branch / environment:** 3 matching lines (first 3 shown)
- L11171: `--candidate-name candidateF_grad_PERS_only_expanded2688 \`
- L11172: `--candidate-dir  data_out/wind_finetune_candidateF_grad_PERS_only_expanded2688 \`
- L11183: `--method     candidateF_grad_PERS_only_expanded2688 \`
**loss coefficients:** 2040 matching lines (first 12 shown)
- L1: `[log] Canonical training log opened: /home/adadhwal/PhIRE/logs/wind_finetune_candidateF_grad_PERS_only_expanded2688.log`
- L15: `L_ttkcv                      13.815626`
- L16: `L_ttkpers                    16.663092`
- L31: `L_ttkcv:               10.343120`
- L32: `L_ttkpers:             10.855334`
- L42: `L_ttkcv:               12.884408`
- L43: `L_ttkpers:             13.117381`
- L53: `L_ttkcv:               21.251972`
- L54: `L_ttkpers:             21.345247`
- L64: `L_ttkcv:               6.866954`
- L65: `L_ttkpers:             7.262928`
- L75: `L_ttkcv:               11.466641`
**training hyperparameters:** 6 matching lines (first 6 shown)
- L22: `Epoch: 1`
- L3720: `Epoch total loss (with TTK) = 0.05393`
- L3722: `Epoch: 2`
- L7420: `Epoch total loss (with TTK) = 0.04800`
- L7422: `Epoch: 3`
- L11120: `Epoch total loss (with TTK) = 0.04568`
**input paths / constraints / checkpoint restore:** 27 matching lines (first 12 shown)
- L1: `[log] Canonical training log opened: /home/adadhwal/PhIRE/logs/wind_finetune_candidateF_grad_PERS_only_expanded2688.log`
- L4: `Loading pretrained checkpoint … Done.`
- L21: `Phase 1: Training on 2688 expanded samples …`
- L2592: `L_levelset:            0.026888`
- L3719: `Checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_PERS_only_expanded2688/cnn00001/cnn`
- L5535: `L_ttkcv:               11.026887`
- L7419: `Checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_PERS_only_expanded2688/cnn00002/cnn`
- L8373: `L_grad:                0.268896`
- L11119: `Checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_PERS_only_expanded2688/cnn00003/cnn`
- L11122: `Final checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_PERS_only_expanded2688/cnn/cnn`
- L11126: `Fine-tuning complete. Final checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_PERS_only_expanded2688/cnn/cnn`
- L11131: `Checkpoint : models_fixed/topology_finetuning/wind_finetune_candidateF_grad_PERS_only_expanded2688/cnn/cnn`
**completion / validation:** 2 matching lines (first 2 shown)
- L11155: `[validate] Post-inference validation complete: shapes, exact idx order, finiteness, and (where the baseline was available) GT/IN alignment all passed.`
- L11158: `Run complete.`

### grad_console: `logs/phase6_grad_pairzero_console.log` (present)
**launch / branch / environment:** 13 matching lines (first 12 shown)
- L11: `Traceback (most recent call last):  File "/home/adadhwal/PhIRE/scripts/run_candidateF_cv_pers_ablation.py", line 97, in <module>`
- L13: `File "/home/adadhwal/PhIRE/scripts/run_candidateF_cv_pers_ablation.py", line 93, in main`
- L15: `File "/home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py", line 510, in main`
- L89: `Traceback (most recent call last):  File "/home/adadhwal/PhIRE/scripts/run_candidateF_cv_pers_ablation.py", line 97, in <module>`
- L91: `File "/home/adadhwal/PhIRE/scripts/run_candidateF_cv_pers_ablation.py", line 93, in main`
- L93: `File "/home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py", line 510, in main`
- L158: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:591: tf_record_iterator (from tensorflow.python.lib.io.tf_record) is deprecated and will be removed in a future version.`
- L169: `Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, 'export AUTOGRAPH_VERBOSITY=10') and attach the full output.`
- L172: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:1138: DatasetV1.output_types (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.`
- L175: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:1138: DatasetV1.output_shapes (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.`
- L11434: `--candidate-name candidateF_grad_pairzero_expanded2688 \`
- L11435: `--candidate-dir  data_out/wind_finetune_candidateF_grad_pairzero_expanded2688 \`
**loss coefficients:** 2055 matching lines (first 12 shown)
- L188: `Candidate F expanded-2688 fine-tuning: variant='grad_pair_zero' (method='candidateF_grad_pairzero_expanded2688')`
- L193: `Output model dir  : models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688`
- L194: `Inference output  : data_out/wind_finetune_candidateF_grad_pairzero_expanded2688`
- L195: `Cheap-eval dir    : ttk_runs_fixed/topology_finetuning/candidateF_grad_pairzero_expanded2688_eval`
- L196: `Topology VTI dir  : ttk_runs_fixed/topology_finetuning/candidateF_grad_pairzero_expanded2688_topology_vti`
- L197: `Topology out dir  : ttk_runs_fixed/topology_finetuning/candidateF_grad_pairzero_expanded2688_topology`
- L198: `Log file          : /home/adadhwal/PhIRE/logs/wind_finetune_candidateF_grad_pairzero_expanded2688.log`
- L201: `lambda_uv       = 1.0  (always on; normalized [u, v])`
- L202: `lambda_speed    = 0.0`
- L203: `lambda_grad     = 0.05`
- L205: `lambda_levelset = 0.0`
- L206: `lambda_crit     = 0.0  (DISABLED)`
**training hyperparameters:** 11 matching lines (first 11 shown)
- L18: `importlib.import_module("keras.src.optimizers")`
- L96: `importlib.import_module("keras.src.optimizers")`
- L213: `learning_rate   = 1e-05`
- L214: `N_epochs        = 3`
- L217: `batch_size      = 4`
- L279: `Epoch: 1`
- L3979: `Epoch total loss (with TTK) = 0.03337`
- L3981: `Epoch: 2`
- L7681: `Epoch total loss (with TTK) = 0.02937`
- L7683: `Epoch: 3`
- L11383: `Epoch total loss (with TTK) = 0.02825`
**input paths / constraints / checkpoint restore:** 51 matching lines (first 12 shown)
- L15: `File "/home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py", line 510, in main`
- L93: `File "/home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py", line 510, in main`
- L158: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:591: tf_record_iterator (from tensorflow.python.lib.io.tf_record) is deprecated and will be removed in a future version.`
- L172: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:1138: DatasetV1.output_types (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.`
- L175: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:1138: DatasetV1.output_shapes (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.`
- L188: `Candidate F expanded-2688 fine-tuning: variant='grad_pair_zero' (method='candidateF_grad_pairzero_expanded2688')`
- L190: `Training data     : example_data_topology_expanded_2688/wind_MR-HR.tfrecord`
- L191: `Evaluation data   : example_data_fixed/wind_MR-HR.tfrecord`
- L192: `Source checkpoint : models/wind_mr-hr/trained_cnn/cnn`
- L193: `Output model dir  : models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688`
- L194: `Inference output  : data_out/wind_finetune_candidateF_grad_pairzero_expanded2688`
- L195: `Cheap-eval dir    : ttk_runs_fixed/topology_finetuning/candidateF_grad_pairzero_expanded2688_eval`
**completion / validation:** 4 matching lines (first 4 shown)
- L231: `Confirmed: 2688-sample training set matches candidateC_expanded2688 / candidateB_expanded2688 / the completed E2-low-2688 scripts.`
- L252: `--- Constraints NPZ metadata validation: PASSED ---`
- L11418: `[validate] Post-inference validation complete: shapes, exact idx order, finiteness, and (where the baseline was available) GT/IN alignment all passed.`
- L11421: `Run complete.`

### cv_console: `logs/phase6_grad_cv_only_console.log` (present)
**launch / branch / environment:** 13 matching lines (first 12 shown)
- L11: `Traceback (most recent call last):  File "/home/adadhwal/PhIRE/scripts/run_candidateF_cv_pers_ablation.py", line 97, in <module>`
- L13: `File "/home/adadhwal/PhIRE/scripts/run_candidateF_cv_pers_ablation.py", line 93, in main`
- L15: `File "/home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py", line 510, in main`
- L89: `Traceback (most recent call last):  File "/home/adadhwal/PhIRE/scripts/run_candidateF_cv_pers_ablation.py", line 97, in <module>`
- L91: `File "/home/adadhwal/PhIRE/scripts/run_candidateF_cv_pers_ablation.py", line 93, in main`
- L93: `File "/home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py", line 510, in main`
- L158: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:591: tf_record_iterator (from tensorflow.python.lib.io.tf_record) is deprecated and will be removed in a future version.`
- L169: `Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, 'export AUTOGRAPH_VERBOSITY=10') and attach the full output.`
- L172: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:1138: DatasetV1.output_types (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.`
- L175: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:1138: DatasetV1.output_shapes (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.`
- L11434: `--candidate-name candidateF_grad_CV_only_expanded2688 \`
- L11435: `--candidate-dir  data_out/wind_finetune_candidateF_grad_CV_only_expanded2688 \`
**loss coefficients:** 2056 matching lines (first 12 shown)
- L188: `Candidate F expanded-2688 fine-tuning: variant='grad_cv_only' (method='candidateF_grad_CV_only_expanded2688')`
- L193: `Output model dir  : models_fixed/topology_finetuning/wind_finetune_candidateF_grad_CV_only_expanded2688`
- L194: `Inference output  : data_out/wind_finetune_candidateF_grad_CV_only_expanded2688`
- L195: `Cheap-eval dir    : ttk_runs_fixed/topology_finetuning/candidateF_grad_CV_only_expanded2688_eval`
- L196: `Topology VTI dir  : ttk_runs_fixed/topology_finetuning/candidateF_grad_CV_only_expanded2688_topology_vti`
- L197: `Topology out dir  : ttk_runs_fixed/topology_finetuning/candidateF_grad_CV_only_expanded2688_topology`
- L198: `Log file          : /home/adadhwal/PhIRE/logs/wind_finetune_candidateF_grad_CV_only_expanded2688.log`
- L201: `lambda_uv       = 1.0  (always on; normalized [u, v])`
- L202: `lambda_speed    = 0.0`
- L203: `lambda_grad     = 0.05`
- L205: `lambda_levelset = 0.0`
- L206: `lambda_crit     = 0.0  (DISABLED)`
**training hyperparameters:** 11 matching lines (first 11 shown)
- L18: `importlib.import_module("keras.src.optimizers")`
- L96: `importlib.import_module("keras.src.optimizers")`
- L213: `learning_rate   = 1e-05`
- L214: `N_epochs        = 3`
- L217: `batch_size      = 4`
- L279: `Epoch: 1`
- L3979: `Epoch total loss (with TTK) = 0.06436`
- L3981: `Epoch: 2`
- L7681: `Epoch total loss (with TTK) = 0.05508`
- L7683: `Epoch: 3`
- L11383: `Epoch total loss (with TTK) = 0.05134`
**input paths / constraints / checkpoint restore:** 54 matching lines (first 12 shown)
- L15: `File "/home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py", line 510, in main`
- L93: `File "/home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py", line 510, in main`
- L158: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:591: tf_record_iterator (from tensorflow.python.lib.io.tf_record) is deprecated and will be removed in a future version.`
- L172: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:1138: DatasetV1.output_types (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.`
- L175: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:1138: DatasetV1.output_shapes (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.`
- L188: `Candidate F expanded-2688 fine-tuning: variant='grad_cv_only' (method='candidateF_grad_CV_only_expanded2688')`
- L190: `Training data     : example_data_topology_expanded_2688/wind_MR-HR.tfrecord`
- L191: `Evaluation data   : example_data_fixed/wind_MR-HR.tfrecord`
- L192: `Source checkpoint : models/wind_mr-hr/trained_cnn/cnn`
- L193: `Output model dir  : models_fixed/topology_finetuning/wind_finetune_candidateF_grad_CV_only_expanded2688`
- L194: `Inference output  : data_out/wind_finetune_candidateF_grad_CV_only_expanded2688`
- L195: `Cheap-eval dir    : ttk_runs_fixed/topology_finetuning/candidateF_grad_CV_only_expanded2688_eval`
**completion / validation:** 4 matching lines (first 4 shown)
- L231: `Confirmed: 2688-sample training set matches candidateC_expanded2688 / candidateB_expanded2688 / the completed E2-low-2688 scripts.`
- L252: `--- Constraints NPZ metadata validation: PASSED ---`
- L11418: `[validate] Post-inference validation complete: shapes, exact idx order, finiteness, and (where the baseline was available) GT/IN alignment all passed.`
- L11421: `Run complete.`

### pers_console: `logs/phase6_grad_pers_only_console.log` (present)
**launch / branch / environment:** 13 matching lines (first 12 shown)
- L11: `Traceback (most recent call last):  File "/home/adadhwal/PhIRE/scripts/run_candidateF_cv_pers_ablation.py", line 97, in <module>`
- L13: `File "/home/adadhwal/PhIRE/scripts/run_candidateF_cv_pers_ablation.py", line 93, in main`
- L15: `File "/home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py", line 510, in main`
- L89: `Traceback (most recent call last):  File "/home/adadhwal/PhIRE/scripts/run_candidateF_cv_pers_ablation.py", line 97, in <module>`
- L91: `File "/home/adadhwal/PhIRE/scripts/run_candidateF_cv_pers_ablation.py", line 93, in main`
- L93: `File "/home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py", line 510, in main`
- L158: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:591: tf_record_iterator (from tensorflow.python.lib.io.tf_record) is deprecated and will be removed in a future version.`
- L169: `Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, 'export AUTOGRAPH_VERBOSITY=10') and attach the full output.`
- L172: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:1138: DatasetV1.output_types (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.`
- L175: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:1138: DatasetV1.output_shapes (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.`
- L11434: `--candidate-name candidateF_grad_PERS_only_expanded2688 \`
- L11435: `--candidate-dir  data_out/wind_finetune_candidateF_grad_PERS_only_expanded2688 \`
**loss coefficients:** 2056 matching lines (first 12 shown)
- L188: `Candidate F expanded-2688 fine-tuning: variant='grad_pers_only' (method='candidateF_grad_PERS_only_expanded2688')`
- L193: `Output model dir  : models_fixed/topology_finetuning/wind_finetune_candidateF_grad_PERS_only_expanded2688`
- L194: `Inference output  : data_out/wind_finetune_candidateF_grad_PERS_only_expanded2688`
- L195: `Cheap-eval dir    : ttk_runs_fixed/topology_finetuning/candidateF_grad_PERS_only_expanded2688_eval`
- L196: `Topology VTI dir  : ttk_runs_fixed/topology_finetuning/candidateF_grad_PERS_only_expanded2688_topology_vti`
- L197: `Topology out dir  : ttk_runs_fixed/topology_finetuning/candidateF_grad_PERS_only_expanded2688_topology`
- L198: `Log file          : /home/adadhwal/PhIRE/logs/wind_finetune_candidateF_grad_PERS_only_expanded2688.log`
- L201: `lambda_uv       = 1.0  (always on; normalized [u, v])`
- L202: `lambda_speed    = 0.0`
- L203: `lambda_grad     = 0.05`
- L205: `lambda_levelset = 0.0`
- L206: `lambda_crit     = 0.0  (DISABLED)`
**training hyperparameters:** 11 matching lines (first 11 shown)
- L18: `importlib.import_module("keras.src.optimizers")`
- L96: `importlib.import_module("keras.src.optimizers")`
- L213: `learning_rate   = 1e-05`
- L214: `N_epochs        = 3`
- L217: `batch_size      = 4`
- L279: `Epoch: 1`
- L3979: `Epoch total loss (with TTK) = 0.05393`
- L3981: `Epoch: 2`
- L7681: `Epoch total loss (with TTK) = 0.04800`
- L7683: `Epoch: 3`
- L11383: `Epoch total loss (with TTK) = 0.04568`
**input paths / constraints / checkpoint restore:** 53 matching lines (first 12 shown)
- L15: `File "/home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py", line 510, in main`
- L93: `File "/home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py", line 510, in main`
- L158: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:591: tf_record_iterator (from tensorflow.python.lib.io.tf_record) is deprecated and will be removed in a future version.`
- L172: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:1138: DatasetV1.output_types (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.`
- L175: `WARNING:tensorflow:From /home/adadhwal/PhIRE/scripts/run_candidateF_expanded2688_finetune.py:1138: DatasetV1.output_shapes (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.`
- L188: `Candidate F expanded-2688 fine-tuning: variant='grad_pers_only' (method='candidateF_grad_PERS_only_expanded2688')`
- L190: `Training data     : example_data_topology_expanded_2688/wind_MR-HR.tfrecord`
- L191: `Evaluation data   : example_data_fixed/wind_MR-HR.tfrecord`
- L192: `Source checkpoint : models/wind_mr-hr/trained_cnn/cnn`
- L193: `Output model dir  : models_fixed/topology_finetuning/wind_finetune_candidateF_grad_PERS_only_expanded2688`
- L194: `Inference output  : data_out/wind_finetune_candidateF_grad_PERS_only_expanded2688`
- L195: `Cheap-eval dir    : ttk_runs_fixed/topology_finetuning/candidateF_grad_PERS_only_expanded2688_eval`
**completion / validation:** 4 matching lines (first 4 shown)
- L231: `Confirmed: 2688-sample training set matches candidateC_expanded2688 / candidateB_expanded2688 / the completed E2-low-2688 scripts.`
- L252: `--- Constraints NPZ metadata validation: PASSED ---`
- L11418: `[validate] Post-inference validation complete: shapes, exact idx order, finiteness, and (where the baseline was available) GT/IN alignment all passed.`
- L11421: `Run complete.`


## Lineage decision, evidence still required

1. Confirm historical F1 loss weights from its actual log and the original launch flags: UV 1.0, gradient 0.05, CV 0.004, PERS 0.002, all other optional losses disabled. Compare with the three new variant launch logs.
2. Confirm both historical and new runs used the same 2688-record training TFRecord, 64-pair repaired constraints and original pretrained CNN initialization. Current **file hashes** do not prove what bytes were used at July 2026 training time.
3. Compare trainer revision/code path and optimizer options. Current source SHA256 only fingerprints the current file; it is **not** the archived training-time source SHA256 unless separately recorded.
4. Record available random seeds and optimizer-state restoration evidence. If they are missing, keep the four-cell contrasts *descriptive* and avoid strict controlled causal attribution.

**Never infer training replication equivalence or identical stochastic trajectories from matching benchmark outputs.**
