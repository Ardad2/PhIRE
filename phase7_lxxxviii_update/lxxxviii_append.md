# Part LXXXVIII — Phase 7: live 64-field wind screening verified, preview parity and footprint review prepared (October 15, 2026)

## LXXXVIII.1 Confirmed Spark-side meteorological inventory

The previous Part LXXXVII screening was completed successfully and verified independently. Its `[PILOT] PASS` reported **64/64** WTK snapshots in four provisional, 500×500, 80-grid-cell-buffer-separated regions (Gulf inland, Upper Midwest, Pacific inland, Southwest borderland), all finite, and saved eight full-500×500 `[u,v]` + factor-5 block-average MR previews. The offline verifier passed (`0`) with **all 11** output files matching `SHA256SUMS`. No inference, model ranking, PD/MT, training, or locked test split has occurred.

| Candidate | Native mean speed (m/s) | Mean per-field p99 speed (m/s) | Mean per-field p95 absolute neighbor speed difference (m/s) |
|---|---:|---:|---:|
| gulf_inland | 6.494 | 12.345 | 0.896 |
| upper_midwest | 6.830 | 13.017 | 0.783 |
| pacific_inland | 6.043 | 15.612 | 2.011 |
| southwest_borderland | 5.277 | 13.437 | 1.566 |

These are selected 16-field **meteorological screening** statistics, not population estimates or confirmation of mountainous/coastal conditions. Original fine-tuning region remains frozen, and the proposed 2010/2012 windows do not overlap documented 2007 fine-tuning times. Original PhIRE pretraining patch membership remains unknown.

## LXXXVIII.2 Next proposed read-only compatibility gate

`Phase7_LXXXVIII_Preview_Parity_20261015.zip` prepares an isolated gate to verify eight preview fields and their WTK/time IDs against the hashed LXXXVII screening data, print exact native-grid geographic center/corners, and compare archived 100×100 MR to **PhIRE's own unchanged TensorFlow `utils.downscale_image()`** (K=5) on 500×500 `[u,v]`, tolerance 1e-8. It includes `--mode preflight` without TensorFlow, `--mode parity` with TensorFlow, and `--mode verify` to recheck the separately written report. It does not read the remote WTK, run any model, train, generate TFRecords, or finalize evaluation splits; source SHA and output digests are preserved. A successful gate will justify an isolated development-only six-model inference pilot, after the geographic split is frozen separately. No actual Spark parity result is claimed here.

Before exposing model-level metrics, manually review geographic footprint corners and ancillary environmental information and declare which regions are development vs locked final test. A **provisional discussion-only** split is Gulf/Southwest development vs Upper Midwest/Pacific final test. Locked regions should not be used to select weights, checkpoints, or model variants. Do not alter archived Phase 5E / Phase 6 records.
