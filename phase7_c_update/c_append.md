# Part C — Phase 7: verified sixteen-field development topology/locality; first complete development week source pilot prepared (October 18, 2026)

## Part XCIX real Spark closeout

- XCIX was executed and verified on Spark with successful selftest, preflight, computation and independent verification. All **112** canonical PD diagrams have verified birth/death critical feature values/coordinates for H0 and H1 after the original zero-persistence filter. The unchanged SHA-pinned Phase5E sparse solver produced **96 model-to-GT comparisons × 8 radii = 768 aggregate locality rows** and **1536 dimension-level rows**; all **96** infinity-radius values agree with XCVIII corrected GUDHI `W22`; all **192** original XCIV aggregate and **384** dimension-level regressions reproduced. Eight region-week × six model summaries (48 rows) verified. Five output SHA256 checks passed. No locked external-region scores accessed.
- Descriptive, equal-snapshot means from 16 fields (two per 2010 region-week, two development regions):

  | model | r4 W22 ↓ | r16 W22 ↓ | r32 W22 ↓ | r∞ W22 ↓ | GT persistence-mass coverage r16 ↑ |
  |---|---:|---:|---:|---:|---:|
  | cnn | 19.523399 | 15.817982 | 13.593439 | 10.462440 | 0.489421 |
  | gan | 26.324920 | 20.457178 | 16.810830 | 12.211472 | 0.798443 |
  | grad | **18.869274** | **14.002953** | **11.729558** | **8.375874** | 0.792718 |
  | cv | 19.407128 | 14.623993 | 11.965772 | 8.502258 | 0.741817 |
  | pers | 19.712260 | 14.615091 | 12.088795 | 8.483273 | 0.747704 |
  | f1 | 19.570265 | 14.393966 | 11.853859 | 8.607247 | 0.724932 |

- Progress log displays `[LOCALITY] 25/24` to `96/24` because of an incorrect diagnostic denominator. The verified CSV count, XCVIII infinity anchors, XCIV historical regressions and numerical metrics are unaffected; keep validated XCIX data unchanged. No inference should treat the sixteen hours as full independent meteorological temporal replication.

## Part C first complete development source-week workpack — prepared; LIVE Spark not yet run

- Goal: test the full 168-hour WTK source-acquisition, streaming, storage, failure recovery and provenance path **before** launching 168-hour inference/TTK/GUDHI. No model predictions or new topological metrics in Part C.
- Frozen target from LXXXIX/XCI: development `gulf_inland` WTK rectangle `[205,705,1471,1971]`, February 1, 2010 00:00 UTC through February 7, 2010 23:00 UTC inclusive, WTK indices **27048–27215**, 168 consecutive hourly records; no locked-region arrays read. Full 500×500×2 float64 HR `[u,v]`, 100×100×2 factor-five MR `[u,v]`, original LXXXVII direction/vector convention, numeric pooling parity checks.
- Reuse XCV's already verified `gulf_inland` 2010-Feb offset hours **0 and 84** bit-for-bit. Fetch only the remaining **166** native WTK slices, with read-only HSDS `datetime` verification before data acquisition, numeric finite/range checks, and immutable per-hour SHA256 ledger. Do not silently skip or replace an hour. Supports `--max-new-hours 6` first batch and idempotent resumption from validated staging after expected interruption.
- Offline preflight recomputes SHA256 lineage of frozen geographic split, frozen XCI protocol, XCV sixteen source fields and completed XCIX locality artifact; prints local RAM/disk availability and exact uncompressed source/model-output array byte budgets. HSDS online probe is separately invoked; extraction does not run before these gates and refuses completed-output overwrite.
- Exact source-only raw HR+MR uncompressed payload per week = **698,880,000 bytes (~666.5 MiB)**; six eventual uncompressed full500 model output arrays = **4,032,000,000 bytes (~3.75 GiB)**. These are bytes for raw numeric arrays, **not** predictions of NPZ/TFRecord/TTK disk size, runtime, network throughput, or peak neural-network memory; measure actual usage and wall time on Spark.
- Complete source-only output (after Spark run): `ttk_runs_fixed/topology_finetuning/phase7_c_dev_gulf_inland_feb2010_week168_20261018/`, 168 per-hour `.npz` plus four reports/config/ledger and SHA256 index, **173 checksum entries**. Stage is `ttk_runs_fixed/topology_finetuning/.phase7_c_dev_gulf_inland_feb2010_week168_20261018.staging/` and is resumable; no auto-deletion on failure. Verification rechecks each numerical field and XCV byte provenance, exact hours/region, reports, and checksums without WTK/TF/TTK.
- Local packaging and synthetic 168-hour HSDS-mock interruption/resume, corruption/overwrite and verification tests are **engineering-only**. Live Spark extraction and measurements remain pending. No locked external tests, no training, no changes to frozen XCI plan, no model selection, and no scientific generalization claim. Original PhIRE pretrained temporal/geographic sample membership is still unknown.
