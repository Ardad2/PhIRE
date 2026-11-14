# Phase 7 / LXXXVII — provisional wind-field screening

64 full 500×500 native WTK snapshots (4 provisional regions × 8 seasonal windows × 2 UTC hours).
No training, inference, PD/MT calculation, model ranking, or locked test-set designation.

| Region | Valid fraction (minimum) | Mean wind speed (m/s) | Mean per-field p99 speed (m/s) | Mean per-field p95 abs pixel gradient (m/s) |
|---|---:|---:|---:|---:|
| gulf_inland | 1.00000 | 6.494 | 12.345 | 0.896 |
| upper_midwest | 1.00000 | 6.830 | 13.017 | 0.783 |
| pacific_inland | 1.00000 | 6.043 | 15.612 | 2.011 |
| southwest_borderland | 1.00000 | 5.277 | 13.437 | 1.566 |

## Caveats

- These 16 snapshots per region are a geographic screening sample, **not** a population estimate or a generalization benchmark.
- WTK coordinates, labels and differences in wind fields do not certify terrain or coastal regime; inspect footprint and ancillary terrain/land metadata before assigning environment types.
- Snapshot winds are from WTK speed/direction at 100 m, converted to native [u,v]. Only complete finite snapshots are saved as previews.
- Derived preview MR is explicit factor-5 block-average. Exact parity with original PhIRE TensorFlow downscale must be checked independently before inference.
- All 2010/2012 timestamps are disjoint from the documented original 2007 fine-tuning windows. Original PhIRE pretraining exposure remains unknown.
- All regions remain provisional. Test-set labels must be frozen before looking at model scores.
