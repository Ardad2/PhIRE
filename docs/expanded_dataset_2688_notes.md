# Expanded Dataset 2688 Notes

**Generated:** 2026-05-29

## Summary

- Total samples: 2688 (16 windows × 168 hours)
- Output directory: `example_data_topology_expanded_2688`
- HR shape: [2688, 500, 500, 2]

## Spatial crop

- Target center: 39.5°N, -75.0°W
- Actual grid center: 39.4998°N, -75.0064°W
- WTK native rows: 720:1220, cols: 2102:2602
- HR patch: 500×500 (same as 168-sample benchmark)

## Seasonal windows

| Season | WTK start | Length | WTK end |
|--------|-----------|--------|---------|
| winter_1 | 336 | 168 | 503 |
| winter_2 | 504 | 168 | 671 |
| winter_3 | 672 | 168 | 839 |
| winter_4 | 840 | 168 | 1007 |
| spring_1 | 2160 | 168 | 2327 |
| spring_2 | 2328 | 168 | 2495 |
| spring_3 | 2496 | 168 | 2663 |
| spring_4 | 2664 | 168 | 2831 |
| summer_1 | 4344 | 168 | 4511 |
| summer_2 | 4512 | 168 | 4679 |
| summer_3 | 4680 | 168 | 4847 |
| summer_4 | 4848 | 168 | 5015 |
| fall_1 | 6552 | 168 | 6719 |
| fall_2 | 6720 | 168 | 6887 |
| fall_3 | 6888 | 168 | 7055 |
| fall_4 | 7056 | 168 | 7223 |

## Resolution hierarchy

- HR: 500×500×2 — native WTK crop
- MR: 100×100×2 — 5× block-average from HR via `utils.downscale_image`
- LR: 10×10×2 — 10× block-average from MR via `utils.downscale_image`

## Statistics over expanded HR stack

- u: mean=3.0778  std=5.7621  min=-37.500  max=37.127
- v: mean=0.1283  std=6.0117  min=-39.461  max=38.160
- speed: mean=7.8383  std=4.1706  p50=7.398  p90=13.425  p95=15.464  p99=19.303

## Normalization compatibility with pretrained PhIRE CNN

Pretrained: mu=[0.7684, -0.4575]  sig=[5.02455, 5.9017]

- u mean z-score vs pretrained: 0.460
- v mean z-score vs pretrained: 0.099
- **compatible=True** (criterion: |z| < 3 for both channels)

The expanded dataset mean falls within 3σ of the pretrained normalization center.  The pretrained CNN checkpoint can be applied to this data without recomputing mu/sig.

## Benchmark non-overlap

Minimum WTK index in the expanded set: 336  (benchmark ends at 167).
None of the 2688 samples overlap the original benchmark (WTK 0..167 = 2007-01-01 00:00 to 2007-01-07 23:00 UTC).

## Output files

| File | Description |
|------|-------------|
| `example_data_topology_expanded_2688/wind_LR-MR.tfrecord` | PhIRE TFRecord LR=10×10→MR=100×100 |
| `example_data_topology_expanded_2688/wind_MR-HR.tfrecord` | PhIRE TFRecord MR=100×100→HR=500×500 |
| `example_data_topology_expanded_2688/hr_stack.npy` | (2688, 500, 500, 2) float64 |
| `example_data_topology_expanded_2688/mr_stack.npy` | (2688, 100, 100, 2) float64 |
| `example_data_topology_expanded_2688/lr_stack.npy` | (2688, 10, 10, 2) float64 |
| `example_data_topology_expanded_2688/manifest.csv` | Per-sample metadata (2688 rows) |
| `example_data_topology_expanded_2688/stats.json` | Channel stats and normalization comparison |
