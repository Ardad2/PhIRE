# Expanded Dataset 1344 Notes

**Generated:** 2026-05-28

## Summary

- Total samples: 1344 (8 windows × 168 hours)
- Output directory: `example_data_topology_expanded_1344`
- HR shape: [1344, 500, 500, 2]

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
| spring_1 | 2160 | 168 | 2327 |
| spring_2 | 2328 | 168 | 2495 |
| summer_1 | 4344 | 168 | 4511 |
| summer_2 | 4512 | 168 | 4679 |
| fall_1 | 6552 | 168 | 6719 |
| fall_2 | 6720 | 168 | 6887 |

## Resolution hierarchy

- HR: 500×500×2 — native WTK crop
- MR: 100×100×2 — 5× block-average from HR via `utils.downscale_image`
- LR: 10×10×2 — 10× block-average from MR via `utils.downscale_image`

## Statistics over expanded HR stack

- u: mean=3.1746  std=5.6360  min=-29.249  max=37.127
- v: mean=-0.0742  std=5.5822  min=-25.524  max=36.243
- speed: mean=7.5653  std=3.9717  p50=7.160  p90=12.912  p95=14.924  p99=18.622

## Normalization compatibility with pretrained PhIRE CNN

Pretrained: mu=[0.7684, -0.4575]  sig=[5.02455, 5.9017]

- u mean z-score vs pretrained: 0.479
- v mean z-score vs pretrained: 0.065
- **compatible=True** (criterion: |z| < 3 for both channels)

The expanded dataset mean falls within 3σ of the pretrained normalization center.  The pretrained CNN checkpoint can be applied to this data without recomputing mu/sig.

## Benchmark non-overlap

Minimum WTK index in the expanded set: 336  (benchmark ends at 167).
None of the 1344 samples overlap the original benchmark (WTK 0..167 = 2007-01-01 00:00 to 2007-01-07 23:00 UTC).

## Output files

| File | Description |
|------|-------------|
| `example_data_topology_expanded_1344/wind_LR-MR.tfrecord` | PhIRE TFRecord LR=10×10→MR=100×100 |
| `example_data_topology_expanded_1344/wind_MR-HR.tfrecord` | PhIRE TFRecord MR=100×100→HR=500×500 |
| `example_data_topology_expanded_1344/hr_stack.npy` | (1344, 500, 500, 2) float64 |
| `example_data_topology_expanded_1344/mr_stack.npy` | (1344, 100, 100, 2) float64 |
| `example_data_topology_expanded_1344/lr_stack.npy` | (1344, 10, 10, 2) float64 |
| `example_data_topology_expanded_1344/manifest.csv` | Per-sample metadata (1344 rows) |
| `example_data_topology_expanded_1344/stats.json` | Channel stats and normalization comparison |
