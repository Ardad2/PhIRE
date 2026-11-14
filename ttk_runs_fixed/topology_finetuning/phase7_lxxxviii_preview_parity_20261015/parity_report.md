# Phase 7 — LXXXVIII preview parity and footprints

Original PhIRE `utils.downscale_image` (TF average pooling, K=5) compared against
eight frozen LXXXVII `hr_uv`/`mr_uv` previews without any inference or model scores.

| Region | Actual WTK center | WTK rectangle | Four geographic corners |
|---|---|---|---|
| gulf_inland | `{'latitude': 32.00639, 'longitude': -91.002563}` | `[205, 705, 1471, 1971]` | `[{'col': 1471, 'corner': 'northwest_grid', 'latitude': 27.643902, 'longitude': -96.331421, 'row': 205}, {'col': 1970, 'corner': 'northeast_grid', 'latitude': 27.127483, 'longitude': -86.355316, 'row': 205}, {'col': 1471, 'corner': 'southwest_grid', 'latitude': 36.691231, 'longitude': -96.378845, 'row': 704}, {'col': 1970, 'corner': 'southeast_grid', 'latitude': 36.080818, 'longitude': -84.991516, 'row': 704}]` |
| upper_midwest | `{'latitude': 43.996811, 'longitude': -93.993713}` | `[849, 1349, 1315, 1815]` | `[{'col': 1315, 'corner': 'northwest_grid', 'latitude': 39.287479, 'longitude': -100.128815, 'row': 849}, {'col': 1814, 'corner': 'northeast_grid', 'latitude': 39.075306, 'longitude': -88.202972, 'row': 849}, {'col': 1315, 'corner': 'southwest_grid', 'latitude': 48.554726, 'longitude': -100.851776, 'row': 1348}, {'col': 1814, 'corner': 'southeast_grid', 'latitude': 48.304329, 'longitude': -86.845612, 'row': 1348}]` |
| pacific_inland | `{'latitude': 40.993176, 'longitude': -119.009521}` | `[820, 1320, 315, 815]` | `[{'col': 315, 'corner': 'northwest_grid', 'latitude': 35.184029, 'longitude': -122.799591, 'row': 820}, {'col': 814, 'corner': 'northeast_grid', 'latitude': 37.602203, 'longitude': -111.788025, 'row': 820}, {'col': 315, 'corner': 'southwest_grid', 'latitude': 43.826946, 'longitude': -127.032013, 'row': 1319}, {'col': 814, 'corner': 'southeast_grid', 'latitude': 46.661736, 'longitude': -114.44162, 'row': 1319}]` |
| southwest_borderland | `{'latitude': 30.505024, 'longitude': -109.996368}` | `[173, 673, 572, 1072]` | `[{'col': 572, 'corner': 'northwest_grid', 'latitude': 25.257706, 'longitude': -113.93634, 'row': 173}, {'col': 1071, 'corner': 'northeast_grid', 'latitude': 26.694355, 'longitude': -104.26976, 'row': 173}, {'col': 572, 'corner': 'southwest_grid', 'latitude': 33.956936, 'longitude': -116.377441, 'row': 672}, {'col': 1071, 'corner': 'southeast_grid', 'latitude': 35.652069, 'longitude': -105.432159, 'row': 672}]` |

| Preview | Max abs MR difference | Mean abs difference |
|---|---:|---:|
| `preview_gulf_inland_20100201.npz` | 5.32907052e-15 | 6.08193664e-16 |
| `preview_gulf_inland_20100801.npz` | 5.32907052e-15 | 3.58896348e-16 |
| `preview_upper_midwest_20100201.npz` | 7.10542736e-15 | 4.99968751e-16 |
| `preview_upper_midwest_20100801.npz` | 5.32907052e-15 | 3.52769065e-16 |
| `preview_pacific_inland_20100201.npz` | 7.10542736e-15 | 3.63421562e-16 |
| `preview_pacific_inland_20100801.npz` | 1.0658141e-14 | 7.07676533e-16 |
| `preview_southwest_borderland_20100201.npz` | 7.10542736e-15 | 4.2385543e-16 |
| `preview_southwest_borderland_20100801.npz` | 7.10542736e-15 | 5.39237632e-16 |

## Decision boundaries

- All geographical/environmental labels are provisional until footprint and auxiliary-land/terrain review.
- No development/test roles or timestamp windows are locked by this report.
- Do **not** view any model scores on prospective locked test regions before declaring roles.
- This establishes input downsampling equivalence, not model output equivalence or preprocessing for a complete two-stage PhIRE cascade.
- Native WTK wind orientation and projected physical distances require separate certification for 500x500 topology.
