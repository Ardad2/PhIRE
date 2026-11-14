# Phase 7 — LXXXIX locked geographic evaluation split

**Frozen before any Phase 7 model predictions or rankings were inspected.**

| Role | Region | WTK grid rectangle |
|---|---|---|
| Development | `gulf_inland` | `[205, 705, 1471, 1971]` |
| Development | `southwest_borderland` | `[173, 673, 572, 1072]` |
| Locked external test | `upper_midwest` | `[849, 1349, 1315, 1815]` |
| Locked external test | `pacific_inland` | `[820, 1320, 315, 815]` |

- Primary evaluation: 2010 February/May/August/November, 168 hourly fields each per region.
- Replication: same four seasonal weeks in 2012; do not reselect after looking at 2010 performance.
- Six frozen model checkpoints; no checkpoint/loss choice may depend on locked external-test scores.
- Only four specified development-region previews may enter the initial model-inference pilot.
- Historical 2007 crop-based scores remain development context, not a pristine test.
- Original PhIRE pretraining membership is unknown; do **not** describe regions as unseen during all training.
- Geographic regime labels are descriptive hypotheses, not independently confirmed land/terrain labels.
- Before final external-test evaluation, freeze inference, metrics, radii, crop positions, sample inclusion and decision criteria.

## Expected field counts

- `primary_fields_per_region`: 672
- `replication_fields_per_region`: 672
- `primary_development_fields`: 1344
- `primary_locked_test_fields`: 1344
- `replication_development_fields`: 1344
- `replication_locked_test_fields`: 1344
