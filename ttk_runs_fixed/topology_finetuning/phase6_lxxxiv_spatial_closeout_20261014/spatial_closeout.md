# Phase 6 LXXXIV — full spatial paired closeout

168 consecutive hourly samples; original Phase5E sparse matching; lower constrained W22 is better.
GT/SR coverage is persistence-mass coverage, **not** localization error or geographic distance.

## Radius-dependent cost and GT/SR coverage

| Radius | Best mean cost | Mean cost | Gradient cost | PERS cost | F1 cost | Gradient GT cov | PERS GT cov | F1 GT cov |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 4.0 | grad | 45.844483 | 45.844483 | 48.570148 | 48.504946 | 0.363808 | 0.253440 | 0.238026 |
| 8.0 | grad | 39.440410 | 39.440410 | 41.670743 | 41.609170 | 0.659982 | 0.515839 | 0.487677 |
| 16.0 | grad | 31.008340 | 31.008340 | 31.841850 | 31.947250 | 0.867886 | 0.747282 | 0.710923 |
| 32.0 | pers | 23.707846 | 23.893099 | 23.707846 | 23.921293 | 0.958785 | 0.881991 | 0.846919 |
| 64.0 | pers | 19.295664 | 19.938698 | 19.295664 | 19.608685 | 0.985527 | 0.943713 | 0.916390 |
| 96.0 | pers | 18.130971 | 18.895101 | 18.130971 | 18.453561 | 0.990089 | 0.958661 | 0.935046 |
| 128.0 | pers | 17.680619 | 18.490081 | 17.680619 | 18.024203 | 0.991550 | 0.963313 | 0.941073 |
| inf | pers | 17.484939 | 18.306367 | 17.484939 | 17.831897 | 0.991978 | 0.964834 | 0.943107 |

## Pairwise effects: Gradient versus PERS and F1 versus PERS

| Radius | Contrast | Mean cost A−B | Median cost A−B | A lower / B lower / ties | Mean GT coverage A−B |
|---|---|---:|---:|---:|---:|
| 4.0 | grad−pers | -2.725665 | -2.552701 | 166 / 2 / 0 | +0.110369 |
| 4.0 | f1−pers | -0.065202 | -0.039432 | 86 / 82 / 0 | -0.015414 |
| 8.0 | grad−pers | -2.230333 | -1.956511 | 157 / 11 / 0 | +0.144143 |
| 8.0 | f1−pers | -0.061573 | -0.018990 | 85 / 83 / 0 | -0.028162 |
| 16.0 | grad−pers | -0.833510 | -0.725593 | 126 / 42 / 0 | +0.120604 |
| 16.0 | f1−pers | +0.105400 | +0.081629 | 78 / 90 / 0 | -0.036359 |
| 32.0 | grad−pers | +0.185252 | +0.128554 | 77 / 91 / 0 | +0.076794 |
| 32.0 | f1−pers | +0.213447 | +0.139000 | 74 / 94 / 0 | -0.035072 |
| 64.0 | grad−pers | +0.643034 | +0.361074 | 63 / 105 / 0 | +0.041813 |
| 64.0 | f1−pers | +0.313022 | +0.309781 | 38 / 130 / 0 | -0.027323 |
| 96.0 | grad−pers | +0.764130 | +0.485796 | 64 / 104 / 0 | +0.031428 |
| 96.0 | f1−pers | +0.322591 | +0.359460 | 38 / 130 / 0 | -0.023615 |
| 128.0 | grad−pers | +0.809462 | +0.545619 | 61 / 107 / 0 | +0.028237 |
| 128.0 | f1−pers | +0.343584 | +0.336651 | 33 / 135 / 0 | -0.022240 |
| inf | grad−pers | +0.821428 | +0.567661 | 63 / 105 / 0 | +0.027144 |
| inf | f1−pers | +0.346958 | +0.330417 | 34 / 134 / 0 | -0.021726 |

## Within-benchmark factorial contrasts by radius

The zero cell is gradient-only, not pretrained CNN. Negative cost effects improve lower-is-better W22. Positive coverage effects raise matched GT persistence mass.

| Radius | Outcome | CV effect | PERS effect | Interaction |
|---|---|---:|---:|---:|
| 4.0 | cost W22 | +0.920547 | +1.739916 | -1.971498 |
| 4.0 | GT coverage | -0.060769 | -0.065013 | +0.090710 |
| 8.0 | cost W22 | +0.668879 | +1.499881 | -1.460903 |
| 8.0 | GT coverage | -0.085797 | -0.086508 | +0.115269 |
| 16.0 | cost W22 | +0.334890 | +0.604020 | -0.458980 |
| 16.0 | GT coverage | -0.081803 | -0.075159 | +0.090888 |
| 32.0 | cost W22 | +0.066319 | -0.038124 | +0.294257 |
| 32.0 | GT coverage | -0.060373 | -0.051492 | +0.050604 |
| 64.0 | cost W22 | -0.015164 | -0.314849 | +0.656371 |
| 64.0 | GT coverage | -0.038078 | -0.031058 | +0.021510 |
| 96.0 | cost W22 | -0.040974 | -0.400565 | +0.727130 |
| 96.0 | GT coverage | -0.030623 | -0.024421 | +0.014015 |
| 128.0 | cost W22 | -0.042981 | -0.422897 | +0.773131 |
| 128.0 | GT coverage | -0.028095 | -0.022382 | +0.011711 |
| inf | cost W22 | -0.041882 | -0.432587 | +0.777680 |
| inf | GT coverage | -0.027250 | -0.021621 | +0.011047 |

## 24-hour descriptive blocks: Gradient versus PERS constrained cost

| Radius | Seven chronological 24-hour mean differences (gradient−PERS) |
|---|---|
| 4.0 | -2.4882, -3.5483, -1.5438, -2.9606, -3.2463, -2.8647, -2.4277 |
| 16.0 | -0.6759, -0.7132, -0.8157, -1.1632, -1.1139, -0.7018, -0.6508 |
| 32.0 | +0.4133, +1.0408, -0.3091, -0.3118, -0.6843, +0.6410, +0.5069 |
| inf | +1.4115, +2.2895, +0.1631, +0.0869, -0.4682, +1.1670, +1.1002 |

**Interpretation limits:** Single training instance for each configuration; temporal dependence between hourly fields; F1 optimizer, checkpoint and random-seed lineage unclosed. Original PD/MT and Phase5E sources unchanged. Merge-tree distances for the three new arms remain pending.
