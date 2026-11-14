# Phase 7 Part CIV — 168-hour development spatial-locality analysis

Gulf Inland, 2010-11 1–7, 2010 UTC; 168 contiguous hourly fields × six frozen models.
**One development region-week. Consecutive hours/days are temporally correlated, not independent generalization replications.**

Unchanged original Phase5E sparse matching on TTK canonical top-left 160×160 speed persistence diagrams.
Radii **4, 8, 16, 32, 64, 96, 128, ∞ are WTK grid cells, not kilometres**.

Lower constrained W₂,₂ means better persistence matching; GT persistence-mass coverage is separate (higher is better).

| Model | r4 W22 ↓ | r16 W22 ↓ | r32 W22 ↓ | r∞ W22 ↓ | GT coverage r16 ↑ |
|---|---:|---:|---:|---:|---:|
| cnn | 14.200781 | 12.510399 | 11.369619 | 9.998829 | 0.386127 |
| gan | 18.825641 | 13.998220 | 11.214817 | 8.265875 | 0.798987 |
| grad | 14.664704 | 10.898016 | 9.281363 | 7.635318 | 0.839654 |
| cv | 14.891363 | 11.272305 | 9.623212 | 7.919711 | 0.770807 |
| pers | 14.775600 | 11.198477 | 9.495676 | 7.800484 | 0.763423 |
| f1 | 14.879445 | 11.372314 | 9.710925 | 7.982836 | 0.749880 |

## Daily descriptive summaries (24 consecutive hours per UTC date)

| UTC day | Arm | N | r4 W22 ↓ | r16 W22 ↓ | r∞ W22 ↓ | GT coverage r16 ↑ |
|---|---|---:|---:|---:|---:|---:|
| 2010-11-01 | cnn | 24 | 17.080086 | 14.884793 | 11.929254 | 0.419041 |
| 2010-11-01 | gan | 24 | 22.315529 | 15.423920 | 8.331621 | 0.848473 |
| 2010-11-01 | grad | 24 | 17.400594 | 12.234456 | 8.229865 | 0.874251 |
| 2010-11-01 | cv | 24 | 17.580090 | 12.825668 | 8.582079 | 0.800522 |
| 2010-11-01 | pers | 24 | 17.400881 | 12.905957 | 8.756954 | 0.799376 |
| 2010-11-01 | f1 | 24 | 17.514743 | 12.976069 | 8.868847 | 0.765862 |
| 2010-11-02 | cnn | 24 | 20.921867 | 17.942355 | 12.707128 | 0.439592 |
| 2010-11-02 | gan | 24 | 29.006425 | 21.808311 | 12.079307 | 0.837502 |
| 2010-11-02 | grad | 24 | 20.681846 | 15.222642 | 9.410164 | 0.830151 |
| 2010-11-02 | cv | 24 | 21.349057 | 15.252261 | 9.337242 | 0.767743 |
| 2010-11-02 | pers | 24 | 21.482810 | 15.663884 | 9.527265 | 0.773478 |
| 2010-11-02 | f1 | 24 | 21.470562 | 15.628919 | 9.454667 | 0.750691 |
| 2010-11-03 | cnn | 24 | 16.200576 | 13.569877 | 10.645888 | 0.490463 |
| 2010-11-03 | gan | 24 | 21.868532 | 15.230716 | 8.733598 | 0.855980 |
| 2010-11-03 | grad | 24 | 16.703970 | 12.203145 | 8.914793 | 0.899932 |
| 2010-11-03 | cv | 24 | 16.432825 | 12.241653 | 8.743583 | 0.796514 |
| 2010-11-03 | pers | 24 | 16.698034 | 12.014996 | 8.486724 | 0.842074 |
| 2010-11-03 | f1 | 24 | 16.461259 | 12.243851 | 8.821076 | 0.806618 |
| 2010-11-04 | cnn | 24 | 14.504182 | 12.458619 | 9.629994 | 0.436282 |
| 2010-11-04 | gan | 24 | 19.617045 | 14.439389 | 9.142795 | 0.842338 |
| 2010-11-04 | grad | 24 | 15.380566 | 11.180287 | 7.875865 | 0.892148 |
| 2010-11-04 | cv | 24 | 16.268993 | 11.967605 | 8.247544 | 0.866781 |
| 2010-11-04 | pers | 24 | 15.664820 | 11.408476 | 7.658741 | 0.833560 |
| 2010-11-04 | f1 | 24 | 16.192635 | 11.870445 | 8.074094 | 0.842480 |
| 2010-11-05 | cnn | 24 | 9.583636 | 8.876535 | 7.580986 | 0.359661 |
| 2010-11-05 | gan | 24 | 14.440465 | 11.399074 | 7.087445 | 0.780399 |
| 2010-11-05 | grad | 24 | 10.620184 | 7.876834 | 5.414314 | 0.867733 |
| 2010-11-05 | cv | 24 | 10.888499 | 8.289168 | 5.894254 | 0.857444 |
| 2010-11-05 | pers | 24 | 10.495854 | 8.094953 | 5.548878 | 0.812629 |
| 2010-11-05 | f1 | 24 | 10.791326 | 8.372137 | 5.837794 | 0.829517 |
| 2010-11-06 | cnn | 24 | 9.753167 | 9.141151 | 8.086655 | 0.276550 |
| 2010-11-06 | gan | 24 | 11.435108 | 9.238696 | 5.895867 | 0.711899 |
| 2010-11-06 | grad | 24 | 10.339906 | 8.219122 | 6.210626 | 0.774902 |
| 2010-11-06 | cv | 24 | 10.173087 | 8.488027 | 6.604690 | 0.682524 |
| 2010-11-06 | pers | 24 | 10.136980 | 8.411511 | 6.560459 | 0.680917 |
| 2010-11-06 | f1 | 24 | 10.192159 | 8.550026 | 6.691433 | 0.661226 |
| 2010-11-07 | cnn | 24 | 11.361955 | 10.699461 | 9.411898 | 0.281299 |
| 2010-11-07 | gan | 24 | 13.096384 | 10.447432 | 6.590489 | 0.716320 |
| 2010-11-07 | grad | 24 | 11.525863 | 9.349622 | 7.391599 | 0.738460 |
| 2010-11-07 | cv | 24 | 11.546992 | 9.841750 | 8.028584 | 0.624123 |
| 2010-11-07 | pers | 24 | 11.549820 | 9.889562 | 8.064366 | 0.601930 |
| 2010-11-07 | f1 | 24 | 11.533427 | 9.964751 | 8.131940 | 0.592769 |

## Scientific and reproducibility boundaries

- All 1,008 infinite-radius W22 model-to-GT values must match the sealed Part CIII corrected GUDHI output.
- All 96 aggregate + 192 dimension rows for historical week hours 0 and 84 must reproduce Part XCIX.
- Canonical zero/near-zero persistence pairs use the frozen original Phase5E tolerance 1e-10 with excluded count validation.
- All 1,176 diagram coordinate sets are validated against their corrected numerical birth/death features.
- GT persistence-mass coverage measures matched mass, not correct spatial positions or physical wind structure.
- This is not topology on full 500×500 fields and does not include merge trees or training.
- No external locked-region predictions or new hyperparameter/model selection. Original pretrained WTK exposure unknown.
- Verifier repeats the same original Phase5E algorithm; it is not an independently implemented solver.

