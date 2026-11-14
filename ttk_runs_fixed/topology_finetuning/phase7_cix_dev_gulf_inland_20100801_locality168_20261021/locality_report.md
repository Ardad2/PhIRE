# Phase 7 Part CIV — 168-hour development spatial-locality analysis

Gulf Inland, 2010-08 1–7, 2010 UTC; 168 contiguous hourly fields × six frozen models.
**One development region-week. Consecutive hours/days are temporally correlated, not independent generalization replications.**

Unchanged original Phase5E sparse matching on TTK canonical top-left 160×160 speed persistence diagrams.
Radii **4, 8, 16, 32, 64, 96, 128, ∞ are WTK grid cells, not kilometres**.

Lower constrained W₂,₂ means better persistence matching; GT persistence-mass coverage is separate (higher is better).

| Model | r4 W22 ↓ | r16 W22 ↓ | r32 W22 ↓ | r∞ W22 ↓ | GT coverage r16 ↑ |
|---|---:|---:|---:|---:|---:|
| cnn | 11.548860 | 10.125259 | 9.259120 | 8.091253 | 0.372259 |
| gan | 15.415590 | 11.231750 | 8.989279 | 6.316753 | 0.791335 |
| grad | 11.721956 | 8.681325 | 7.428425 | 5.842914 | 0.817263 |
| cv | 11.850340 | 8.935643 | 7.658981 | 6.071065 | 0.769276 |
| pers | 11.826266 | 8.965929 | 7.664422 | 6.044826 | 0.779672 |
| f1 | 11.886896 | 9.072653 | 7.779399 | 6.188861 | 0.743058 |

## Daily descriptive summaries (24 consecutive hours per UTC date)

| UTC day | Arm | N | r4 W22 ↓ | r16 W22 ↓ | r∞ W22 ↓ | GT coverage r16 ↑ |
|---|---|---:|---:|---:|---:|---:|
| 2010-08-01 | cnn | 24 | 11.119112 | 10.184296 | 8.482186 | 0.351167 |
| 2010-08-01 | gan | 24 | 14.291257 | 10.640261 | 6.206442 | 0.780835 |
| 2010-08-01 | grad | 24 | 11.426833 | 8.621228 | 5.904251 | 0.832074 |
| 2010-08-01 | cv | 24 | 11.456550 | 8.987395 | 6.270727 | 0.768633 |
| 2010-08-01 | pers | 24 | 11.517238 | 8.955799 | 6.223617 | 0.792569 |
| 2010-08-01 | f1 | 24 | 11.485830 | 8.956322 | 6.352192 | 0.749511 |
| 2010-08-02 | cnn | 24 | 8.588553 | 7.477383 | 5.889154 | 0.341376 |
| 2010-08-02 | gan | 24 | 12.498086 | 9.752936 | 6.131591 | 0.756419 |
| 2010-08-02 | grad | 24 | 8.705359 | 6.651736 | 4.336325 | 0.761913 |
| 2010-08-02 | cv | 24 | 8.858191 | 6.773720 | 4.533611 | 0.739440 |
| 2010-08-02 | pers | 24 | 8.947445 | 6.971823 | 4.548851 | 0.752746 |
| 2010-08-02 | f1 | 24 | 8.887272 | 7.007652 | 4.589962 | 0.723787 |
| 2010-08-03 | cnn | 24 | 10.707390 | 9.238213 | 7.438259 | 0.376291 |
| 2010-08-03 | gan | 24 | 14.397406 | 10.513224 | 5.905589 | 0.773484 |
| 2010-08-03 | grad | 24 | 10.716476 | 7.753975 | 5.493327 | 0.824019 |
| 2010-08-03 | cv | 24 | 10.778402 | 7.998092 | 5.658874 | 0.766704 |
| 2010-08-03 | pers | 24 | 10.849086 | 8.240215 | 5.598036 | 0.777385 |
| 2010-08-03 | f1 | 24 | 10.932307 | 8.218419 | 5.699918 | 0.756586 |
| 2010-08-04 | cnn | 24 | 11.422577 | 10.124702 | 8.481140 | 0.368924 |
| 2010-08-04 | gan | 24 | 15.229545 | 10.994184 | 5.995681 | 0.801047 |
| 2010-08-04 | grad | 24 | 11.540400 | 8.484855 | 6.110552 | 0.816373 |
| 2010-08-04 | cv | 24 | 11.724789 | 8.878449 | 6.327799 | 0.769971 |
| 2010-08-04 | pers | 24 | 11.654824 | 8.921410 | 6.257618 | 0.775656 |
| 2010-08-04 | f1 | 24 | 11.807939 | 8.865878 | 6.445618 | 0.738135 |
| 2010-08-05 | cnn | 24 | 10.445537 | 9.069502 | 7.372038 | 0.366382 |
| 2010-08-05 | gan | 24 | 13.837511 | 10.025906 | 5.943084 | 0.795074 |
| 2010-08-05 | grad | 24 | 10.607086 | 7.991647 | 5.403510 | 0.819060 |
| 2010-08-05 | cv | 24 | 10.698460 | 8.004767 | 5.602097 | 0.787343 |
| 2010-08-05 | pers | 24 | 10.722708 | 8.061627 | 5.549249 | 0.794329 |
| 2010-08-05 | f1 | 24 | 10.681624 | 8.077691 | 5.723633 | 0.760468 |
| 2010-08-06 | cnn | 24 | 12.298270 | 10.518265 | 8.246999 | 0.404368 |
| 2010-08-06 | gan | 24 | 16.359733 | 11.631764 | 6.539064 | 0.805042 |
| 2010-08-06 | grad | 24 | 12.518654 | 9.203872 | 6.144126 | 0.818951 |
| 2010-08-06 | cv | 24 | 12.516227 | 9.244157 | 6.389897 | 0.774987 |
| 2010-08-06 | pers | 24 | 12.369984 | 9.119122 | 6.364297 | 0.781119 |
| 2010-08-06 | f1 | 24 | 12.464951 | 9.494313 | 6.561482 | 0.728100 |
| 2010-08-07 | cnn | 24 | 16.260581 | 14.264450 | 10.728994 | 0.397304 |
| 2010-08-07 | gan | 24 | 21.295590 | 15.063972 | 7.495823 | 0.827443 |
| 2010-08-07 | grad | 24 | 16.538885 | 12.061960 | 7.508306 | 0.848454 |
| 2010-08-07 | cv | 24 | 16.919761 | 12.662918 | 7.714452 | 0.777855 |
| 2010-08-07 | pers | 24 | 16.722579 | 12.491504 | 7.772115 | 0.783898 |
| 2010-08-07 | f1 | 24 | 16.948351 | 12.888299 | 7.949221 | 0.744822 |

## Scientific and reproducibility boundaries

- All 1,008 infinite-radius W22 model-to-GT values must match the sealed Part CIII corrected GUDHI output.
- All 96 aggregate + 192 dimension rows for historical week hours 0 and 84 must reproduce Part XCIX.
- Canonical zero/near-zero persistence pairs use the frozen original Phase5E tolerance 1e-10 with excluded count validation.
- All 1,176 diagram coordinate sets are validated against their corrected numerical birth/death features.
- GT persistence-mass coverage measures matched mass, not correct spatial positions or physical wind structure.
- This is not topology on full 500×500 fields and does not include merge trees or training.
- No external locked-region predictions or new hyperparameter/model selection. Original pretrained WTK exposure unknown.
- Verifier repeats the same original Phase5E algorithm; it is not an independently implemented solver.

