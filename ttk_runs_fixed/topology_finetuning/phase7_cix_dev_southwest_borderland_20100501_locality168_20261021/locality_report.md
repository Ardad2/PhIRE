# Phase 7 Part CIV — 168-hour development spatial-locality analysis

Southwest Borderland, 2010-05 1–7, 2010 UTC; 168 contiguous hourly fields × six frozen models.
**One development region-week. Consecutive hours/days are temporally correlated, not independent generalization replications.**

Unchanged original Phase5E sparse matching on TTK canonical top-left 160×160 speed persistence diagrams.
Radii **4, 8, 16, 32, 64, 96, 128, ∞ are WTK grid cells, not kilometres**.

Lower constrained W₂,₂ means better persistence matching; GT persistence-mass coverage is separate (higher is better).

| Model | r4 W22 ↓ | r16 W22 ↓ | r32 W22 ↓ | r∞ W22 ↓ | GT coverage r16 ↑ |
|---|---:|---:|---:|---:|---:|
| cnn | 24.819860 | 19.749577 | 16.293934 | 12.273587 | 0.518795 |
| gan | 34.329340 | 26.640915 | 21.733965 | 16.260901 | 0.818417 |
| grad | 23.989181 | 17.564704 | 14.055091 | 10.176955 | 0.755544 |
| cv | 24.688211 | 18.275539 | 14.456521 | 10.161532 | 0.685652 |
| pers | 24.822507 | 18.269998 | 14.382374 | 10.188375 | 0.709142 |
| f1 | 24.861093 | 18.371783 | 14.408071 | 10.258091 | 0.683339 |

## Daily descriptive summaries (24 consecutive hours per UTC date)

| UTC day | Arm | N | r4 W22 ↓ | r16 W22 ↓ | r∞ W22 ↓ | GT coverage r16 ↑ |
|---|---|---:|---:|---:|---:|---:|
| 2010-05-01 | cnn | 24 | 23.720635 | 19.117328 | 10.956488 | 0.526930 |
| 2010-05-01 | gan | 24 | 34.985135 | 28.887359 | 19.538030 | 0.785227 |
| 2010-05-01 | grad | 24 | 22.081081 | 16.729832 | 9.241969 | 0.775955 |
| 2010-05-01 | cv | 24 | 22.986950 | 17.378665 | 9.151673 | 0.681841 |
| 2010-05-01 | pers | 24 | 23.122052 | 17.153831 | 9.011131 | 0.720394 |
| 2010-05-01 | f1 | 24 | 22.997264 | 17.366984 | 9.160667 | 0.689048 |
| 2010-05-02 | cnn | 24 | 24.990711 | 20.084664 | 13.090971 | 0.503313 |
| 2010-05-02 | gan | 24 | 36.123884 | 28.554366 | 18.184565 | 0.806774 |
| 2010-05-02 | grad | 24 | 24.155374 | 17.600478 | 10.375289 | 0.781893 |
| 2010-05-02 | cv | 24 | 25.210056 | 18.566402 | 10.576646 | 0.695971 |
| 2010-05-02 | pers | 24 | 25.479309 | 18.730549 | 10.495246 | 0.730670 |
| 2010-05-02 | f1 | 24 | 25.813991 | 18.989136 | 10.785143 | 0.698026 |
| 2010-05-03 | cnn | 24 | 25.031018 | 19.988923 | 12.167873 | 0.513090 |
| 2010-05-03 | gan | 24 | 35.776670 | 27.979513 | 16.659398 | 0.803274 |
| 2010-05-03 | grad | 24 | 24.362012 | 18.238011 | 10.236273 | 0.727678 |
| 2010-05-03 | cv | 24 | 24.965342 | 18.800879 | 10.073134 | 0.675193 |
| 2010-05-03 | pers | 24 | 24.894552 | 18.588613 | 10.143832 | 0.695890 |
| 2010-05-03 | f1 | 24 | 24.792124 | 18.728022 | 10.184355 | 0.672741 |
| 2010-05-04 | cnn | 24 | 31.211377 | 23.992705 | 13.376680 | 0.560792 |
| 2010-05-04 | gan | 24 | 39.777083 | 30.392355 | 17.611639 | 0.815956 |
| 2010-05-04 | grad | 24 | 29.360704 | 20.997089 | 11.271180 | 0.770683 |
| 2010-05-04 | cv | 24 | 30.170139 | 21.779163 | 11.212993 | 0.707059 |
| 2010-05-04 | pers | 24 | 30.908512 | 21.964802 | 11.267798 | 0.721507 |
| 2010-05-04 | f1 | 24 | 30.803552 | 22.070571 | 11.263903 | 0.705800 |
| 2010-05-05 | cnn | 24 | 20.988010 | 17.011818 | 11.143187 | 0.495232 |
| 2010-05-05 | gan | 24 | 28.848070 | 22.161040 | 13.523747 | 0.837818 |
| 2010-05-05 | grad | 24 | 20.315515 | 14.422463 | 9.112019 | 0.764625 |
| 2010-05-05 | cv | 24 | 20.763523 | 14.899276 | 9.105319 | 0.690987 |
| 2010-05-05 | pers | 24 | 20.819296 | 15.162931 | 9.197223 | 0.712342 |
| 2010-05-05 | f1 | 24 | 20.954797 | 14.913134 | 9.183357 | 0.684472 |
| 2010-05-06 | cnn | 24 | 24.631079 | 19.673515 | 12.944745 | 0.516982 |
| 2010-05-06 | gan | 24 | 33.451222 | 25.022622 | 14.746806 | 0.834507 |
| 2010-05-06 | grad | 24 | 24.754599 | 18.247119 | 10.685215 | 0.726219 |
| 2010-05-06 | cv | 24 | 25.313051 | 19.097128 | 10.722990 | 0.666322 |
| 2010-05-06 | pers | 24 | 25.220193 | 18.927083 | 10.854175 | 0.684954 |
| 2010-05-06 | f1 | 24 | 25.294367 | 18.887996 | 10.866513 | 0.666841 |
| 2010-05-07 | cnn | 24 | 23.166189 | 18.378083 | 12.235164 | 0.515227 |
| 2010-05-07 | gan | 24 | 31.343316 | 23.489147 | 13.562123 | 0.845362 |
| 2010-05-07 | grad | 24 | 22.894984 | 16.717936 | 10.316740 | 0.741752 |
| 2010-05-07 | cv | 24 | 23.408414 | 17.407257 | 10.287967 | 0.682190 |
| 2010-05-07 | pers | 24 | 23.313637 | 17.362179 | 10.349219 | 0.698238 |
| 2010-05-07 | f1 | 24 | 23.371556 | 17.646639 | 10.362700 | 0.666448 |

## Scientific and reproducibility boundaries

- All 1,008 infinite-radius W22 model-to-GT values must match the sealed Part CIII corrected GUDHI output.
- All 96 aggregate + 192 dimension rows for historical week hours 0 and 84 must reproduce Part XCIX.
- Canonical zero/near-zero persistence pairs use the frozen original Phase5E tolerance 1e-10 with excluded count validation.
- All 1,176 diagram coordinate sets are validated against their corrected numerical birth/death features.
- GT persistence-mass coverage measures matched mass, not correct spatial positions or physical wind structure.
- This is not topology on full 500×500 fields and does not include merge trees or training.
- No external locked-region predictions or new hyperparameter/model selection. Original pretrained WTK exposure unknown.
- Verifier repeats the same original Phase5E algorithm; it is not an independently implemented solver.

