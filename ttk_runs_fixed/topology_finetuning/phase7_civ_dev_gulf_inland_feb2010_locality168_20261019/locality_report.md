# Phase 7 Part CIV — 168-hour development spatial-locality analysis

Gulf inland, February 1–7, 2010 UTC; 168 contiguous hourly fields × six frozen models.
**One development region-week. Consecutive hours/days are temporally correlated, not independent generalization replications.**

Unchanged original Phase5E sparse matching on TTK canonical top-left 160×160 speed persistence diagrams.
Radii **4, 8, 16, 32, 64, 96, 128, ∞ are WTK grid cells, not kilometres**.

Lower constrained W₂,₂ means better persistence matching; GT persistence-mass coverage is separate (higher is better).

| Model | r4 W22 ↓ | r16 W22 ↓ | r32 W22 ↓ | r∞ W22 ↓ | GT coverage r16 ↑ |
|---|---:|---:|---:|---:|---:|
| cnn | 10.531914 | 9.022663 | 7.967261 | 6.662873 | 0.437959 |
| gan | 14.233724 | 10.966813 | 8.945736 | 6.633682 | 0.789305 |
| grad | 10.611706 | 7.644812 | 6.414692 | 4.993435 | 0.840827 |
| cv | 10.617959 | 7.816083 | 6.546095 | 5.090495 | 0.771318 |
| pers | 10.683903 | 7.874390 | 6.580390 | 5.154307 | 0.771487 |
| f1 | 10.683088 | 7.973749 | 6.700959 | 5.228484 | 0.749903 |

## Daily descriptive summaries (24 consecutive hours per UTC date)

| UTC day | Arm | N | r4 W22 ↓ | r16 W22 ↓ | r∞ W22 ↓ | GT coverage r16 ↑ |
|---|---|---:|---:|---:|---:|---:|
| 2010-02-01 | cnn | 24 | 13.563409 | 10.591224 | 6.666202 | 0.550591 |
| 2010-02-01 | gan | 24 | 16.596833 | 12.169358 | 7.068801 | 0.792775 |
| 2010-02-01 | grad | 24 | 12.445184 | 8.248133 | 4.818149 | 0.845376 |
| 2010-02-01 | cv | 24 | 12.642249 | 8.641595 | 5.017137 | 0.746855 |
| 2010-02-01 | pers | 24 | 12.624061 | 8.557697 | 4.938293 | 0.740285 |
| 2010-02-01 | f1 | 24 | 12.830803 | 8.851925 | 5.072048 | 0.700893 |
| 2010-02-02 | cnn | 24 | 10.934053 | 9.714709 | 7.684686 | 0.441311 |
| 2010-02-02 | gan | 24 | 13.671689 | 10.312940 | 6.029745 | 0.807794 |
| 2010-02-02 | grad | 24 | 11.370819 | 8.104375 | 5.036779 | 0.881812 |
| 2010-02-02 | cv | 24 | 11.362126 | 8.169162 | 5.407817 | 0.797672 |
| 2010-02-02 | pers | 24 | 11.362764 | 8.337285 | 5.394156 | 0.786039 |
| 2010-02-02 | f1 | 24 | 11.391177 | 8.495141 | 5.517940 | 0.759211 |
| 2010-02-03 | cnn | 24 | 7.039523 | 6.452181 | 5.268312 | 0.379613 |
| 2010-02-03 | gan | 24 | 10.918298 | 8.835929 | 6.094571 | 0.774918 |
| 2010-02-03 | grad | 24 | 7.660933 | 5.873528 | 4.287849 | 0.790018 |
| 2010-02-03 | cv | 24 | 7.563110 | 5.684207 | 4.057136 | 0.745054 |
| 2010-02-03 | pers | 24 | 7.773726 | 6.012046 | 4.369886 | 0.727538 |
| 2010-02-03 | f1 | 24 | 7.506395 | 5.699394 | 4.130586 | 0.718502 |
| 2010-02-04 | cnn | 24 | 16.513932 | 13.712141 | 9.344390 | 0.485802 |
| 2010-02-04 | gan | 24 | 21.526093 | 16.537848 | 9.826274 | 0.809841 |
| 2010-02-04 | grad | 24 | 15.482838 | 11.201298 | 7.359081 | 0.804796 |
| 2010-02-04 | cv | 24 | 15.579024 | 11.687823 | 7.623293 | 0.729272 |
| 2010-02-04 | pers | 24 | 15.721169 | 11.767728 | 7.712394 | 0.723922 |
| 2010-02-04 | f1 | 24 | 15.702194 | 11.839289 | 7.806125 | 0.699128 |
| 2010-02-05 | cnn | 24 | 11.405800 | 9.772276 | 7.360806 | 0.472603 |
| 2010-02-05 | gan | 24 | 16.026600 | 11.837289 | 6.404961 | 0.819501 |
| 2010-02-05 | grad | 24 | 12.063810 | 8.633827 | 5.751908 | 0.883674 |
| 2010-02-05 | cv | 24 | 11.907957 | 8.590869 | 5.335445 | 0.813137 |
| 2010-02-05 | pers | 24 | 12.097888 | 8.559067 | 5.464261 | 0.844563 |
| 2010-02-05 | f1 | 24 | 11.989246 | 8.669686 | 5.518704 | 0.825944 |
| 2010-02-06 | cnn | 24 | 6.608495 | 5.833239 | 4.403196 | 0.384126 |
| 2010-02-06 | gan | 24 | 10.965825 | 9.239822 | 6.193273 | 0.775486 |
| 2010-02-06 | grad | 24 | 6.940775 | 5.225859 | 3.439493 | 0.860581 |
| 2010-02-06 | cv | 24 | 7.131308 | 5.477548 | 3.633331 | 0.833015 |
| 2010-02-06 | pers | 24 | 7.081191 | 5.389531 | 3.602271 | 0.851208 |
| 2010-02-06 | f1 | 24 | 7.240437 | 5.681335 | 3.845248 | 0.842149 |
| 2010-02-07 | cnn | 24 | 7.658184 | 7.082870 | 5.912520 | 0.351666 |
| 2010-02-07 | gan | 24 | 9.930733 | 7.834501 | 4.818147 | 0.744819 |
| 2010-02-07 | grad | 24 | 8.317583 | 6.226665 | 4.260783 | 0.819534 |
| 2010-02-07 | cv | 24 | 8.139940 | 6.461379 | 4.559308 | 0.734218 |
| 2010-02-07 | pers | 24 | 8.126519 | 6.497376 | 4.598887 | 0.726856 |
| 2010-02-07 | f1 | 24 | 8.121366 | 6.579472 | 4.708739 | 0.703492 |

## Scientific and reproducibility boundaries

- All 1,008 infinite-radius W22 model-to-GT values must match the sealed Part CIII corrected GUDHI output.
- All 96 aggregate + 192 dimension rows for historical week hours 0 and 84 must reproduce Part XCIX.
- Canonical zero/near-zero persistence pairs use the frozen original Phase5E tolerance 1e-10 with excluded count validation.
- All 1,176 diagram coordinate sets are validated against their corrected numerical birth/death features.
- GT persistence-mass coverage measures matched mass, not correct spatial positions or physical wind structure.
- This is not topology on full 500×500 fields and does not include merge trees or training.
- No external locked-region predictions or new hyperparameter/model selection. Original pretrained WTK exposure unknown.
- Verifier repeats the same original Phase5E algorithm; it is not an independently implemented solver.

