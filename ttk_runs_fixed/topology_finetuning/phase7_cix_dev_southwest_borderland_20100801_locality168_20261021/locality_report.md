# Phase 7 Part CIV — 168-hour development spatial-locality analysis

Southwest Borderland, 2010-08 1–7, 2010 UTC; 168 contiguous hourly fields × six frozen models.
**One development region-week. Consecutive hours/days are temporally correlated, not independent generalization replications.**

Unchanged original Phase5E sparse matching on TTK canonical top-left 160×160 speed persistence diagrams.
Radii **4, 8, 16, 32, 64, 96, 128, ∞ are WTK grid cells, not kilometres**.

Lower constrained W₂,₂ means better persistence matching; GT persistence-mass coverage is separate (higher is better).

| Model | r4 W22 ↓ | r16 W22 ↓ | r32 W22 ↓ | r∞ W22 ↓ | GT coverage r16 ↑ |
|---|---:|---:|---:|---:|---:|
| cnn | 21.676553 | 17.209153 | 14.494040 | 11.269066 | 0.485026 |
| gan | 30.543941 | 23.577811 | 19.159047 | 14.396649 | 0.811476 |
| grad | 21.265824 | 15.499961 | 12.731715 | 9.173748 | 0.735738 |
| cv | 21.588703 | 15.673302 | 12.769636 | 9.200345 | 0.680462 |
| pers | 21.722661 | 15.857388 | 12.932969 | 9.291722 | 0.704095 |
| f1 | 21.696970 | 15.783581 | 12.850215 | 9.300173 | 0.673434 |

## Daily descriptive summaries (24 consecutive hours per UTC date)

| UTC day | Arm | N | r4 W22 ↓ | r16 W22 ↓ | r∞ W22 ↓ | GT coverage r16 ↑ |
|---|---|---:|---:|---:|---:|---:|
| 2010-08-01 | cnn | 24 | 20.165162 | 16.128320 | 10.990957 | 0.507599 |
| 2010-08-01 | gan | 24 | 28.081776 | 21.114290 | 13.445963 | 0.845489 |
| 2010-08-01 | grad | 24 | 19.784346 | 14.759917 | 8.853921 | 0.752271 |
| 2010-08-01 | cv | 24 | 20.082499 | 14.510717 | 8.867254 | 0.698541 |
| 2010-08-01 | pers | 24 | 20.266759 | 14.716872 | 9.114444 | 0.707271 |
| 2010-08-01 | f1 | 24 | 20.214275 | 14.699746 | 8.990379 | 0.679970 |
| 2010-08-02 | cnn | 24 | 18.859347 | 14.286884 | 9.773849 | 0.478385 |
| 2010-08-02 | gan | 24 | 27.796665 | 21.926070 | 14.565100 | 0.812102 |
| 2010-08-02 | grad | 24 | 18.796398 | 13.273385 | 8.087574 | 0.739461 |
| 2010-08-02 | cv | 24 | 18.808757 | 13.548609 | 8.021186 | 0.687600 |
| 2010-08-02 | pers | 24 | 19.086627 | 13.867802 | 8.141023 | 0.699176 |
| 2010-08-02 | f1 | 24 | 19.038571 | 13.779148 | 8.095625 | 0.672432 |
| 2010-08-03 | cnn | 24 | 21.499724 | 16.983597 | 10.814144 | 0.465506 |
| 2010-08-03 | gan | 24 | 30.276190 | 23.948423 | 14.337935 | 0.784201 |
| 2010-08-03 | grad | 24 | 20.855859 | 15.161486 | 8.918021 | 0.721565 |
| 2010-08-03 | cv | 24 | 21.321479 | 15.216789 | 8.838770 | 0.657997 |
| 2010-08-03 | pers | 24 | 21.517305 | 15.309404 | 8.869916 | 0.700407 |
| 2010-08-03 | f1 | 24 | 21.422110 | 15.547865 | 8.858241 | 0.662122 |
| 2010-08-04 | cnn | 24 | 23.641819 | 18.450666 | 11.253898 | 0.480506 |
| 2010-08-04 | gan | 24 | 32.463251 | 25.362302 | 14.974190 | 0.786913 |
| 2010-08-04 | grad | 24 | 22.486378 | 16.565826 | 9.387818 | 0.729023 |
| 2010-08-04 | cv | 24 | 22.959883 | 16.663157 | 9.260638 | 0.671975 |
| 2010-08-04 | pers | 24 | 23.179283 | 17.271900 | 9.276809 | 0.697318 |
| 2010-08-04 | f1 | 24 | 23.199561 | 16.744641 | 9.316484 | 0.671437 |
| 2010-08-05 | cnn | 24 | 23.544917 | 18.599206 | 11.664931 | 0.491241 |
| 2010-08-05 | gan | 24 | 32.957508 | 25.849863 | 15.892309 | 0.803231 |
| 2010-08-05 | grad | 24 | 23.044354 | 16.734803 | 9.697245 | 0.731892 |
| 2010-08-05 | cv | 24 | 23.317667 | 16.967901 | 9.680315 | 0.673320 |
| 2010-08-05 | pers | 24 | 23.471332 | 17.002439 | 9.641416 | 0.701102 |
| 2010-08-05 | f1 | 24 | 23.481800 | 16.759207 | 9.766956 | 0.671739 |
| 2010-08-06 | cnn | 24 | 23.632767 | 19.010175 | 12.822615 | 0.479830 |
| 2010-08-06 | gan | 24 | 32.514942 | 23.887250 | 13.727667 | 0.825590 |
| 2010-08-06 | grad | 24 | 23.566889 | 17.015141 | 10.322045 | 0.737457 |
| 2010-08-06 | cv | 24 | 23.931925 | 17.451001 | 10.456172 | 0.684002 |
| 2010-08-06 | pers | 24 | 23.915661 | 17.304232 | 10.520591 | 0.709922 |
| 2010-08-06 | f1 | 24 | 23.991986 | 17.380585 | 10.593666 | 0.682177 |
| 2010-08-07 | cnn | 24 | 20.392135 | 17.005224 | 11.563066 | 0.492114 |
| 2010-08-07 | gan | 24 | 29.717259 | 22.956481 | 13.833380 | 0.822802 |
| 2010-08-07 | grad | 24 | 20.326544 | 14.989172 | 8.949614 | 0.738501 |
| 2010-08-07 | cv | 24 | 20.698710 | 15.354939 | 9.278080 | 0.689800 |
| 2010-08-07 | pers | 24 | 20.621661 | 15.529070 | 9.477855 | 0.713469 |
| 2010-08-07 | f1 | 24 | 20.530488 | 15.573875 | 9.479862 | 0.674161 |

## Scientific and reproducibility boundaries

- All 1,008 infinite-radius W22 model-to-GT values must match the sealed Part CIII corrected GUDHI output.
- All 96 aggregate + 192 dimension rows for historical week hours 0 and 84 must reproduce Part XCIX.
- Canonical zero/near-zero persistence pairs use the frozen original Phase5E tolerance 1e-10 with excluded count validation.
- All 1,176 diagram coordinate sets are validated against their corrected numerical birth/death features.
- GT persistence-mass coverage measures matched mass, not correct spatial positions or physical wind structure.
- This is not topology on full 500×500 fields and does not include merge trees or training.
- No external locked-region predictions or new hyperparameter/model selection. Original pretrained WTK exposure unknown.
- Verifier repeats the same original Phase5E algorithm; it is not an independently implemented solver.

