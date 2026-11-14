# Phase 7 Part CIV — 168-hour development spatial-locality analysis

Gulf Inland, 2010-05 1–7, 2010 UTC; 168 contiguous hourly fields × six frozen models.
**One development region-week. Consecutive hours/days are temporally correlated, not independent generalization replications.**

Unchanged original Phase5E sparse matching on TTK canonical top-left 160×160 speed persistence diagrams.
Radii **4, 8, 16, 32, 64, 96, 128, ∞ are WTK grid cells, not kilometres**.

Lower constrained W₂,₂ means better persistence matching; GT persistence-mass coverage is separate (higher is better).

| Model | r4 W22 ↓ | r16 W22 ↓ | r32 W22 ↓ | r∞ W22 ↓ | GT coverage r16 ↑ |
|---|---:|---:|---:|---:|---:|
| cnn | 12.785880 | 10.938897 | 9.675959 | 8.194879 | 0.423084 |
| gan | 16.829878 | 12.417936 | 9.998048 | 7.371158 | 0.814536 |
| grad | 12.637182 | 9.243755 | 7.838590 | 6.319847 | 0.830186 |
| cv | 12.672468 | 9.516135 | 8.071591 | 6.530926 | 0.776172 |
| pers | 12.663180 | 9.535601 | 8.064492 | 6.546583 | 0.782773 |
| f1 | 12.656715 | 9.686644 | 8.198756 | 6.699722 | 0.751184 |

## Daily descriptive summaries (24 consecutive hours per UTC date)

| UTC day | Arm | N | r4 W22 ↓ | r16 W22 ↓ | r∞ W22 ↓ | GT coverage r16 ↑ |
|---|---|---:|---:|---:|---:|---:|
| 2010-05-01 | cnn | 24 | 17.254324 | 13.927083 | 9.462943 | 0.574696 |
| 2010-05-01 | gan | 24 | 20.370891 | 14.352926 | 9.058935 | 0.853249 |
| 2010-05-01 | grad | 24 | 15.481511 | 11.330921 | 8.185516 | 0.869603 |
| 2010-05-01 | cv | 24 | 15.558355 | 11.244874 | 7.844724 | 0.807960 |
| 2010-05-01 | pers | 24 | 15.567488 | 11.348698 | 7.865256 | 0.816396 |
| 2010-05-01 | f1 | 24 | 15.569705 | 11.357256 | 8.010348 | 0.783704 |
| 2010-05-02 | cnn | 24 | 17.441562 | 14.349546 | 10.266473 | 0.486648 |
| 2010-05-02 | gan | 24 | 21.506651 | 15.415495 | 8.416213 | 0.799914 |
| 2010-05-02 | grad | 24 | 17.238792 | 12.827672 | 8.686149 | 0.810085 |
| 2010-05-02 | cv | 24 | 17.214545 | 13.035378 | 8.887815 | 0.746872 |
| 2010-05-02 | pers | 24 | 17.265398 | 12.986421 | 8.857723 | 0.771281 |
| 2010-05-02 | f1 | 24 | 17.279108 | 13.151348 | 9.111110 | 0.717800 |
| 2010-05-03 | cnn | 24 | 9.837518 | 8.542641 | 6.082130 | 0.382062 |
| 2010-05-03 | gan | 24 | 14.323563 | 11.452479 | 6.717880 | 0.759428 |
| 2010-05-03 | grad | 24 | 9.823180 | 7.049558 | 4.389911 | 0.788508 |
| 2010-05-03 | cv | 24 | 9.986514 | 7.537314 | 4.737556 | 0.738031 |
| 2010-05-03 | pers | 24 | 10.015338 | 7.582664 | 4.756453 | 0.747132 |
| 2010-05-03 | f1 | 24 | 10.057808 | 7.683356 | 4.819199 | 0.702188 |
| 2010-05-04 | cnn | 24 | 10.865189 | 9.348489 | 6.675844 | 0.425248 |
| 2010-05-04 | gan | 24 | 15.316993 | 12.364327 | 7.494094 | 0.741478 |
| 2010-05-04 | grad | 24 | 10.678744 | 7.719854 | 5.028001 | 0.814325 |
| 2010-05-04 | cv | 24 | 11.167855 | 8.201991 | 5.273034 | 0.764142 |
| 2010-05-04 | pers | 24 | 11.121227 | 8.130543 | 5.284625 | 0.785818 |
| 2010-05-04 | f1 | 24 | 11.197070 | 8.439140 | 5.324324 | 0.770810 |
| 2010-05-05 | cnn | 24 | 9.295608 | 8.038552 | 5.795216 | 0.399285 |
| 2010-05-05 | gan | 24 | 14.128135 | 10.998478 | 7.015459 | 0.830228 |
| 2010-05-05 | grad | 24 | 9.615882 | 7.015624 | 4.217209 | 0.826402 |
| 2010-05-05 | cv | 24 | 9.580954 | 7.095404 | 4.433814 | 0.771654 |
| 2010-05-05 | pers | 24 | 9.502368 | 7.215367 | 4.385460 | 0.764174 |
| 2010-05-05 | f1 | 24 | 9.384244 | 7.425266 | 4.553237 | 0.725056 |
| 2010-05-06 | cnn | 24 | 10.873391 | 10.073061 | 8.890319 | 0.292993 |
| 2010-05-06 | gan | 24 | 14.350667 | 10.038547 | 5.718469 | 0.865758 |
| 2010-05-06 | grad | 24 | 11.270704 | 8.202736 | 6.211481 | 0.852614 |
| 2010-05-06 | cv | 24 | 11.137219 | 8.555919 | 6.579356 | 0.820412 |
| 2010-05-06 | pers | 24 | 11.080977 | 8.612355 | 6.677841 | 0.807437 |
| 2010-05-06 | f1 | 24 | 11.084475 | 8.684668 | 6.771707 | 0.805466 |
| 2010-05-07 | cnn | 24 | 13.933571 | 12.292906 | 10.191227 | 0.400654 |
| 2010-05-07 | gan | 24 | 17.812244 | 12.303304 | 7.177057 | 0.851694 |
| 2010-05-07 | grad | 24 | 14.351464 | 10.559921 | 7.520665 | 0.849765 |
| 2010-05-07 | cv | 24 | 14.061834 | 10.942067 | 7.960185 | 0.784133 |
| 2010-05-07 | pers | 24 | 14.089466 | 10.873159 | 7.998723 | 0.787174 |
| 2010-05-07 | f1 | 24 | 14.024593 | 11.065473 | 8.308130 | 0.753260 |

## Scientific and reproducibility boundaries

- All 1,008 infinite-radius W22 model-to-GT values must match the sealed Part CIII corrected GUDHI output.
- All 96 aggregate + 192 dimension rows for historical week hours 0 and 84 must reproduce Part XCIX.
- Canonical zero/near-zero persistence pairs use the frozen original Phase5E tolerance 1e-10 with excluded count validation.
- All 1,176 diagram coordinate sets are validated against their corrected numerical birth/death features.
- GT persistence-mass coverage measures matched mass, not correct spatial positions or physical wind structure.
- This is not topology on full 500×500 fields and does not include merge trees or training.
- No external locked-region predictions or new hyperparameter/model selection. Original pretrained WTK exposure unknown.
- Verifier repeats the same original Phase5E algorithm; it is not an independently implemented solver.

