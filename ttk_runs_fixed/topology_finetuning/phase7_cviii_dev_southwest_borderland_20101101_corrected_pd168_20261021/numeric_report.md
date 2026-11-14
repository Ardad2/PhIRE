# Phase 7 Part CIII — corrected GUDHI distances, 168-hour development week

Southwest Borderland, 2010-11 1–7 2010 UTC, 168 consecutive hourly samples.
**This is one development region-week, not 168 independent generalization replicates.**

1176 numerically decoded canonical TTK diagrams; 1008 model-to-GT comparisons,
using the exact Phase6 corrected GUDHI functions, with historical 2007 and XCVIII hour-0/84 regressions.

| Model | Mean hourly dB ↓ | Mean hourly W2inf ↓ | Mean hourly W22 ↓ |
|---|---:|---:|---:|
| cnn | 2.342061 | 9.926023 | 12.311039 |
| gan | 3.997939 | 12.623548 | 15.668255 |
| grad | 1.840135 | 8.348800 | 10.145199 |
| cv | 1.908359 | 8.307390 | 10.047159 |
| pers | 1.979588 | 8.356561 | 10.083989 |
| f1 | 1.957231 | 8.343213 | 10.073751 |

## Daily descriptive means

| Day UTC | Model | N hours | dB ↓ | W2inf ↓ | W22 ↓ |
|---|---|---:|---:|---:|---:|
| 2010-11-01 | cnn | 24 | 2.494253 | 11.828230 | 14.635893 |
| 2010-11-01 | gan | 24 | 3.589867 | 14.328214 | 17.609065 |
| 2010-11-01 | grad | 24 | 2.101940 | 10.595914 | 12.923442 |
| 2010-11-01 | cv | 24 | 2.085209 | 10.272299 | 12.466202 |
| 2010-11-01 | pers | 24 | 2.197899 | 10.325747 | 12.494942 |
| 2010-11-01 | f1 | 24 | 2.151194 | 10.291572 | 12.436433 |
| 2010-11-02 | cnn | 24 | 3.016191 | 11.706833 | 14.405962 |
| 2010-11-02 | gan | 24 | 3.665766 | 14.168614 | 17.505127 |
| 2010-11-02 | grad | 24 | 2.313766 | 9.982621 | 12.093672 |
| 2010-11-02 | cv | 24 | 2.381685 | 9.660281 | 11.581252 |
| 2010-11-02 | pers | 24 | 2.563469 | 9.707566 | 11.626405 |
| 2010-11-02 | f1 | 24 | 2.487962 | 9.575938 | 11.466820 |
| 2010-11-03 | cnn | 24 | 2.508691 | 11.301982 | 13.918234 |
| 2010-11-03 | gan | 24 | 5.080427 | 15.231006 | 18.747469 |
| 2010-11-03 | grad | 24 | 2.258250 | 9.951784 | 12.047355 |
| 2010-11-03 | cv | 24 | 2.563045 | 10.021995 | 11.995443 |
| 2010-11-03 | pers | 24 | 2.358817 | 9.796169 | 11.685620 |
| 2010-11-03 | f1 | 24 | 2.501290 | 10.002591 | 11.954472 |
| 2010-11-04 | cnn | 24 | 2.486124 | 10.502723 | 12.835768 |
| 2010-11-04 | gan | 24 | 4.503142 | 13.683148 | 16.713884 |
| 2010-11-04 | grad | 24 | 2.280637 | 8.991240 | 10.758295 |
| 2010-11-04 | cv | 24 | 2.226101 | 8.919731 | 10.631281 |
| 2010-11-04 | pers | 24 | 2.361790 | 9.098986 | 10.833094 |
| 2010-11-04 | f1 | 24 | 2.301538 | 9.040095 | 10.778228 |
| 2010-11-05 | cnn | 24 | 2.486310 | 8.647090 | 10.600610 |
| 2010-11-05 | gan | 24 | 4.746204 | 12.910384 | 16.451080 |
| 2010-11-05 | grad | 24 | 1.408355 | 6.445124 | 7.797881 |
| 2010-11-05 | cv | 24 | 1.616740 | 6.673543 | 8.076092 |
| 2010-11-05 | pers | 24 | 1.744303 | 6.854082 | 8.263881 |
| 2010-11-05 | f1 | 24 | 1.738792 | 6.815560 | 8.221402 |
| 2010-11-06 | cnn | 24 | 1.471480 | 7.468649 | 9.573531 |
| 2010-11-06 | gan | 24 | 3.516583 | 8.660992 | 10.928192 |
| 2010-11-06 | grad | 24 | 1.124475 | 5.913550 | 7.324874 |
| 2010-11-06 | cv | 24 | 1.174012 | 6.016871 | 7.461230 |
| 2010-11-06 | pers | 24 | 1.184242 | 6.051542 | 7.516318 |
| 2010-11-06 | f1 | 24 | 1.141217 | 6.059167 | 7.525697 |
| 2010-11-07 | cnn | 24 | 1.931375 | 8.026650 | 10.207273 |
| 2010-11-07 | gan | 24 | 2.883582 | 9.382479 | 11.722967 |
| 2010-11-07 | grad | 24 | 1.393519 | 6.561368 | 8.070872 |
| 2010-11-07 | cv | 24 | 1.311721 | 6.587010 | 8.118612 |
| 2010-11-07 | pers | 24 | 1.446594 | 6.661833 | 8.167663 |
| 2010-11-07 | f1 | 24 | 1.378624 | 6.617571 | 8.133206 |

**Scope and limitations:** numeric topology concerns only the unchanged top-left 160×160 scalar-speed crop,
not the full 500×500 vector field. Day-level summaries describe correlated hours, not independent statistical blocks.
Verification recomputes every value with the same trusted GUDHI decoder and distance implementation; this is not an
independently implemented numerical algorithm. No Phase5E locality, merge trees, training, model selection,
locked-region inference or external geographic generalization is performed in CIII.
Original PhIRE pretrained WTK spatial/temporal exposure is unverified.

