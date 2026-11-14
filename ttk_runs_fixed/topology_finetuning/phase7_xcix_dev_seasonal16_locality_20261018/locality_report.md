# Phase 7 XCIX — spatial locality for 16 development-only wind fields

Original, unchanged Phase5E sparse locality matching, canonical top-left 160×160 speed crop.
16 predeclared 2010 development snapshots, two per region-week in two regions.
**Descriptive engineering data, NOT 168-hour regional/seasonal generalization statistics.**

Lower spatially constrained W₂,₂ is better; GT persistence-mass coverage is separate and higher is better.
All radii are **WTK grid cells**, not kilometers.

| Model | r4 W22 ↓ | r16 W22 ↓ | r32 W22 ↓ | r∞ W22 ↓ | r16 GT coverage ↑ |
|---|---:|---:|---:|---:|---:|
| cnn | 19.523399 | 15.817982 | 13.593439 | 10.462440 | 0.489421 |
| gan | 26.324920 | 20.457178 | 16.810830 | 12.211472 | 0.798443 |
| grad | 18.869274 | 14.002953 | 11.729558 | 8.375874 | 0.792718 |
| cv | 19.407128 | 14.623993 | 11.965772 | 8.502258 | 0.741817 |
| pers | 19.712260 | 14.615091 | 12.088795 | 8.483273 | 0.747704 |
| f1 | 19.570265 | 14.393966 | 11.853859 | 8.607247 | 0.724932 |

## Region-week means (two fields each)

| Region | 2010 week | Model | r4 W22 ↓ | r16 W22 ↓ | r∞ W22 ↓ | GT coverage r16 ↑ |
|---|---|---|---:|---:|---:|---:|
| gulf_inland | 2010-02-01 | cnn | 19.93045 | 15.70620 | 10.66159 | 0.60118 |
| gulf_inland | 2010-02-01 | gan | 23.73765 | 17.32724 | 10.67738 | 0.80022 |
| gulf_inland | 2010-02-01 | grad | 18.20804 | 13.06316 | 7.93018 | 0.84313 |
| gulf_inland | 2010-02-01 | cv | 18.92761 | 14.06459 | 8.61710 | 0.74542 |
| gulf_inland | 2010-02-01 | pers | 18.89674 | 14.13604 | 8.64268 | 0.73655 |
| gulf_inland | 2010-02-01 | f1 | 19.25309 | 14.07017 | 8.74791 | 0.71129 |
| gulf_inland | 2010-05-01 | cnn | 13.09097 | 11.63969 | 8.05044 | 0.40748 |
| gulf_inland | 2010-05-01 | gan | 17.22141 | 13.05790 | 7.38467 | 0.73054 |
| gulf_inland | 2010-05-01 | grad | 13.87063 | 10.18191 | 6.52243 | 0.83311 |
| gulf_inland | 2010-05-01 | cv | 13.61240 | 10.28136 | 6.51161 | 0.79370 |
| gulf_inland | 2010-05-01 | pers | 13.76485 | 10.72212 | 6.47413 | 0.78452 |
| gulf_inland | 2010-05-01 | f1 | 13.83384 | 10.49642 | 6.75621 | 0.76851 |
| gulf_inland | 2010-08-01 | cnn | 10.63325 | 9.69842 | 7.51351 | 0.36890 |
| gulf_inland | 2010-08-01 | gan | 14.01205 | 10.91545 | 5.78625 | 0.75598 |
| gulf_inland | 2010-08-01 | grad | 10.84129 | 8.60372 | 5.40603 | 0.78229 |
| gulf_inland | 2010-08-01 | cv | 11.14214 | 8.92814 | 5.67189 | 0.75091 |
| gulf_inland | 2010-08-01 | pers | 11.15005 | 8.92435 | 5.64235 | 0.76602 |
| gulf_inland | 2010-08-01 | f1 | 11.15478 | 9.03502 | 5.81075 | 0.71042 |
| gulf_inland | 2010-11-01 | cnn | 11.99614 | 10.71287 | 8.27968 | 0.38734 |
| gulf_inland | 2010-11-01 | gan | 16.28517 | 12.32980 | 7.84937 | 0.83711 |
| gulf_inland | 2010-11-01 | grad | 13.09692 | 10.15903 | 6.76073 | 0.86633 |
| gulf_inland | 2010-11-01 | cv | 13.64491 | 10.55074 | 7.03448 | 0.83110 |
| gulf_inland | 2010-11-01 | pers | 13.22660 | 9.99715 | 6.41841 | 0.80857 |
| gulf_inland | 2010-11-01 | f1 | 13.53574 | 10.45654 | 6.87835 | 0.80256 |
| southwest_borderland | 2010-02-01 | cnn | 18.25955 | 14.48075 | 10.55991 | 0.52188 |
| southwest_borderland | 2010-02-01 | gan | 28.24265 | 21.07678 | 13.03732 | 0.85460 |
| southwest_borderland | 2010-02-01 | grad | 17.82291 | 13.10556 | 8.46021 | 0.80481 |
| southwest_borderland | 2010-02-01 | cv | 17.89435 | 13.24650 | 8.41284 | 0.74664 |
| southwest_borderland | 2010-02-01 | pers | 17.95450 | 13.24144 | 8.42840 | 0.75471 |
| southwest_borderland | 2010-02-01 | f1 | 17.72533 | 12.77192 | 8.47596 | 0.73739 |
| southwest_borderland | 2010-05-01 | cnn | 28.58571 | 21.64358 | 13.00791 | 0.58744 |
| southwest_borderland | 2010-05-01 | gan | 38.71238 | 31.44101 | 19.37919 | 0.79369 |
| southwest_borderland | 2010-05-01 | grad | 26.95600 | 19.24179 | 10.21384 | 0.78621 |
| southwest_borderland | 2010-05-01 | cv | 28.05834 | 19.50834 | 10.22621 | 0.73567 |
| southwest_borderland | 2010-05-01 | pers | 28.94573 | 19.75928 | 10.36200 | 0.75091 |
| southwest_borderland | 2010-05-01 | f1 | 28.23536 | 19.70019 | 10.37172 | 0.72606 |
| southwest_borderland | 2010-08-01 | cnn | 23.27785 | 19.31988 | 12.17461 | 0.49776 |
| southwest_borderland | 2010-08-01 | gan | 33.61122 | 27.39261 | 17.01780 | 0.82081 |
| southwest_borderland | 2010-08-01 | grad | 22.12778 | 17.39439 | 10.25656 | 0.73960 |
| southwest_borderland | 2010-08-01 | cv | 22.19753 | 17.49525 | 10.14099 | 0.67849 |
| southwest_borderland | 2010-08-01 | pers | 23.85238 | 17.57152 | 9.98459 | 0.72101 |
| southwest_borderland | 2010-08-01 | f1 | 22.91431 | 15.65770 | 10.12254 | 0.70484 |
| southwest_borderland | 2010-11-01 | cnn | 30.41328 | 23.34246 | 13.45186 | 0.54339 |
| southwest_borderland | 2010-11-01 | gan | 38.77682 | 30.11663 | 16.55980 | 0.79460 |
| southwest_borderland | 2010-11-01 | grad | 28.03061 | 20.27407 | 11.45701 | 0.68626 |
| southwest_borderland | 2010-11-01 | cv | 29.77976 | 22.91703 | 11.40293 | 0.65262 |
| southwest_borderland | 2010-11-01 | pers | 29.90722 | 22.56882 | 11.91362 | 0.65934 |
| southwest_borderland | 2010-11-01 | f1 | 29.90966 | 22.96375 | 11.69453 | 0.63838 |

## Method and evidence boundaries

- Reuses the original Phase5E feature-coordinate reader and SparsePrepared assignment optimizer without source edits.
- Canonical zero/near-zero persistence pairs excluded using the original Phase5E tolerance, and counts compared in all 112 H0/H1 diagrams.
- All 96 infinity-radius W22 distances must reproduce frozen XCVIII corrected GUDHI values.
- All 192 aggregate and 384 dimension-level rows for the original four development fields must reproduce XCIV.
- Persistence-mass coverage is not equivalent to accurate physical placement of features.
- Geographical roles, all checkpoint parameters, and all eight locality radii remain frozen by XCI.
- These 16 fields are two chosen hours per week, not independent full-week meteorology or an external benchmark.
- Original PhIRE pretrained-model WTK exposure remains unverified; locked regions were not evaluated.

