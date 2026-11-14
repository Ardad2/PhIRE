# Phase 7 XCIII — corrected persistence distances (development-only)

Four selected fields from Gulf inland / Southwest borderland, February and August 2010.
Corrected Phase 6 canonical TTK decoder and GUDHI metrics; unchanged XCI protocol.
**These are descriptive smoke-test results on 4 development fields, not a generalization conclusion.**

| Model | dB mean ↓ | W2inf mean ↓ | W22 mean ↓ |
|---|---:|---:|---:|
| cnn | 1.541602 | 7.535081 | 9.560749 |
| gan | 3.191603 | 8.791652 | 11.388065 |
| grad | 1.301406 | 6.097335 | 7.412320 |
| cv | 1.221196 | 6.335671 | 7.806714 |
| pers | 1.309084 | 6.427607 | 7.919260 |
| f1 | 1.259638 | 6.400259 | 7.905475 |

## Sample-level results

| Region | Sample | Model | dB ↓ | W2inf ↓ | W22 ↓ |
|---|---:|---|---:|---:|---:|
| gulf_inland | 0 | cnn | 0.956505 | 7.473528 | 9.566773 |
| gulf_inland | 0 | gan | 0.915970 | 5.620726 | 6.826467 |
| gulf_inland | 0 | grad | 0.754823 | 5.631466 | 6.898749 |
| gulf_inland | 0 | cv | 0.703755 | 6.104265 | 7.568729 |
| gulf_inland | 0 | pers | 0.728603 | 6.049191 | 7.496531 |
| gulf_inland | 0 | f1 | 0.795945 | 6.121240 | 7.681889 |
| gulf_inland | 1 | cnn | 0.871560 | 5.589499 | 7.338493 |
| gulf_inland | 1 | gan | 1.964610 | 5.767463 | 7.335569 |
| gulf_inland | 1 | grad | 0.918666 | 4.009392 | 4.944503 |
| gulf_inland | 1 | cv | 0.895115 | 4.371828 | 5.477133 |
| gulf_inland | 1 | pers | 0.868369 | 4.378554 | 5.445667 |
| gulf_inland | 1 | f1 | 0.835142 | 4.461333 | 5.617662 |
| southwest_borderland | 2 | cnn | 2.119957 | 7.539590 | 9.306957 |
| southwest_borderland | 2 | gan | 4.250786 | 10.353819 | 13.538811 |
| southwest_borderland | 2 | grad | 1.348323 | 6.231740 | 7.420990 |
| southwest_borderland | 2 | cv | 1.471618 | 6.442978 | 7.646477 |
| southwest_borderland | 2 | pers | 1.407103 | 6.600511 | 7.968145 |
| southwest_borderland | 2 | f1 | 1.427228 | 6.464721 | 7.744004 |
| southwest_borderland | 3 | cnn | 2.218386 | 9.537704 | 12.030773 |
| southwest_borderland | 3 | gan | 5.635047 | 13.424601 | 17.851412 |
| southwest_borderland | 3 | grad | 2.183811 | 8.516741 | 10.385039 |
| southwest_borderland | 3 | cv | 1.814294 | 8.423613 | 10.534517 |
| southwest_borderland | 3 | pers | 2.232262 | 8.682172 | 10.766699 |
| southwest_borderland | 3 | f1 | 1.980238 | 8.553742 | 10.578345 |

## Boundaries

- Distances are from 160×160 top-left patches only, not full 500×500 topology.
- Diagram decomposition: H0/H1 combined using the original Phase 6 functions.
- Numeric verification reruns the same validated implementation; it is not a second independent algorithm.
- No locality scores, merge-tree scores, new training or external-test fields in this workpack.
- Four fields are neither independent seasonal replicates nor a geographic generalization benchmark.

