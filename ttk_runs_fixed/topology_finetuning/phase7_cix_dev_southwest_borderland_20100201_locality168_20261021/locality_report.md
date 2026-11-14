# Phase 7 Part CIV — 168-hour development spatial-locality analysis

Southwest Borderland, 2010-02 1–7, 2010 UTC; 168 contiguous hourly fields × six frozen models.
**One development region-week. Consecutive hours/days are temporally correlated, not independent generalization replications.**

Unchanged original Phase5E sparse matching on TTK canonical top-left 160×160 speed persistence diagrams.
Radii **4, 8, 16, 32, 64, 96, 128, ∞ are WTK grid cells, not kilometres**.

Lower constrained W₂,₂ means better persistence matching; GT persistence-mass coverage is separate (higher is better).

| Model | r4 W22 ↓ | r16 W22 ↓ | r32 W22 ↓ | r∞ W22 ↓ | GT coverage r16 ↑ |
|---|---:|---:|---:|---:|---:|
| cnn | 20.147683 | 16.434888 | 14.014028 | 11.252322 | 0.481628 |
| gan | 28.937510 | 22.097818 | 17.777263 | 13.187466 | 0.827128 |
| grad | 19.865759 | 14.509567 | 11.902651 | 9.035153 | 0.774080 |
| cv | 20.232538 | 14.907889 | 12.073039 | 9.067438 | 0.702710 |
| pers | 20.350028 | 14.934804 | 12.161505 | 9.014435 | 0.729217 |
| f1 | 20.414096 | 15.010816 | 12.169053 | 9.126462 | 0.696254 |

## Daily descriptive summaries (24 consecutive hours per UTC date)

| UTC day | Arm | N | r4 W22 ↓ | r16 W22 ↓ | r∞ W22 ↓ | GT coverage r16 ↑ |
|---|---|---:|---:|---:|---:|---:|
| 2010-02-01 | cnn | 24 | 16.271474 | 13.490982 | 9.690320 | 0.465372 |
| 2010-02-01 | gan | 24 | 24.139694 | 18.251191 | 11.047225 | 0.833264 |
| 2010-02-01 | grad | 24 | 16.264409 | 11.979755 | 7.567805 | 0.769643 |
| 2010-02-01 | cv | 24 | 16.432877 | 12.284513 | 7.823755 | 0.698533 |
| 2010-02-01 | pers | 24 | 16.414402 | 12.284903 | 7.845311 | 0.724614 |
| 2010-02-01 | f1 | 24 | 16.466951 | 12.558486 | 7.887144 | 0.680893 |
| 2010-02-02 | cnn | 24 | 16.117836 | 13.882577 | 10.005569 | 0.455603 |
| 2010-02-02 | gan | 24 | 23.395961 | 17.160590 | 9.665438 | 0.836186 |
| 2010-02-02 | grad | 24 | 16.138463 | 12.239847 | 7.702815 | 0.753370 |
| 2010-02-02 | cv | 24 | 16.277641 | 12.518575 | 7.961156 | 0.688837 |
| 2010-02-02 | pers | 24 | 16.299307 | 12.293355 | 8.013736 | 0.722570 |
| 2010-02-02 | f1 | 24 | 16.326354 | 12.504360 | 8.072818 | 0.678118 |
| 2010-02-03 | cnn | 24 | 25.442362 | 20.990399 | 13.476024 | 0.488456 |
| 2010-02-03 | gan | 24 | 35.353475 | 27.003895 | 15.246231 | 0.824129 |
| 2010-02-03 | grad | 24 | 25.016866 | 18.295013 | 10.689700 | 0.812128 |
| 2010-02-03 | cv | 24 | 25.335708 | 18.821783 | 10.466687 | 0.716388 |
| 2010-02-03 | pers | 24 | 25.599991 | 19.178381 | 10.236181 | 0.747734 |
| 2010-02-03 | f1 | 24 | 25.565829 | 19.053482 | 10.471734 | 0.717741 |
| 2010-02-04 | cnn | 24 | 19.066988 | 15.273989 | 10.443233 | 0.492386 |
| 2010-02-04 | gan | 24 | 27.430224 | 21.151792 | 12.569326 | 0.825454 |
| 2010-02-04 | grad | 24 | 18.346829 | 13.356071 | 8.474387 | 0.778017 |
| 2010-02-04 | cv | 24 | 18.856491 | 13.656068 | 8.438702 | 0.708424 |
| 2010-02-04 | pers | 24 | 19.198258 | 13.765447 | 8.365968 | 0.736936 |
| 2010-02-04 | f1 | 24 | 19.315582 | 13.759750 | 8.519711 | 0.705860 |
| 2010-02-05 | cnn | 24 | 17.615325 | 14.786536 | 10.708194 | 0.472281 |
| 2010-02-05 | gan | 24 | 25.474291 | 19.270763 | 11.388124 | 0.832949 |
| 2010-02-05 | grad | 24 | 17.384350 | 12.756862 | 8.428290 | 0.767868 |
| 2010-02-05 | cv | 24 | 17.852837 | 13.420740 | 8.760110 | 0.707581 |
| 2010-02-05 | pers | 24 | 17.854090 | 13.279172 | 8.757318 | 0.727525 |
| 2010-02-05 | f1 | 24 | 17.914353 | 13.509570 | 8.816800 | 0.693757 |
| 2010-02-06 | cnn | 24 | 20.806364 | 16.966221 | 11.267751 | 0.485405 |
| 2010-02-06 | gan | 24 | 30.313839 | 23.456458 | 14.750653 | 0.829846 |
| 2010-02-06 | grad | 24 | 20.183087 | 15.051820 | 9.519606 | 0.752379 |
| 2010-02-06 | cv | 24 | 20.879920 | 15.400505 | 9.564334 | 0.690884 |
| 2010-02-06 | pers | 24 | 20.944320 | 15.701679 | 9.474861 | 0.712911 |
| 2010-02-06 | f1 | 24 | 21.061364 | 15.454730 | 9.582255 | 0.689458 |
| 2010-02-07 | cnn | 24 | 25.713430 | 19.653514 | 13.175163 | 0.511892 |
| 2010-02-07 | gan | 24 | 36.455083 | 28.390036 | 17.645268 | 0.808071 |
| 2010-02-07 | grad | 24 | 25.726311 | 17.887601 | 10.863468 | 0.785152 |
| 2010-02-07 | cv | 24 | 25.992294 | 18.253037 | 10.457322 | 0.708323 |
| 2010-02-07 | pers | 24 | 26.139823 | 18.040693 | 10.407671 | 0.732232 |
| 2010-02-07 | f1 | 24 | 26.248236 | 18.235338 | 10.534776 | 0.707954 |

## Scientific and reproducibility boundaries

- All 1,008 infinite-radius W22 model-to-GT values must match the sealed Part CIII corrected GUDHI output.
- All 96 aggregate + 192 dimension rows for historical week hours 0 and 84 must reproduce Part XCIX.
- Canonical zero/near-zero persistence pairs use the frozen original Phase5E tolerance 1e-10 with excluded count validation.
- All 1,176 diagram coordinate sets are validated against their corrected numerical birth/death features.
- GT persistence-mass coverage measures matched mass, not correct spatial positions or physical wind structure.
- This is not topology on full 500×500 fields and does not include merge trees or training.
- No external locked-region predictions or new hyperparameter/model selection. Original pretrained WTK exposure unknown.
- Verifier repeats the same original Phase5E algorithm; it is not an independently implemented solver.

