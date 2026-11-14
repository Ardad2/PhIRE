# Phase 7 XCVIII — 16-field corrected persistence distances (development only)

16 predeclared 2010 source fields (two development regions × four weeks × offsets 0 h / 84 h).
112 numerically decoded TTK port-0 diagrams; 96 model-to-GT comparisons via the **unchanged Phase6 GUDHI functions**.
**Two snapshots per region-week are engineering checks, not independent observations or a generalization benchmark.**

| Model | dB mean ↓ | W2inf mean ↓ | W22 mean ↓ |
|---|---:|---:|---:|
| cnn | 2.218899 | 8.349543 | 10.462440 |
| gan | 3.371328 | 9.788407 | 12.211472 |
| grad | 1.540991 | 6.892901 | 8.375874 |
| cv | 1.597790 | 6.981211 | 8.502258 |
| pers | 1.570721 | 6.955128 | 8.483273 |
| f1 | 1.603079 | 7.033517 | 8.607247 |

## Region–week summaries

Each entry aggregates **two predeclared fields**, 0 h and 84 h; do not interpret 16 hours as independent replications.

| Region | Week start | Model | dB ↓ | W2inf ↓ | W22 ↓ |
|---|---|---|---:|---:|---:|
| gulf_inland | 2010-02-01 | cnn | 3.775670 | 8.702617 | 10.661594 |
| gulf_inland | 2010-02-01 | gan | 3.478360 | 8.763101 | 10.677384 |
| gulf_inland | 2010-02-01 | grad | 1.265122 | 6.615094 | 7.930180 |
| gulf_inland | 2010-02-01 | cv | 1.486701 | 7.143508 | 8.617100 |
| gulf_inland | 2010-02-01 | pers | 1.426707 | 7.115222 | 8.642680 |
| gulf_inland | 2010-02-01 | f1 | 1.435218 | 7.149356 | 8.747914 |
| gulf_inland | 2010-05-01 | cnn | 1.257483 | 6.226622 | 8.050436 |
| gulf_inland | 2010-05-01 | gan | 1.435784 | 5.965068 | 7.384671 |
| gulf_inland | 2010-05-01 | grad | 1.334421 | 5.193930 | 6.522433 |
| gulf_inland | 2010-05-01 | cv | 1.317879 | 5.321402 | 6.511612 |
| gulf_inland | 2010-05-01 | pers | 1.236596 | 5.242218 | 6.474129 |
| gulf_inland | 2010-05-01 | f1 | 1.238719 | 5.470706 | 6.756209 |
| gulf_inland | 2010-08-01 | cnn | 1.114841 | 5.685891 | 7.513515 |
| gulf_inland | 2010-08-01 | gan | 1.427838 | 4.668582 | 5.786247 |
| gulf_inland | 2010-08-01 | grad | 1.006535 | 4.303887 | 5.406034 |
| gulf_inland | 2010-08-01 | cv | 1.001694 | 4.507268 | 5.671890 |
| gulf_inland | 2010-08-01 | pers | 0.973492 | 4.481244 | 5.642352 |
| gulf_inland | 2010-08-01 | f1 | 1.002587 | 4.599267 | 5.810751 |
| gulf_inland | 2010-11-01 | cnn | 1.790031 | 6.394223 | 8.279684 |
| gulf_inland | 2010-11-01 | gan | 1.992329 | 6.091092 | 7.849366 |
| gulf_inland | 2010-11-01 | grad | 1.308756 | 5.327369 | 6.760734 |
| gulf_inland | 2010-11-01 | cv | 1.345676 | 5.530626 | 7.034483 |
| gulf_inland | 2010-11-01 | pers | 1.216812 | 5.110699 | 6.418407 |
| gulf_inland | 2010-11-01 | f1 | 1.278087 | 5.400589 | 6.878348 |
| southwest_borderland | 2010-02-01 | cnn | 1.893995 | 8.377296 | 10.559911 |
| southwest_borderland | 2010-02-01 | gan | 3.805558 | 10.500160 | 13.037317 |
| southwest_borderland | 2010-02-01 | grad | 1.442916 | 7.044541 | 8.460207 |
| southwest_borderland | 2010-02-01 | cv | 1.456494 | 6.941262 | 8.412844 |
| southwest_borderland | 2010-02-01 | pers | 1.388010 | 6.912341 | 8.428401 |
| southwest_borderland | 2010-02-01 | f1 | 1.468913 | 6.947241 | 8.475964 |
| southwest_borderland | 2010-05-01 | cnn | 3.120291 | 10.690266 | 13.007913 |
| southwest_borderland | 2010-05-01 | gan | 5.626349 | 15.389355 | 19.379188 |
| southwest_borderland | 2010-05-01 | grad | 1.968554 | 8.627363 | 10.213842 |
| southwest_borderland | 2010-05-01 | cv | 1.879440 | 8.647922 | 10.226215 |
| southwest_borderland | 2010-05-01 | pers | 2.042360 | 8.774220 | 10.362001 |
| southwest_borderland | 2010-05-01 | f1 | 1.958105 | 8.715358 | 10.371723 |
| southwest_borderland | 2010-08-01 | cnn | 2.340958 | 9.673192 | 12.174605 |
| southwest_borderland | 2010-08-01 | gan | 4.423358 | 13.267138 | 17.017804 |
| southwest_borderland | 2010-08-01 | grad | 2.165791 | 8.469159 | 10.256556 |
| southwest_borderland | 2010-08-01 | cv | 2.058955 | 8.237814 | 10.140992 |
| southwest_borderland | 2010-08-01 | pers | 2.166681 | 8.179143 | 9.984592 |
| southwest_borderland | 2010-08-01 | f1 | 2.073456 | 8.217160 | 10.122539 |
| southwest_borderland | 2010-11-01 | cnn | 2.457925 | 11.046241 | 13.451865 |
| southwest_borderland | 2010-11-01 | gan | 4.781044 | 13.662762 | 16.559799 |
| southwest_borderland | 2010-11-01 | grad | 1.835828 | 9.561864 | 11.457008 |
| southwest_borderland | 2010-11-01 | cv | 2.235486 | 9.519888 | 11.402926 |
| southwest_borderland | 2010-11-01 | pers | 2.115115 | 9.825936 | 11.913622 |
| southwest_borderland | 2010-11-01 | f1 | 2.369547 | 9.768461 | 11.694528 |

## Important boundaries

- Topology is for the primary **160×160 top-left scalar-speed crop**, not the full 500×500 wind field.
- The four XCIII fields map to XCVIII field indices 0, 4, 8, 12; 24 archived distance values and feature counts must agree.
- Same trusted Phase6 corrected numeric decoder and GUDHI distance implementation, rechecked against frozen 2007 sample-0.
- Verification independently repeats numerical computation but **uses the same algorithm**, not a separate GUDHI implementation.
- No Phase5E spatial-locality measures, merge trees, independent hourly uncertainty, training or external locked tests here.
- Original pretrained PhIRE geographic/temporal training membership remains unknown.
