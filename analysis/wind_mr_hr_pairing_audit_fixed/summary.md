# Wind MR→HR pairing audit

TFRecord: `example_data_fixed/wind_MR-HR.tfrecord`

Candidate comparison set: all rows.

## Pairing result summary
- Samples audited: 168
- Paired MR rank-1 by RMSE: 168/168
- Paired MR rank-1 by MAE: 168/168
- Median paired-rank by RMSE: 1
- Mean paired RMSE: 0
- Median paired RMSE: 0
- Mean best-wrong minus paired RMSE gap: 1.15227
- Min best-wrong minus paired RMSE gap: 0.597061

## Orientation sanity check
- Representative samples checked: 3
- Cases where a non-identity transform beat identity: 0
- Identity was best on all representative orientation checks.

## Interpretation guide
- If the paired MR is usually rank-1 or near-rank-1 against wrong MR candidates, that strongly supports valid per-sample pairing.
- If the best-wrong minus paired gap is consistently positive, the paired MR is meaningfully closer to its own downsampled HR than to wrong samples.
- If identity beats transpose/flip/channel-swap checks, that argues against an obvious layout/orientation bug.
