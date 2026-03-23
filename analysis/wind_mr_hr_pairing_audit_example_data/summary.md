# Wind MR→HR pairing audit

TFRecord: `example_data/wind_MR-HR.tfrecord`

Candidate comparison set: all rows.

## Pairing result summary
- Samples audited: 168
- Paired MR rank-1 by RMSE: 16/168
- Paired MR rank-1 by MAE: 11/168
- Median paired-rank by RMSE: 21
- Mean paired RMSE: 7.24387
- Median paired RMSE: 7.42831
- Mean best-wrong minus paired RMSE gap: -1.45293
- Min best-wrong minus paired RMSE gap: -4.56722

## Orientation sanity check
- Representative samples checked: 3
- Cases where a non-identity transform beat identity: 0
- Identity was best on all representative orientation checks.

## Interpretation guide
- If the paired MR is usually rank-1 or near-rank-1 against wrong MR candidates, that strongly supports valid per-sample pairing.
- If the best-wrong minus paired gap is consistently positive, the paired MR is meaningfully closer to its own downsampled HR than to wrong samples.
- If identity beats transpose/flip/channel-swap checks, that argues against an obvious layout/orientation bug.
