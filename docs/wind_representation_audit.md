# Wind representation audit (u/v vs speed)

This note records a repo-specific audit of whether PhIRE wind SR outputs are vector components or scalar speed.

## Key findings
- The network outputs the same channel count `C` as the input (`deconv_out` outputs `C`).
- TFRecords for wind are documented as `[ua, va]` channels.
- `test_paired()` de-normalizes LR/HR/SR and saves `dataIN.npy`, `dataGT.npy`, `dataSR.npy` as physical-unit arrays.
- The current topology pipeline explicitly converts arrays to **speed** (`sqrt(u^2+v^2)`) before VTI/TTK.

## Practical interpretation
The current PSNR/SSIM/topology study is valid as a **derived speed-magnitude evaluation** (consistent conversion for GT/SR), not as direct evaluation of the original vector-field target in channel space.
