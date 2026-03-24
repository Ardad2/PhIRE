from pathlib import Path
import numpy as np
import utils
from build_example_data_clone_exact import extract_native_hr_stack

OUTDIR = Path("example_data_fixed")
OUTDIR.mkdir(parents=True, exist_ok=True)

print("Extracting native HR stack...")
hr_stack = extract_native_hr_stack()
print("hr_stack shape:", hr_stack.shape)

out_mr_hr = OUTDIR / "wind_MR-HR.tfrecord"
print("Writing fixed MR->HR TFRecord...")
utils.generate_TFRecords(str(out_mr_hr), hr_stack.astype(np.float64), mode="train", K=5)

print("Done:", out_mr_hr.resolve())
