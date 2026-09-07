from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.home() / "PhIRE"

parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=int, default=75)
parser.add_argument(
    "--data-dir",
    type=Path,
    default=ROOT / "data_out/wind_finetune_candidateC_expanded2688"
)
parser.add_argument(
    "--out",
    type=Path,
    default=ROOT / "poster_wind_sr_context"
)
args = parser.parse_args()

idx = np.load(args.data_dir / "idx.npy")
assert np.array_equal(idx, np.arange(168))

data_in = np.load(args.data_dir / "dataIN.npy", mmap_mode="r")
data_sr = np.load(args.data_dir / "dataSR.npy", mmap_mode="r")
data_gt = np.load(args.data_dir / "dataGT.npy", mmap_mode="r")

def speed(uv):
    return np.sqrt(uv[..., 0]**2 + uv[..., 1]**2)

i = args.sample

mr = speed(data_in[i])
sr = speed(data_sr[i])
gt = speed(data_gt[i])

# Shared scale for the HR/SR comparison.
# Include the MR field as well so all three use the same visual scale.
vmin = min(float(mr.min()), float(sr.min()), float(gt.min()))
vmax = max(float(mr.max()), float(sr.max()), float(gt.max()))

fig, axes = plt.subplots(
    1, 3,
    figsize=(10.5, 3.0),
    constrained_layout=True
)

titles = [
    "MR input  •  100×100",
    "SR prediction  •  500×500",
    "HR ground truth  •  500×500"
]

fields = [mr, sr, gt]

for ax, field, title in zip(axes, fields, titles):
    im = ax.imshow(
        field,
        cmap="cividis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest"
    )

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=7
    )

    ax.set_xticks([])
    ax.set_yticks([])

cbar = fig.colorbar(
    im,
    ax=axes,
    fraction=0.025,
    pad=0.02
)

cbar.set_label(
    "Wind-speed magnitude",
    fontsize=12
)

args.out.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(
    str(args.out) + ".pdf",
    bbox_inches="tight"
)

plt.savefig(
    str(args.out) + ".svg",
    bbox_inches="tight"
)

plt.savefig(
    str(args.out) + ".png",
    dpi=350,
    bbox_inches="tight"
)

print("Saved:")
print(str(args.out) + ".pdf")
print(str(args.out) + ".svg")
print(str(args.out) + ".png")