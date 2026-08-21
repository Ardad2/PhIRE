#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SAMPLE = 107
PATCH = 160

OUT = Path("poster_figures")
OUT.mkdir(exist_ok=True)

PATHS = {
    "GT": "data_out_fixed/wind_mrhr_cnn/dataGT.npy",
    "CNN": "data_out_fixed/wind_mrhr_cnn/dataSR.npy",
    "Vector-only":
        "data_out/wind_finetune_candidateUV_expanded2688/dataSR.npy",
    "Topology-inspired":
        "data_out/wind_finetune_candidateC_expanded2688/dataSR.npy",
}

COLORS = {
    "CNN": "#0072B2",
    "Vector-only": "#009E73",
    "Topology-inspired": "#CC79A7",
}

def speed(a):
    return np.hypot(a[..., 0], a[..., 1])

fields = {}

for name, path in PATHS.items():
    a = np.load(path, mmap_mode="r")

    field = speed(
        a[SAMPLE, :PATCH, :PATCH, :]
    )

    fields[name] = np.asarray(field)

gt = fields["GT"]

errors = {
    "CNN": np.abs(fields["CNN"] - gt),
    "Vector-only":
        np.abs(fields["Vector-only"] - gt),
    "Topology-inspired":
        np.abs(fields["Topology-inspired"] - gt),
}

topo_vector_difference = np.abs(
    fields["Topology-inspired"]
    - fields["Vector-only"]
)

field_min = min(
    float(x.min())
    for x in fields.values()
)

field_max = max(
    float(x.max())
    for x in fields.values()
)

error_max = max(
    float(x.max())
    for x in [
        *errors.values(),
        topo_vector_difference,
    ]
)

fig, axes = plt.subplots(
    2, 4,
    figsize=(12.2, 6.3),
    constrained_layout=True,
)

# --------------------------
# Wind-speed fields
# --------------------------

names = [
    "GT",
    "CNN",
    "Vector-only",
    "Topology-inspired",
]

ims = []

for j, name in enumerate(names):
    im = axes[0, j].imshow(
        fields[name],
        origin="lower",
        cmap="viridis",
        vmin=field_min,
        vmax=field_max,
    )

    ims.append(im)

    axes[0, j].set_title(
        name,
        fontsize=13,
        weight="bold",
        color=COLORS.get(name, "black"),
    )

    axes[0, j].set_xticks([])
    axes[0, j].set_yticks([])

cb1 = fig.colorbar(
    ims[-1],
    ax=axes[0, :],
    fraction=0.018,
    pad=0.015,
)

cb1.set_label(
    r"Wind speed (m s$^{-1}$)"
)

# --------------------------
# Error / difference maps
# --------------------------

bottom = [
    ("CNN", errors["CNN"], r"$|\mathrm{CNN}-\mathrm{GT}|$"),
    (
        "Vector-only",
        errors["Vector-only"],
        r"$|\mathrm{Vector}-\mathrm{GT}|$",
    ),
    (
        "Topology-inspired",
        errors["Topology-inspired"],
        r"$|\mathrm{Topology}-\mathrm{GT}|$",
    ),
    (
        "Difference",
        topo_vector_difference,
        r"$|\mathrm{Topology}-\mathrm{Vector}|$",
    ),
]

ims_err = []

for j, (name, arr, title) in enumerate(bottom):
    im = axes[1, j].imshow(
        arr,
        origin="lower",
        cmap="magma",
        vmin=0,
        vmax=error_max,
    )

    ims_err.append(im)

    axes[1, j].set_title(
        title,
        fontsize=12,
        color=COLORS.get(name, "black"),
    )

    axes[1, j].set_xticks([])
    axes[1, j].set_yticks([])

cb2 = fig.colorbar(
    ims_err[-1],
    ax=axes[1, :],
    fraction=0.018,
    pad=0.015,
)

cb2.set_label(
    r"Absolute difference (m s$^{-1}$)"
)

fig.suptitle(
    "Representative Wind-Speed Reconstruction — Sample 107",
    fontsize=16,
    weight="bold",
)

for suffix in ["pdf", "svg"]:
    fig.savefig(
        OUT / f"sample107_fields_errors.{suffix}",
        bbox_inches="tight",
    )

fig.savefig(
    OUT / "sample107_fields_errors.png",
    dpi=400,
    bbox_inches="tight",
)

print("Field range:", field_min, field_max)
print("Error range:", 0, error_max)
print("Wrote sample107_fields_errors.[pdf|svg|png]")
