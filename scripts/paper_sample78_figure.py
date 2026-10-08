#!/usr/bin/env python3
"""
Paper version of the Sample 78 qualitative figure (manuscript Fig. 6).

Same data and validation as scripts/poster_sample78_qualitative_final.py
(fixed 160x160 topology crop, shared speed scale, audited finite D0+D1
persistence diagrams, GT-diagram equality check across runs), re-laid-out for
a 7-inch two-column-paper text width:

  row 1: wind speed on the 160x160 topology crop for GT / CNN / control / final,
         one shared colorbar
  row 2: persistence diagrams (GT alone, then each model over the GT diagram),
         with tick values in m/s and identical limits in every panel
  right: persistence-survival curve (pairs with persistence >= t)

Fonts are set at their printed size (7-8 pt); include the PDF in LaTeX at
\\textwidth without further scaling.

Run from ~/PhIRE (same environment as the poster script):
    python3 scripts/paper_sample78_figure.py
Optional overrides: AUDIT=..., W22=..., PHIRE_ROOT=..., OUT_DIR=...
"""

from pathlib import Path
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

# =============================================================================
# CONFIG (paths identical to the poster script)
# =============================================================================
ROOT = Path(os.environ.get("PHIRE_ROOT", str(Path.home() / "PhIRE")))

CNN_DIR = ROOT / "data_out_fixed" / "wind_mrhr_cnn"
CONTROL_DIR = ROOT / "data_out" / "wind_finetune_candidateUV_expanded2688"
TOPO_DIR = ROOT / "data_out" / "wind_finetune_candidateF_grad_E2_low_expanded2688"

SAMPLE_ID = 78
X0, Y0, PATCH = 0, 0, 160          # fixed topology-evaluation crop

AUDIT = Path(os.environ.get("AUDIT", str(Path.home() / "phire_runtime_audit_20260809_221548")))
W22 = Path(os.environ.get("W22", str(AUDIT / "recompute_pd_w22")))
W22_SWEEP = W22 / "w22_full_sweep.csv"
CANONICAL_DIR = AUDIT / "recompute_pd"
sys.path.insert(0, str(CANONICAL_DIR))
import canonical_pd_pilot as canonical  # noqa: E402

CNN_RUN = "cnn"
CONTROL_RUN = "topology_finetuning/candidateUV_expanded2688_topology"
TOPO_RUN = "topology_finetuning/candidateF_grad_E2_low_expanded2688_topology"

OUT_DIR = Path(os.environ.get("OUT_DIR", str(ROOT / "figures" / "paper_sample78")))
STEM = "sample078_paper"

# Display-only threshold for the PD scatter panels and survival curve.
# All reported distances use the complete finite diagrams.
MIN_PERSISTENCE = 3.0  # m/s

# Values quoted in the manuscript text (Sec. 6.7); the script warns on mismatch.
EXPECTED_RMSE = {"CNN": 1.62, "Control": 1.34, "Topology": 1.58}
EXPECTED_PD = {  # (W22, W2inf, dB)
    "CNN": (30.6, 23.7, 4.15), "Control": (31.2, 24.2, 3.63), "Topology": (20.1, 16.1, 2.59),
}

WIND_CMAP = "viridis"
COL = {"GT": "#111111", "CNN": "#777777", "Control": "#C77C22", "Topology": "#009E73"}
TITLE = {"GT": "Ground truth", "CNN": "Pretrained CNN",
         "Control": "Recon.-only", "Topology": "Topology-inspired"}
LEGEND = {"GT": "Ground truth", "CNN": "Pretrained CNN",
          "Control": "Reconstruction-only", "Topology": "Topology-inspired"}
INK2, GRID = "#52514e", "#e6e5e0"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Nimbus Roman", "Nimbus Roman No9 L", "TeX Gyre Termes",
                   "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# =============================================================================
# IO / validation (from the poster script)
# =============================================================================
def require_file(p: Path):
    if not p.is_file():
        raise FileNotFoundError(str(p))


def read_csv(p: Path):
    require_file(p)
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def repo_path(v):
    p = Path(v)
    return p if p.is_absolute() else ROOT / p


def load_model_arrays(d: Path):
    for n in ("idx.npy", "dataGT.npy", "dataSR.npy"):
        require_file(d / n)
    return {"idx": np.load(d / "idx.npy"),
            "gt": np.load(d / "dataGT.npy", mmap_mode="r"),
            "sr": np.load(d / "dataSR.npy", mmap_mode="r")}


def row_for_sample(idx, sid, label):
    hits = np.flatnonzero(np.asarray(idx).astype(int) == int(sid))
    if len(hits) != 1:
        raise RuntimeError(f"{label}: expected one row for sample {sid}, got {len(hits)}")
    return int(hits[0])


def speed(uv):
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 3 or uv.shape[-1] != 2:
        raise RuntimeError(f"Expected [H,W,2], got {uv.shape}")
    return np.hypot(uv[..., 0], uv[..., 1])


def crop(a):
    return np.asarray(a[Y0:Y0 + PATCH, X0:X0 + PATCH])


def build_sweep_index():
    out = {}
    for r in read_csv(W22_SWEEP):
        k = (r["run"], int(r["sample"]))
        if k in out:
            raise RuntimeError(f"Duplicate W22 row: {k}")
        out[k] = r
    return out


def read_diagrams(sweep, run, sample):
    row = sweep[(run, sample)]
    gp, sp = repo_path(row["gt_path"]), repo_path(row["sr_path"])
    require_file(gp); require_file(sp)
    gt_pd, _ = canonical.read_pd(str(gp))
    sr_pd, _ = canonical.read_pd(str(sp))
    return gt_pd, sr_pd, row


def positive_diagram(D, label):
    D = np.asarray(D, dtype=np.float64)
    if D.size == 0:
        return np.empty((0, 2))
    if D.ndim != 2 or D.shape[1] != 2:
        raise RuntimeError(f"{label}: expected [N,2], got {D.shape}")
    p = D[:, 1] - D[:, 0]
    if np.any(p < 0):
        raise RuntimeError(f"{label}: negative persistence")
    D = D[p > 0]
    return D[np.lexsort((D[:, 1], D[:, 0]))] if len(D) else D


def clean_pd(pd, label):
    return {d: positive_diagram(pd[d], f"{label} D{d}") for d in (0, 1)}


def assert_same_gt(*named):
    ref_name, ref = named[0]
    ref = clean_pd(ref, ref_name)
    for name, pd in named[1:]:
        o = clean_pd(pd, name)
        for d in (0, 1):
            if ref[d].shape != o[d].shape or not np.array_equal(ref[d], o[d]):
                raise RuntimeError(f"GT PD mismatch {ref_name} vs {name}, D{d}")
    return ref


def combine(pd):
    parts = [pd[d] for d in (0, 1) if len(pd[d])]
    return np.vstack(parts) if parts else np.empty((0, 2))


def pers(D):
    return D[:, 1] - D[:, 0] if len(D) else np.empty(0)


def metric(row, key):
    for c in {"db": ("bottleneck_all", "db_all", "dB_all"),
              "w2inf": ("w2inf_all", "w2_inf_all"), "w22": ("w22_all",)}[key]:
        if row.get(c) not in ("", None):
            return float(row[c])
    return None


# =============================================================================
# Figure
# =============================================================================
def make_figure(fields, rmse, diagrams, dists):
    vmax = max(float(np.max(f)) for f in fields.values())
    shown = {k: D[pers(D) >= MIN_PERSISTENCE] for k, D in diagrams.items()}
    allpts = np.vstack([v for v in shown.values() if len(v)])
    lo = np.floor(allpts.min() / 5.0) * 5.0
    hi = np.ceil(allpts.max() / 5.0) * 5.0

    fig = plt.figure(figsize=(7.0, 3.35))
    gs = GridSpec(2, 7, figure=fig, width_ratios=[1, 1, 1, 1, 0.06, 0.40, 1.45],
                  height_ratios=[1, 1], left=0.055, right=0.985, top=0.93, bottom=0.17,
                  wspace=0.2, hspace=0.42)
    keys = ["GT", "CNN", "Control", "Topology"]
    letters = "abcd"

    for j, k in enumerate(keys):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(fields[k], cmap=WIND_CMAP, vmin=0, vmax=vmax, origin="upper",
                       interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"({letters[j]}) {TITLE[k]}", fontsize=7, pad=3,
                     fontweight="bold" if k == "Topology" else "normal")
        ax.text(0.5, -0.05, "reference" if k == "GT" else f"RMSE {rmse[k]:.2f} m/s",
                transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color=INK2,
                style="italic" if k == "GT" else "normal")

    cax = fig.add_subplot(gs[0, 4])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6, length=2, width=0.5)
    cb.set_label("Wind speed (m/s)", fontsize=6.5, labelpad=2)

    ticks = np.arange(lo, hi + 0.1, 5.0 if hi - lo <= 25 else 10.0)
    for j, k in enumerate(keys):
        ax = fig.add_subplot(gs[1, j])
        ax.plot([lo, hi], [lo, hi], color="#9a9a9a", lw=0.6, ls=(0, (3, 2)), zorder=1)
        g = shown["GT"]
        ax.scatter(g[:, 0], g[:, 1], s=7, facecolors="none", edgecolors=COL["GT"],
                   linewidths=0.45, zorder=2, label="GT")
        if k != "GT":
            p = shown[k]
            ax.scatter(p[:, 0], p[:, 1], s=7, c=COL[k], edgecolors="white", linewidths=0.25,
                       zorder=3, label=LEGEND[k])
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        if j:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Death (m/s)", labelpad=1)
        ax.set_xlabel("Birth (m/s)", labelpad=1)
        ax.grid(True, color=GRID, lw=0.4, zorder=0)
        ax.set_axisbelow(True)
        if k != "GT":
            w22, w2i, db = dists[k]
            c = dists["CNN"]
            ax.text(0.5, -0.33, f"{w22:.1f}  \u00b7  {w2i:.1f}  \u00b7  {db:.2f}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=6.3, color="#0b0b0b",
                    fontweight="bold" if k == "Topology" else "normal")
            if k != "CNN":
                ch = [100.0 * (v - r) / r for v, r in zip((w22, w2i, db), c)]
                ax.text(0.5, -0.46, "  \u00b7  ".join(f"{x:+.0f}%" for x in ch).replace("-", "\u2212"),
                        transform=ax.transAxes, ha="center", va="top", fontsize=6.0, color=INK2,
                        fontweight="bold" if k == "Topology" else "normal")
            else:
                ax.text(0.5, -0.46, "change vs. CNN \u2192", transform=ax.transAxes, ha="center",
                        va="top", fontsize=6.0, color=INK2, style="italic")
        else:
            ax.text(0.5, -0.33, r"$W_{2,2}$  $\cdot$  $W_{2,\infty}$  $\cdot$  $d_B$:", transform=ax.transAxes,
                    ha="center", va="top", fontsize=6.3, color=INK2)
        ax.set_title(f"({'efgh'[j]}) " + ("GT diagram" if k == "GT" else "vs. GT diagram"),
                     fontsize=7, pad=2)

    ax = fig.add_subplot(gs[1, 6])
    allp = np.concatenate([pers(D) for D in diagrams.values()])
    t = np.linspace(MIN_PERSISTENCE, float(allp.max()), 300)
    style = {"GT": ("-", 1.3), "CNN": ((0, (4, 2)), 1.1), "Control": ((0, (1, 1.2)), 1.3),
             "Topology": ("-", 1.3)}
    for k in keys:
        p = pers(diagrams[k])
        ax.step(t, [np.count_nonzero(p >= x) for x in t], where="post", color=COL[k],
                ls=style[k][0], lw=style[k][1], label=LEGEND[k])
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.8)
    ax.set_xlim(MIN_PERSISTENCE, float(allp.max()))
    ax.set_xlabel("Persistence threshold $t$ (m/s)", labelpad=1)
    ax.set_ylabel("Number of pairs", labelpad=1)
    ax.set_title("(i) Persistence survival", fontsize=7, pad=2)
    ax.grid(True, which="both", color=GRID, lw=0.4)
    handles, labels = ax.get_legend_handles_labels()

    # Shared legend in the free top-right cell (for both the PD panels and the survival curve).
    lax = fig.add_subplot(gs[0, 6])
    lax.set_axis_off()
    lax.legend(handles, labels, loc="center left", frameon=False, handlelength=2.4,
               borderaxespad=0.0, bbox_to_anchor=(0.0, 0.62), fontsize=6.8)
    lax.text(0.0, 0.08, "In (e)\u2013(h), open circles are\nground-truth pairs and filled\nmarkers are model pairs.",
             transform=lax.transAxes, ha="left", va="bottom", fontsize=6.0, color=INK2, linespacing=1.3)
    return fig


def main():
    cnn, ctl, top = (load_model_arrays(d) for d in (CNN_DIR, CONTROL_DIR, TOPO_DIR))
    r = {"CNN": row_for_sample(cnn["idx"], SAMPLE_ID, "CNN"),
         "Control": row_for_sample(ctl["idx"], SAMPLE_ID, "Control"),
         "Topology": row_for_sample(top["idx"], SAMPLE_ID, "Topology")}
    gt_uv = np.asarray(cnn["gt"][r["CNN"]])
    for name, m in (("Control", ctl), ("Topology", top)):
        d = float(np.max(np.abs(gt_uv - np.asarray(m["gt"][r[name]]))))
        if d > 1e-6:
            raise RuntimeError(f"GT vector arrays not aligned (CNN vs {name}: {d:.3e})")

    fields = {"GT": crop(speed(gt_uv)),
              "CNN": crop(speed(cnn["sr"][r["CNN"]])),
              "Control": crop(speed(ctl["sr"][r["Control"]])),
              "Topology": crop(speed(top["sr"][r["Topology"]]))}
    rmse = {k: float(np.sqrt(np.mean((fields[k] - fields["GT"]) ** 2)))
            for k in ("CNN", "Control", "Topology")}

    sweep = build_sweep_index()
    g1, p_cnn, row_cnn = read_diagrams(sweep, CNN_RUN, SAMPLE_ID)
    g2, p_ctl, row_ctl = read_diagrams(sweep, CONTROL_RUN, SAMPLE_ID)
    g3, p_top, row_top = read_diagrams(sweep, TOPO_RUN, SAMPLE_ID)
    gt_pd = assert_same_gt(("CNN", g1), ("Control", g2), ("Topology", g3))
    diagrams = {"GT": combine(gt_pd), "CNN": combine(clean_pd(p_cnn, "CNN")),
                "Control": combine(clean_pd(p_ctl, "Control")),
                "Topology": combine(clean_pd(p_top, "Topology"))}
    dists = {}
    for k, row in (("CNN", row_cnn), ("Control", row_ctl), ("Topology", row_top)):
        v = (metric(row, "w22"), metric(row, "w2inf"), metric(row, "db"))
        if None in v:
            raise RuntimeError(f"Missing audited distance for {k}: {v}")
        dists[k] = v

    # Consistency with the numbers quoted in the manuscript.
    warn = []
    for k, e in EXPECTED_RMSE.items():
        if round(rmse[k], 2) != e:
            warn.append(f"RMSE {k}: figure {rmse[k]:.3f} vs text {e}")
    for k, e in EXPECTED_PD.items():
        got = (round(dists[k][0], 1), round(dists[k][1], 1), round(dists[k][2], 2))
        if got != e:
            warn.append(f"PD {k}: figure {got} vs text {e}")

    fig = make_figure(fields, rmse, diagrams, dists)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{STEM}.pdf", facecolor="white")
    fig.savefig(OUT_DIR / f"{STEM}.png", dpi=300, facecolor="white")
    print("RMSE on crop:", {k: round(v, 3) for k, v in rmse.items()})
    print("Audited PD (W22, W2inf, dB):", dists)
    print("Pair counts:", {k: len(v) for k, v in diagrams.items()})
    print("WARNINGS:\n  " + "\n  ".join(warn) if warn else "All quoted numbers match the manuscript.")
    print("Wrote", OUT_DIR / f"{STEM}.pdf")


if __name__ == "__main__":
    main()
