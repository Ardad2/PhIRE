#!/usr/bin/env python3
"""Figures for the loss study, generated from the same verified means as
make_tables.py:

  figures/loss_heatmap.pdf    percent change in mean distance vs. the
                              reconstruction-only control, every configuration
                              x {d_B, W_2inf, W_22, MT}
  figures/pd_mt_tradeoff.pdf  three panels: each PD distance vs. MT distance,
                              with the Pareto front of each pair dashed

Run:  python3 tools/make_figures.py [--joined corrected_pd_mt_joined.csv] [--out DIR]
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_tables import MEAN, pareto, IDS, pct  # noqa: E402

FIG = os.path.join(HERE, "..", "figures")
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.edgecolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "pdf.fonttype": 42,
})
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

# compact component labels: S speed, G gradient, L level-set, M local-max, FP fixed-pair
SHORT = {
    "cnn": "Pretrained CNN", "gan": "Pretrained GAN", "uv": "Recon.-only control",
    "speed_only": "S", "levelset_only": "L", "speed_levelset": "S + L", "grad_only": "G",
    "speed_grad": "S + G", "grad_levelset": "G + L", "candidate_b": "S + G + L",
    "uv_crit": "M", "f3_grad_crit": "G + M", "candidate_c": "S + G + L + M",
    "uv_e2": "FP", "f1_grad_e2": "G + FP (final)", "f2_grad_levelset_e2": "G + L + FP",
    "b_e2": "S + G + L + FP", "c_e2": "S + G + L + M + FP",
}
GROUPS = [
    ("Baselines", ["cnn", "gan", "uv"]),
    ("Scalar-field terms", ["speed_only", "levelset_only", "speed_levelset", "grad_only",
                            "speed_grad", "grad_levelset", "candidate_b"]),
    ("Local-maximum term", ["uv_crit", "f3_grad_crit", "candidate_c"]),
    ("Fixed-pair supervision", ["uv_e2", "f1_grad_e2", "f2_grad_levelset_e2", "b_e2", "c_e2"]),
]
COLS = [r"$d_B$", r"$W_{2,\infty}$", r"$W_{2,2}$", "MT"]


def heatmap():
    cmap = LinearSegmentedColormap.from_list(
        "div", ["#b8302f", "#e34948", "#f0efec", "#2a78d6", "#104281"], N=256)
    rows, labels, gaps = [], [], []
    for gname, ids in GROUPS:
        gaps.append((len(rows), gname))
        for k in ids:
            rows.append([pct(MEAN["uv"][m], MEAN[k][m]) for m in range(4)])
            labels.append(SHORT[k])
    V = np.array(rows)
    # insert a blank band before each group for its header
    n = len(rows) + len(GROUPS)
    fig, ax = plt.subplots(figsize=(3.45, 4.15))
    y = 0
    yt, ylab = [], []
    for gi, (start, gname) in enumerate(gaps):
        end = gaps[gi + 1][0] if gi + 1 < len(gaps) else len(rows)
        ax.text(-0.02, y + 0.62, gname, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=6.6, fontweight="bold", color=INK)
        y += 1
        for r in range(start, end):
            for c in range(4):
                v = V[r, c]
                ax.add_patch(plt.Rectangle((c + 0.04, y + 0.04), 0.92, 0.92,
                                           color=cmap((np.clip(v, -40, 40) + 40) / 80), lw=0))
                txt = "0" if (labels[r].startswith("Recon.")) else f"{v:+.1f}"
                dark = abs(v) > 22
                ax.text(c + 0.5, y + 0.5, txt, ha="center", va="center", fontsize=6.0,
                        color="white" if dark else INK)
            yt.append(y + 0.5)
            ylab.append(labels[r])
            y += 1
    ax.set_xlim(0, 4)
    ax.set_ylim(y, 0)
    ax.set_yticks(yt)
    ax.set_yticklabels(ylab, fontsize=6.4)
    for t in ax.get_yticklabels():
        if "final" in t.get_text():
            t.set_fontweight("bold")
    ax.set_xticks([0.5, 1.5, 2.5, 3.5])
    ax.set_xticklabels(COLS, fontsize=7)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(2.0, y + 0.55, "% lower mean distance than the reconstruction-only control\n"
            "(blue: lower/better; red: higher/worse; color clipped at $\\pm$40%)",
            ha="center", va="top", fontsize=5.8, color=INK2)
    fig.subplots_adjust(left=0.43, right=0.985, top=0.95, bottom=0.07)
    fig.savefig(os.path.join(FIG, "loss_heatmap.pdf"))
    fig.savefig(os.path.join(FIG, "loss_heatmap.png"), dpi=300)


FAMILY = [
    (["cnn", "uv"], "#7a7a7a", "s", "Pretrained CNN / recon.-only control"),
    (["speed_only", "levelset_only", "speed_levelset", "grad_only", "speed_grad",
      "grad_levelset", "candidate_b"], "#2a78d6", "o", "Speed / gradient / level-set terms"),
    (["uv_crit", "f3_grad_crit", "candidate_c"], "#eb6834", "^", "Local-maximum term"),
    (["uv_e2", "f1_grad_e2", "f2_grad_levelset_e2", "b_e2", "c_e2"], "#1baf7a", "D",
     "Fixed-pair supervision"),
]
LABELS = {"cnn": ("CNN", 4, 2), "uv": ("Control", 4, -7), "grad_only": ("G", 4, -6),
          "uv_crit": ("M", 4, 1), "uv_e2": ("FP", 5, -5), "f1_grad_e2": ("G+FP", -24, 3)}


def tradeoff():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), sharey=True)
    xl = [(2.0, 3.45), (13.6, 21.8), (16.9, 28.4)]
    for ax, X, lim in zip(axes, range(3), xl):
        front = pareto(IDS, [X, 3])
        fr = sorted((k for k in front if k != "gan"), key=lambda k: MEAN[k][X])
        xs, ys = [], []
        for k in fr:
            if xs:
                xs.append(MEAN[k][X]); ys.append(ys[-1])
            xs.append(MEAN[k][X]); ys.append(MEAN[k][3])
        ax.plot(xs, ys, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=1)
        for ids, col, mk, lab in FAMILY:
            for k in ids:
                on = k in front
                ax.scatter(MEAN[k][X], MEAN[k][3], s=34 if k == "f1_grad_e2" else 17, c=col, marker=mk,
                           edgecolors=INK if on else SURF, linewidths=0.8 if on else 0.4, zorder=4 if k in ("cnn", "uv") else 3,
                           label=lab if (X == 0 and k == ids[0]) else None)
        for k, (t, dx, dy) in LABELS.items():
            ax.annotate(t, (MEAN[k][X], MEAN[k][3]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=6, color=INK if k in front else INK2,
                        fontweight="bold" if k == "f1_grad_e2" else "normal")
        ax.set_xlim(*lim)
        ax.set_ylim(5.5, 6.4)
        ax.grid(True, color=GRID, lw=0.4, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel(f"Mean {COLS[X]} (lower is better)")
        g = MEAN["gan"]
        ax.text(0.98, 0.97, f"GAN off-axis: ({g[X]:.2f}, {g[3]:.2f})", transform=ax.transAxes,
                ha="right", va="top", fontsize=5.6, color=INK2)
    axes[0].set_ylabel("Mean MT distance (lower is better)")
    fig.legend(loc="upper center", ncol=4, frameon=False, fontsize=6.2, handletextpad=0.2,
               columnspacing=1.2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=0.8)
    fig.savefig(os.path.join(FIG, "pd_mt_tradeoff.pdf"))
    fig.savefig(os.path.join(FIG, "pd_mt_tradeoff.png"), dpi=300)
    return {c: sorted(pareto(IDS, [i, 3])) for i, c in enumerate(["dB", "W2inf", "W22"])}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", help="corrected_pd_mt_joined.csv (per-field; preferred)")
    ap.add_argument("--out", help="output directory for the figures")
    a = ap.parse_args()
    if a.out:
        FIG = a.out
        os.makedirs(FIG, exist_ok=True)
    if a.joined:
        import make_tables
        make_tables.load_joined(a.joined)
    heatmap()
    print("fronts:", tradeoff())
