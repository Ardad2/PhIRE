from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUTDIR = Path("figs_phase_b")
CSV_RESULTS = Path("phase_b_persistence_results.csv")
CSV_SUMMARY = OUTDIR / "phase_b_summary_stats_speed.csv"

PLOTS = [
    ("Mean PSNR (higher better)", OUTDIR / "bar_mean_psnr_speed.png"),
    ("Mean W1(PD0) (lower better)", OUTDIR / "bar_mean_w1_pd0_speed.png"),
    ("Mean W1(PD1) (lower better)", OUTDIR / "bar_mean_w1_pd1_speed.png"),
    ("Mean W2(PD0) audit subset", OUTDIR / "bar_mean_w2_pd0_speed.png"),
    ("Mean W2(PD1) audit subset", OUTDIR / "bar_mean_w2_pd1_speed.png"),
    ("Scatter: PSNR vs W1(PD0) GAN", OUTDIR / "scatter_psnr_w1_pd0_speed_GAN_MRHR.png"),
    ("Scatter: PSNR vs W1(PD0) CNN", OUTDIR / "scatter_psnr_w1_pd0_speed_CNN_MRHR.png"),
    ("Scatter: PSNR vs W1(PD1) GAN", OUTDIR / "scatter_psnr_w1_pd1_speed_GAN_MRHR.png"),
    ("Scatter: PSNR vs W1(PD1) CNN", OUTDIR / "scatter_psnr_w1_pd1_speed_CNN_MRHR.png"),
]

def _add_text_page(pp: PdfPages, title: str, lines: list[str]):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(0.06, 0.95, title, fontsize=18, fontweight="bold", va="top")
    y = 0.90
    for ln in lines:
        ax.text(0.06, y, ln, fontsize=11, va="top")
        y -= 0.03

    pp.savefig(fig)
    plt.close(fig)

def _add_table_page(pp: PdfPages, title: str, df: pd.DataFrame):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.90])
    ax.axis("off")

    ax.text(0.0, 1.02, title, fontsize=14, fontweight="bold", transform=ax.transAxes)

    show = df.copy()
    for col in show.columns:
        if col not in ["label"]:
            show[col] = show[col].apply(lambda x: f"{x:.4g}" if pd.notnull(x) else "")

    table = ax.table(
        cellText=show.values,
        colLabels=show.columns,
        loc="upper left",
        cellLoc="center",
        colLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)

    pp.savefig(fig)
    plt.close(fig)

def _add_image_page(pp: PdfPages, caption: str, img_path: Path):
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.90])
    ax.axis("off")

    if img_path.exists():
        img = plt.imread(str(img_path))
        ax.imshow(img)
        fig.suptitle(caption, fontsize=14, fontweight="bold", y=0.98)
    else:
        ax.text(0.1, 0.5, f"Missing image: {img_path}", fontsize=12)

    pp.savefig(fig)
    plt.close(fig)

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if not CSV_RESULTS.exists():
        raise FileNotFoundError(f"Missing {CSV_RESULTS}. Run phase_b_persistence_eval.py first.")
    if not CSV_SUMMARY.exists():
        raise FileNotFoundError(f"Missing {CSV_SUMMARY}. Run plot_phase_b_summary.py first.")

    results = pd.read_csv(CSV_RESULTS)
    summary = pd.read_csv(CSV_SUMMARY)

    # Pull run configuration from the first row (these are stored in your results CSV)
    r0 = results.iloc[0].to_dict()
    cfg_lines = [
        f"- Dataset: example_data/wind_MR-HR.tfrecord (5 records) -> 5 images per run",
        f"- Runs: {', '.join(sorted(results['run'].unique()))}",
        f"- Field(s): {r0.get('fields', 'speed')}",
        f"- Patch/stride: patch={r0.get('patch', '')}, stride={r0.get('stride', '')}",
        f"- PD pruning: min_pers={r0.get('min_pers', '')}, max_pts={r0.get('max_pts', '')}",
        f"- W2 audit: w2_audit_every={r0.get('w2_audit_every', '')} (~1/K patches)",
        f"- Rows: {len(results)} (patch-level), summarized per-sample in stats CSV",
    ]

    # Keep a compact table for professor-facing summary
    cols_wanted = [
        "label", "n_samples", "patches_total",
        "psnr_mean", "psnr_std",
        "w1_pd0_mean", "w1_pd0_std",
        "w1_pd1_mean", "w1_pd1_std",
        "w2_pd0_mean", "w2_pd0_std",
        "w2_pd1_mean", "w2_pd1_std",
        "w2_pd0_count", "w2_pd1_count",
    ]
    show_cols = [c for c in cols_wanted if c in summary.columns]
    summary_show = summary[show_cols].copy()

    pdf_path = OUTDIR / "phase_b_results_brief.pdf"
    with PdfPages(str(pdf_path)) as pp:
        _add_text_page(
            pp,
            "Phase B — Persistence Diagram Metrics (Wind MR→HR)",
            [
                "Goal: compare SR outputs (GAN vs CNN) against GT using PSNR + PD distances.",
                "Baseline: bicubic upsampling from MR input.",
                "",
                "Configuration:",
                *cfg_lines,
                "",
                "Notes:",
                "- PD computed on speed field (||u,v||) in [0,1] (pairwise normalized by GT min/max).",
                "- W1 computed for all patches; W2 computed on a deterministic audit subset to keep runtime reasonable.",
            ],
        )

        _add_table_page(pp, "Summary statistics (per-sample avg of patches)", summary_show)

        for caption, imgp in PLOTS:
            _add_image_page(pp, caption, imgp)

    print(f"[OK] Wrote: {pdf_path}")

if __name__ == "__main__":
    main()
