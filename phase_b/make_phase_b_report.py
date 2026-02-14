from pathlib import Path
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

FIGDIR = Path("figs_phase_b")
OUTPDF = FIGDIR / "phase_b_report.pdf"

styles = getSampleStyleSheet()
doc = SimpleDocTemplate(str(OUTPDF), pagesize=letter,
                        leftMargin=0.8*inch, rightMargin=0.8*inch,
                        topMargin=0.75*inch, bottomMargin=0.75*inch)

story = []
story.append(Paragraph("Phase B.1 — Persistence Diagrams (Patch-based Preliminary)", styles["Title"]))
story.append(Spacer(1, 10))

# Pick a field to report (prefer speed)
stats_files = sorted(FIGDIR.glob("phase_b_summary_stats_*.csv"))
if not stats_files:
    raise RuntimeError("No summary stats found in figs_phase_b. Run plot_phase_b_summary.py first.")

# Prefer speed if present
stats_path = None
for p in stats_files:
    if p.name.endswith("_speed.csv"):
        stats_path = p
        break
stats_path = stats_path or stats_files[0]

field = stats_path.stem.split("_")[-1]
story.append(Paragraph(f"Field: {field}", styles["Heading2"]))
story.append(Spacer(1, 6))

stats = pd.read_csv(stats_path)
cols = [c for c in ["label","n_samples","patches_total","psnr_mean","psnr_std","w1_pd0_mean","w1_pd0_std","w1_pd1_mean","w1_pd1_std","w2_pd0_count","w2_pd1_count"] if c in stats.columns]
tbl_data = [cols] + stats[cols].values.tolist()

tbl = Table(tbl_data, hAlign="LEFT")
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.black),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey]),
]))
story.append(tbl)
story.append(PageBreak())

# Add bar plots if present
for name in [f"bar_mean_psnr_{field}.png", f"bar_mean_w1_pd0_{field}.png", f"bar_mean_w1_pd1_{field}.png"]:
    p = FIGDIR / name
    if p.exists():
        story.append(Paragraph(p.name, styles["BodyText"]))
        im = RLImage(str(p))
        im.drawWidth = 6.7*inch
        im.drawHeight = im.drawHeight * (im.drawWidth / im.imageWidth)
        story.append(im)
        story.append(Spacer(1, 12))

doc.build(story)
print("[WROTE]", OUTPDF)
