import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "topo_euler_results_mrhr.csv"
OUT_PATH = "chi_bar.png"

df = pd.read_csv(CSV_PATH)

# --- Robustly set the index to the run name ---
if "run" in df.columns:
    df = df.set_index("run")
else:
    first_col = df.columns[0]
    if str(first_col).lower().startswith("unnamed"):
        df = df.set_index(first_col)

df.index = df.index.astype(str).str.strip()

# Ensure numeric columns
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Aggregate duplicates (e.g., 5 samples per run) into a single mean row
df_mean = df.groupby(level=0).mean(numeric_only=True)

runs = ["Bicubic", "CNN_MRHR", "GAN_MRHR"]

# ✅ Robust bicubic: mean over all rows (independent of run ordering)
bic_chi_l1 = float(df["BIC_chi_L1"].mean())

cnn_chi_l1 = float(df_mean.loc["CNN_MRHR", "SR_chi_L1"])
gan_chi_l1 = float(df_mean.loc["GAN_MRHR", "SR_chi_L1"])

chi_l1 = [bic_chi_l1, cnn_chi_l1, gan_chi_l1]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(runs, chi_l1)
ax.set_title("Mean Euler-curve L1 distance vs GT (lower is better)")
ax.set_ylabel("Mean |Δχ| (L1)")
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=200)

print("Loaded:", CSV_PATH)
print("Raw index counts:", df.index.value_counts().to_dict())
print("Aggregated index labels:", list(df_mean.index))
print("Wrote:", OUT_PATH)
print("Values:", dict(zip(runs, chi_l1)))
