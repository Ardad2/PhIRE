import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fill these from your computed values
PSNR = {"GAN_MRHR": 32.48845799644413, "CNN_MRHR": 33.721410861175244}

df = pd.read_csv("topo_euler_results_mrhr.csv", index_col=0)

# We will plot SR_chi_L2 (or L1) + PSNR
runs = ["GAN_MRHR", "CNN_MRHR"]
chi = [df.loc["GAN_MRHR","SR_chi_L2"], df.loc["CNN_MRHR","SR_chi_L2"]]
psnr = [PSNR["GAN_MRHR"], PSNR["CNN_MRHR"]]

fig, ax = plt.subplots(figsize=(7,4))
ax.bar(runs, psnr)
ax.set_ylabel("PSNR (dB)")
ax.set_title("PSNR comparison (higher better)")
plt.tight_layout()
plt.savefig("psnr_bar.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7,4))
ax.bar(runs, chi)
ax.set_ylabel("Euler χ-curve distance (L2)")
ax.set_title("Topology distance vs GT (lower better)")
plt.tight_layout()
plt.savefig("chiL2_bar.png", dpi=200)
plt.close(fig)

print("Wrote psnr_bar.png and chiL2_bar.png")
