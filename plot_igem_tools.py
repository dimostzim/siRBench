import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("igem_tools_metrics.csv")

df = df.dropna(subset=["tool", "regime", "pcc"])
comparison_df = df[~df["tool"].isin(["Agentomics", "TabPFN"])]

pivot = comparison_df.pivot(index="tool", columns="regime", values="pcc").reset_index()
pivot = pivot.dropna(subset=["standard", "best_10fold"])

pivot = pivot.sort_values("standard", ascending=False)

names = pivot["tool"].str.replace("siRNADiscovery", "siRNA\nDiscovery")
standard = pivot["standard"] * 100
best_10fold = pivot["best_10fold"] * 100

fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.2
gap = 0.03
group_spacing = 4 * gap
x = np.arange(len(names)) * (2 * bar_width + gap + group_spacing)

ax.bar(x - bar_width / 2 - gap / 2, standard, bar_width,
       label="Standard Split", color="#4682B4", edgecolor="black", linewidth=2)
ax.bar(x + bar_width / 2 + gap / 2, best_10fold, bar_width,
       label="Best 10-fold Split", color="#a6c8e8", edgecolor="black", linewidth=2)

ax.set_ylabel("PCC (%)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(names, fontweight="bold", fontsize=12)
ax.legend()

ax.set_yticks(np.arange(0, 81, 20))
for label in ax.get_yticklabels():
    label.set_weight("bold")
    label.set_fontsize(12)

ax.set_ylim(0, 80)

for i, (s, b) in enumerate(zip(standard, best_10fold)):
    ax.text(x[i] - bar_width / 2 - gap / 2, s + 1, f"{s:.2f}", ha="center", va="bottom", fontsize=10)
    ax.text(x[i] + bar_width / 2 + gap / 2, b + 1, f"{b:.2f}", ha="center", va="bottom", fontsize=10)

ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("igem_tools_comparison.png", dpi=300, bbox_inches="tight")
