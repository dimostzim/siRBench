import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("igem_tools_metrics.csv")

standard_df = df[(df["regime"] == "standard") & df["tool"].notna()]
standard_df = standard_df.sort_values("pcc", ascending=False)

names = standard_df["tool"].str.replace("siRNADiscovery", "siRNA\nDiscovery")
values = standard_df["pcc"] * 100

colors = ["red" if tool in {"TabPFN", "Agentomics"} else "#4682B4" for tool in standard_df["tool"]]

fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.2
gap = 0.03
group_spacing = 2 * gap
x = np.arange(len(names)) * (2 * bar_width + gap + group_spacing)

ax.bar(x, values, bar_width, color=colors, edgecolor="black", linewidth=2)

ax.set_ylabel("PCC (%)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(names, fontweight="bold", fontsize=10)

ax.set_yticks(np.arange(0, 81, 20))
for label in ax.get_yticklabels():
    label.set_weight("bold")
    label.set_fontsize(12)

ax.set_ylim(0, 80)

for i, v in enumerate(values):
    ax.text(x[i], v + 1, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("igem_tools_standard_only.png", dpi=300, bbox_inches="tight")
