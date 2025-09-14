import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('igem_tools.csv')

# Remove any empty rows
df = df.dropna()

# Sort by standard_split in descending order
df = df.sort_values('standard_split', ascending=False)

# Set up the data
tools = df['tool']
standard_split = df['standard_split'] * 100  # Convert to percentage
best_10fold_split = df['best_10-fold_split'] * 100  # Convert to percentage

# Set up the plot (more rectangular - narrower width)
fig, ax = plt.subplots(figsize=(10, 6))

# Set the width of the bars and positions (narrower bars with more separation)
bar_width = 0.2
gap = 0.03  # Gap between grouped bars (reduced)
group_spacing = 4 * gap  # Distance between groups is 4x the distance between grouped bars
x = np.arange(len(tools)) * (2 * bar_width + gap + group_spacing)

# Create the bars with more distinct blue tones and bolder black borders
bars1 = ax.bar(x - bar_width/2 - gap/2, standard_split, bar_width,
               label='Standard Split', color='#4682B4', edgecolor='black', linewidth=2)
bars2 = ax.bar(x + bar_width/2 + gap/2, best_10fold_split, bar_width,
               label='Best 10-fold Split', color='#a6c8e8', edgecolor='black', linewidth=2)

# Customize the plot
ax.set_ylabel('PCC (%)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tools, fontweight='bold', fontsize=12)  # Bold x-axis ticks with larger font
ax.legend()

# Make y-axis tick labels bold and larger, with ticks every 20
ax.set_yticks(np.arange(0, 81, 20))  # Y-axis ticks by 20
for label in ax.get_yticklabels():
    label.set_weight('bold')
    label.set_fontsize(12)

# Set y-axis limits from 0 to 80
ax.set_ylim(0, 80)

# Add numbers above each bar
for i, (val1, val2) in enumerate(zip(standard_split, best_10fold_split)):
    ax.text(x[i] - bar_width/2 - gap/2, val1 + 1, f'{val1:.2f}', ha='center', va='bottom', fontsize=10)
    ax.text(x[i] + bar_width/2 + gap/2, val2 + 1, f'{val2:.2f}', ha='center', va='bottom', fontsize=10)

# Show grid for better readability
ax.grid(True, alpha=0.3, axis='y')

# Save the plot (only PNG, no show)
plt.tight_layout()
plt.savefig('igem_tools_comparison.png', dpi=300, bbox_inches='tight')