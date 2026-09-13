import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Data extraction from the image
components = [
    "Interaction Stacking", "Early Fusion", "Confidence-Weighted", 
    "Text-Only", "Audio-Only", "Mixture of Experts", 
    "Model-Based Stacking", "Cross-Attention Fusion", 
    "Bilinear Fusion", "Late Fusion", "Adaptive Weighted Fusion"
]
perf_drop = [0.112, 0.109, 0.061, 0.053, 0.035, 0.031, 0.024, 0.021, 0.021, 0.020, 0.007]
p_values = [0.007, 0.025, 0.075, 0.005, 0.036, 0.217, 0.070, 0.134, 0.134, 0.132, 0.668]

is_significant = [p < 0.05 for p in p_values]
colors = ['#5CDB95' if sig else '#EE6C4D' for sig in is_significant]

# Set up the figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 7.5), gridspec_kw={'height_ratios': [1, 1]})

# --- Top Plot: Performance Drop ---
y_pos = np.arange(len(components))
bars1 = ax1.barh(y_pos, perf_drop, color=colors, edgecolor='black', linewidth=0.5)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(components, fontsize=9)
ax1.invert_yaxis()

# --- KEY CHANGE 1: Remove x-ticks and x-label from top plot ---
ax1.set_xticks([]) 
ax1.set_xlabel('')

ax1.set_title('Ablation vs Reference (Classification)\nReference: Voting Ensemble = 0.864', fontsize=10, fontweight='bold')

for bar in bars1:
    width = bar.get_width()
    ax1.text(width + 0.002, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center', fontsize=8, fontweight='bold')

# --- Bottom Plot: Statistical Significance ---
bars2 = ax2.barh(y_pos, [-np.log10(p) for p in p_values], color=colors, edgecolor='black', linewidth=0.5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(components, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel('-log10(p-value) (paired t-test vs reference)', fontsize=10)

# --- KEY CHANGE 2: Added pad to the title to push it slightly down ---
ax2.set_title('Statistical Significance (Classification)', fontsize=10, fontweight='bold', pad=8)

for bar, p in zip(bars2, p_values):
    width = bar.get_width()
    ax2.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
             f'{p:.3f}', ha='left', va='center', fontsize=8, fontweight='bold')

ax2.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=1.5)

# --- Shared Legend (Moved Outside) ---
legend_elements = [
    Patch(facecolor='#5CDB95', edgecolor='black', label='Significant (p < 0.05)'),
    Patch(facecolor='#EE6C4D', edgecolor='black', label='Not Significant'),
    Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='p = 0.05')
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
           ncol=2, fontsize=8, frameon=True)

# --- KEY CHANGE 3: Slightly increased hspace to 0.15 to give titles breathing room ---
plt.subplots_adjust(top=0.88, hspace=0.15)

plt.savefig('single_column_ablation_tight.pdf', bbox_inches='tight')
plt.show()