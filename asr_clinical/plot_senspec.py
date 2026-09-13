import matplotlib.pyplot as plt
import numpy as np

# Data extraction from the image
models = ['DeBERTa-B', 'RoBERTa-B', 'RoBERTa-L', 'DistilR', 'PubMed-B']

dys_sens = [1.00, 1.00, 0.99, 0.99, 0.98]
dys_spec = [0.70, 0.89, 0.71, 0.75, 0.75]
typ_sens = [0.90, 0.70, 0.88, 0.74, 0.81]
typ_spec = [0.82, 0.88, 0.86, 0.88, 0.85]

# --- KEY CHANGE: Reduced height from 4.5 to 2.8 ---
fig, ax = plt.subplots(figsize=(3.5, 2.8))

# Bar width and positions
x = np.arange(len(models))
width = 0.18

# Define colors
color_dys_sens = '#E95A4D'
color_dys_spec = '#F4A59B'
color_typ_sens = '#5BA2D6'
color_typ_spec = '#A8D0E6'

# Plot the bars
rects1 = ax.bar(x - 1.5*width, dys_sens, width, label='Dys - Sensitivity', 
                color=color_dys_sens, edgecolor='black', linewidth=0.5)
rects2 = ax.bar(x - 0.5*width, dys_spec, width, label='Dys - Specificity', 
                color=color_dys_spec, edgecolor='black', linewidth=0.5, hatch='//')
rects3 = ax.bar(x + 0.5*width, typ_sens, width, label='Typ - Sensitivity', 
                color=color_typ_sens, edgecolor='black', linewidth=0.5)
rects4 = ax.bar(x + 1.5*width, typ_spec, width, label='Typ - Specificity', 
                color=color_typ_spec, edgecolor='black', linewidth=0.5, hatch='//')

# Add value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 1.5), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=5.5, fontweight='bold') # Reduced font slightly

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

# Formatting
ax.set_ylabel('Score', fontsize=8)
ax.set_xlabel('Model', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=7.5)
ax.set_ylim(0, 1.15)  # Keep room for labels on top
#ax.set_title('Sensitivity & Specificity: Dys vs Typ\n(Classification)', fontsize=9, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Legend placed above, but with smaller font
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=6, frameon=True)

# --- KEY CHANGE: Adjusted layout to fit the tighter space ---
plt.tight_layout()
# Manually adjust top and bottom to ensure labels don't get cut off
plt.subplots_adjust(top=0.82, bottom=0.25) 

plt.savefig('single_column_sens_spec_short.pdf', bbox_inches='tight')
plt.show()