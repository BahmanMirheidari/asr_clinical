import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- CRITICAL: Set font embedding BEFORE creating the figure ---
plt.rcParams['pdf.fonttype'] = 42      # Embed TrueType fonts (fixes blurry text)
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'

# 1. Prepare the data
data = {
    'BioMedNLP': [0.828, 0.804, 0.689, 0.863, 0.848, 0.804, 0.832, 0.843, 0.804, 0.735, 0.848, 0.843, 0.868],
    'ClinicalBERT': [0.828, 0.720, 0.735, 0.794, 0.785, 0.828, 0.828, 0.832, 0.828, 0.735, 0.832, 0.832, 0.832],
    'DistilRoBERTa': [0.828, 0.804, 0.735, 0.828, 0.868, 0.838, 0.868, 0.745, 0.838, 0.712, 0.828, 0.745, 0.843],
    'RoBERTa-B': [0.828, 0.858, 0.852, 0.863, 0.884, 0.858, 0.863, 0.811, 0.858, 0.843, 0.858, 0.811, 0.884],
    'RoBERTa-L': [0.828, 0.809, 0.712, 0.832, 0.809, 0.884, 0.852, 0.764, 0.884, 0.717, 0.832, 0.764, 0.891]
}
methods = [
    "Audio-Only", "Text-Only", "Early Fusion", "Late Fusion", 
    "Mixture of Experts", "Bilinear Fusion", "Adaptive Weighted Fusion", 
    "Confidence-Weighted", "Cross-Attention Fusion", "Interaction Stacking", 
    "Model-Based Stacking", "Confidence Selection", "Voting Ensemble"
]
df = pd.DataFrame(data, index=methods)

# 2. Set up the figure
fig, ax = plt.subplots(figsize=(6.5, 8))

# 3. Create the heatmap
sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu", 
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Macro-F1', 'shrink': 0.7},
            annot_kws={"size": 10, "weight": "bold"},
            ax=ax)

# 4. Formatting
ax.set_title('Macro-F1 Scores: Fusion & Ensemble Methods', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Method', fontsize=12, fontweight='bold')
plt.xticks(fontsize=11, rotation=45, ha='right')
plt.yticks(fontsize=11, rotation=0)

# --- CRITICAL: Use subplots_adjust instead of tight_layout ---
# This manually gives space for labels so nothing gets squished
plt.subplots_adjust(left=0.35, bottom=0.15, right=0.95, top=0.92)

# 5. Save as vector PDF (NO bbox_inches='tight'!)
plt.savefig('readable_heatmap.pdf', dpi=300, pad_inches=0.1)
plt.savefig('readable_heatmap.png', dpi=300, pad_inches=0.1) # Also save PNG as backup
plt.show()