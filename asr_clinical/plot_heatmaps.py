"""
Standalone heatmap generator — grouped by method family.

Reads all_results.csv from the aggregator and produces a heatmap with:

  Top:            Clinical-Only reference bar (LLM-independent), label on the left
  Middle blocks:  Baseline / Fusion / Ensemble, separated by thin gaps
  Within a block: sorted by mean macro-F1 descending

Palette: Blues (sequential, light = low, dark = high).

Outputs:
    heatmap_macro_f1_classification.pdf
    heatmap_macro_f1_classification.png
    heatmap_macro_f1_classification_matrix.csv
    heatmap_macro_f1_classification_baselines.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize


# =======================================================================
#  CONFIG
# =======================================================================

INPUT_CSV  = Path('/mnt/parscratch/users/ac1bm/MND-expr/outputs-ensemble-aggregate-classification/all_results.csv')
OUTPUT_DIR = Path('/mnt/parscratch/users/ac1bm/MND-expr/outputs-ensemble-aggregate-classification') 

METRIC    = 'macro_f1'
TASK_TYPE = 'classification'

FIG_WIDTH  = 4.5
FIG_HEIGHT = 5.0
CMAP       = 'Blues'
VMIN, VMAX = 0.68, 0.90


# =======================================================================
#  NAME HELPERS
# =======================================================================

SHORT_LABELS = {
    'Clinical-Feature-Only': 'Clinical-Only',
    'Text-Embedding-Only':   'Text-Only',
    'Audio-Only':            'Clinical-Only',
    'Text-Only':             'Text-Only',
}


def shorten_model(name: str) -> str:
    """Compact LLM names for axis ticks."""
    ll = name.lower()
    rules = [
        ('biomednlp', 'BioMedNLP'),
        ('biomed',    'BioMed'),
        ('clinical',  'ClinicalBERT'),
        ('distil',    'DistilRoBERTa'),
        ('roberta-large', 'RoBERTa-L'),
        ('roberta-l', 'RoBERTa-L'),
        ('roberta-base',  'RoBERTa-B'),
        ('roberta-b', 'RoBERTa-B'),
        ('roberta',   'RoBERTa'),
        ('deberta',   'DeBERTa'),
        ('albert',    'ALBERT'),
        ('bert',      'BERT'),
    ]
    for needle, label in rules:
        if needle in ll:
            return label
    return name[:12]


def classify_block(label: str) -> str:
    """Return 'Baseline', 'Fusion', or 'Ensemble'."""
    ll = str(label).lower().strip()
    if ('clinical-feature-only' in ll
            or 'text-embedding-only' in ll
            or 'audio-only' in ll
            or 'text-only' in ll):
        return 'Baseline'
    if ('ensemble' in ll
            or 'meta-fusion' in ll
            or 'metafusion' in ll
            or 'confidence selection' in ll
            or 'best method' in ll):
        return 'Ensemble'
    return 'Fusion'


# =======================================================================
#  LOAD
# =======================================================================

if not INPUT_CSV.exists():
    print(f"ERROR: {INPUT_CSV} not found.")
    sys.exit(1)

df = pd.read_csv(INPUT_CSV)
df = df[(df['Task'] == TASK_TYPE) & df[METRIC].notna()].copy()

if df.empty:
    print(f"ERROR: no valid {TASK_TYPE} / {METRIC} rows in {INPUT_CSV}")
    sys.exit(1)

df['Model_Short'] = df['Model'].apply(shorten_model)

method_col = 'Method_Label' if 'Method_Label' in df.columns else 'Method'

pivot = df.pivot_table(
    index=method_col, columns='Model_Short',
    values=METRIC, aggfunc='mean',
)
pivot = pivot.dropna(axis=1, how='all')

if pivot.empty:
    print("ERROR: pivot table empty")
    sys.exit(1)


# =======================================================================
#  SPLIT: LLM-INDEPENDENT BASELINES vs LLM-DEPENDENT ROWS
# =======================================================================

row_std = pivot.std(axis=1, skipna=True).fillna(0)
is_independent = row_std < 1e-6

baseline_rows = pivot[is_independent].copy()
main_rows     = pivot[~is_independent].copy()

if main_rows.empty:
    print("ERROR: no LLM-dependent rows to plot")
    sys.exit(0)

# Shorten labels for display
main_rows.index = [SHORT_LABELS.get(str(m), str(m)) for m in main_rows.index]
baseline_rows.index = [SHORT_LABELS.get(str(m), str(m))
                       for m in baseline_rows.index]


# =======================================================================
#  GROUP + ORDER
# =======================================================================

block_map = {m: classify_block(m) for m in main_rows.index}
block_order = ['Baseline', 'Fusion', 'Ensemble']

ordered_rows = []
block_sizes = {}

for block in block_order:
    members = [m for m in main_rows.index if block_map[m] == block]
    if not members:
        continue
    sub = main_rows.loc[members]
    means = sub.mean(axis=1).sort_values(ascending=False)
    ordered_rows.extend(means.index.tolist())
    block_sizes[block] = len(means)

# Fallback for any method that didn't match a block
unknown = [m for m in main_rows.index if m not in ordered_rows]
if unknown:
    sub = main_rows.loc[unknown]
    means = sub.mean(axis=1).sort_values(ascending=False)
    ordered_rows.extend(means.index.tolist())
    block_sizes['Other'] = len(means)

main_rows = main_rows.loc[ordered_rows]

print("Row ordering:")
for block in block_order + ['Other']:
    n = block_sizes.get(block, 0)
    if n == 0:
        continue
    print(f"  {block} ({n}):")
    for m in ordered_rows:
        if block_map.get(m, 'Other') == block:
            print(f"    {m}")


# =======================================================================
#  FIGURE
# =======================================================================

n_rows = len(main_rows)
n_cols = len(main_rows.columns)

TOP_PAD    = 0.9    # extra vertical space above the matrix for the reference bar
BOTTOM_PAD = 0.2

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

sns.heatmap(
    main_rows,
    annot=True, fmt='.3f',
    cmap=CMAP,
    cbar_kws={'label': 'Macro-F1', 'pad': 0.02, 'shrink': 0.8},
    linewidths=0.5, linecolor='white',
    ax=ax, annot_kws={'fontsize': 7.5},
    vmin=VMIN, vmax=VMAX,
)

# ---- Separator lines between blocks ----
block_boundaries = []
cursor = 0
for block in block_order:
    n = block_sizes.get(block, 0)
    if n == 0:
        continue
    cursor += n
    if cursor < n_rows:
        block_boundaries.append(cursor)
if 'Other' in block_sizes:
    cursor += block_sizes['Other']
    if cursor < n_rows:
        block_boundaries.append(cursor)

for y in block_boundaries:
    ax.axhline(y, color='white', linewidth=3.0, zorder=4)
    ax.axhline(y, color='#888888', linewidth=0.6, zorder=5)

# ---- Baseline reference bar(s) at the top, label on the LEFT ----
cmap_obj = plt.get_cmap(CMAP)
norm = Normalize(vmin=VMIN, vmax=VMAX)

for i, (name, row) in enumerate(baseline_rows.iterrows()):
    vals = row.dropna()
    if vals.empty:
        continue
    value = float(vals.iloc[0])
    colour = cmap_obj(norm(value))

    y_line = -0.55 - i * 0.55      # above the matrix (negative y in heatmap)

    # Coloured bar spanning the full matrix width
    ax.plot([0, n_cols], [y_line, y_line],
            color=colour, linewidth=8.0, solid_capstyle='butt',
            clip_on=False, zorder=5)

    # Thin black outlines for readability
    ax.plot([0, n_cols], [y_line + 0.13, y_line + 0.13],
            color='black', linewidth=0.4, clip_on=False, zorder=4)
    ax.plot([0, n_cols], [y_line - 0.13, y_line - 0.13],
            color='black', linewidth=0.4, clip_on=False, zorder=4)

    # Label on the LEFT side of the bar
    ax.text(-0.15, y_line,
            f'{name} = {value:.3f}',
            va='center', ha='right', fontsize=7.5, clip_on=False)

# Extend y-limits to include the top reference bar
ax.set_ylim(n_rows + BOTTOM_PAD, -TOP_PAD)

# ---- Cosmetics ----
ax.set_title('Macro-F1 by Method and LLM',
             fontsize=11, fontweight='bold', pad=6)
ax.set_xlabel('LLM', fontsize=9)
ax.set_ylabel('Method', fontsize=9)
ax.tick_params(axis='x', labelsize=8, rotation=30)
ax.tick_params(axis='y', labelsize=8, rotation=0)
for label in ax.get_xticklabels():
    label.set_ha('right')

plt.tight_layout()


# =======================================================================
#  SAVE
# =======================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

out_pdf = OUTPUT_DIR / f'heatmap_{METRIC}_{TASK_TYPE}.pdf'
out_png = OUTPUT_DIR / f'heatmap_{METRIC}_{TASK_TYPE}.png'
plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Saved: {out_pdf.name}")
print(f"✓ Saved: {out_png.name}")

# Sidecar CSV for the LLM-independent baselines
if not baseline_rows.empty:
    bl_csv = OUTPUT_DIR / f'heatmap_{METRIC}_{TASK_TYPE}_baselines.csv'
    baseline_rows.to_csv(bl_csv)
    print(f"✓ Saved baselines: {bl_csv.name}")

# Full sorted matrix as a CSV
matrix_csv = OUTPUT_DIR / f'heatmap_{METRIC}_{TASK_TYPE}_matrix.csv'
main_rows.round(4).to_csv(matrix_csv)
print(f"✓ Saved matrix:   {matrix_csv.name}")