"""
Standalone Sen/Spec figure generator — FOUR alternatives in one script.

Generates four compact figures for the same data (11 methods × 4 metrics):
  1. sen_spec_heatmap.png/pdf        — heatmap           (most compact)
  2. sen_spec_dumbbell.png/pdf       — dumbbell / gap    (best story)
  3. sen_spec_smallmultiples.png/pdf — 2×2 small panels  (most familiar)
  4. sen_spec_dotplot.png/pdf        — Cleveland dot     (least ink)

Each is sized for a single-column paper (3.5" wide).
Run:  python sen_spec_figures.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUT = Path('.')
OUT.mkdir(parents=True, exist_ok=True)


# ======================================================================
#  1. DATA — replace with your real (method × metric) numbers
# ======================================================================
# Columns:  Dys-Sen | Dys-Spec | Typ-Sen | Typ-Spec
DATA = {
    'Adaptive Weighted Fusion': (1.00, 0.78, 0.89, 0.89),
    'Audio-Only':               (1.00, 0.76, 0.86, 0.84),
    'Bilinear Fusion':          (1.00, 0.84, 0.69, 0.89),
    'Confidence-Weighted':      (1.00, 0.71, 0.93, 0.87),
    'Cross-Attention Fusion':   (1.00, 0.84, 0.69, 0.89),
    'Interaction Stacking':     (1.00, 0.66, 0.80, 0.75),
    'Mixture of Experts':       (1.00, 0.73, 0.86, 0.88),
    'Model-Based Stacking':     (1.00, 0.75, 0.86, 0.88),
    'Text-Only':                (1.00, 0.74, 0.77, 0.85),
    'Late Fusion':              (0.97, 0.75, 0.89, 0.89),
    'Early Fusion':             (0.95, 0.70, 0.75, 0.75),
}

df = pd.DataFrame(
    DATA, index=['Dys-Sen', 'Dys-Spec', 'Typ-Sen', 'Typ-Spec']
).T


# ======================================================================
#  Shared helper — abbreviate method names for tick labels
# ======================================================================
def shorten(name: str) -> str:
    return (
        name.replace(' Fusion', '')
            .replace('-Weighted', '')
            .replace('Model-Based ', 'MB-')
            .replace('Mixture of Experts', 'MoE')
            .replace('Cross-Attention', 'CrossAtt')
            .replace('Interaction Stacking', 'InterStack')
            .replace('Adaptive Weighted', 'AdaptW')
            .replace('Bilinear', 'Bilin')
    ).strip()


# ======================================================================
#  Figure 1 — Heatmap
# ======================================================================
def make_heatmap(df: pd.DataFrame):
    df_sorted = df.sort_values('Dys-Sen', ascending=False)

    fig_h = max(2.8, len(df_sorted) * 0.24)
    fig, ax = plt.subplots(figsize=(3.5, fig_h))

    sns.heatmap(
        df_sorted, annot=True, fmt='.3f', cmap='RdYlGn',
        vmin=0.5, vmax=1.0,
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Score', 'shrink': 0.6, 'pad': 0.02},
        ax=ax, annot_kws={'fontsize': 6.5, 'fontweight': 'bold'},
    )

    ax.set_title('Dys vs Typ Sen/Spec (Classification)',
                 fontsize=8, fontweight='bold', pad=4)
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=6.5, pad=1, length=0)
    ax.tick_params(axis='y', labelsize=6.5, pad=2, length=0)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', labeltop=True, labelbottom=False)

    plt.tight_layout()
    plt.savefig(OUT / 'sen_spec_heatmap.png', dpi=300,
                bbox_inches='tight', pad_inches=0.03)
    plt.savefig(OUT / 'sen_spec_heatmap.pdf',
                bbox_inches='tight', pad_inches=0.03)
    plt.close()
    print("✓ Saved sen_spec_heatmap.png / .pdf")


# ======================================================================
#  Figure 2 — Dumbbell / gap plot
# ======================================================================
def make_dumbbell(df: pd.DataFrame):
    d = df.copy()
    d['gap_spec'] = d['Typ-Spec'] - d['Dys-Spec']
    d = d.sort_values('gap_spec', ascending=False)

    methods = d.index.tolist()
    y = np.arange(len(methods))

    fig_h = max(3.0, len(methods) * 0.28)
    fig, ax = plt.subplots(figsize=(3.5, fig_h))

    off = 0.16
    for yi, m in enumerate(methods):
        # Sensitivity (top row of the pair)
        ds, ts = d.loc[m, 'Dys-Sen'], d.loc[m, 'Typ-Sen']
        ax.plot([ds, ts], [yi + off, yi + off],
                color='grey', lw=1.0, alpha=0.7, zorder=2)
        ax.scatter([ds], [yi + off], color='#E74C3C', s=22,
                   edgecolor='black', linewidth=0.5, zorder=3,
                   label='Dys – Sen' if yi == 0 else None)
        ax.scatter([ts], [yi + off], color='#3498DB', s=22,
                   edgecolor='black', linewidth=0.5, zorder=3,
                   label='Typ – Sen' if yi == 0 else None)

        # Specificity (bottom row of the pair)
        dp, tp = d.loc[m, 'Dys-Spec'], d.loc[m, 'Typ-Spec']
        ax.plot([dp, tp], [yi - off, yi - off],
                color='grey', lw=1.0, alpha=0.7, zorder=2)
        ax.scatter([dp], [yi - off], color='#E74C3C', s=22,
                   marker='s', edgecolor='black', linewidth=0.5, zorder=3,
                   label='Dys – Spec' if yi == 0 else None)
        ax.scatter([tp], [yi - off], color='#3498DB', s=22,
                   marker='s', edgecolor='black', linewidth=0.5, zorder=3,
                   label='Typ – Spec' if yi == 0 else None)

    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=6.2)
    ax.set_xlabel('Score', fontsize=7.5, labelpad=2)
    ax.set_xlim(0.4, 1.05)
    ax.set_ylim(-0.6, len(methods) - 0.4)
    ax.tick_params(axis='x', labelsize=6.5, pad=2, length=2.5)
    ax.tick_params(axis='y', length=0, pad=2)
    ax.grid(True, alpha=0.30, axis='x', linewidth=0.5)

    for yi in range(len(methods) - 1):
        ax.axhline(yi + 0.5, color='lightgrey', linewidth=0.4, alpha=0.6)

    ax.set_title('Dys↔Typ gaps — Sen (●) and Spec (■) (Classification)',
                 fontsize=8, fontweight='bold', pad=4)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.005),
              fontsize=6, ncol=4, frameon=True, framealpha=0.95,
              handletextpad=0.3, columnspacing=0.8, borderpad=0.3,
              labelspacing=0.2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(OUT / 'sen_spec_dumbbell.png', dpi=300,
                bbox_inches='tight', pad_inches=0.03)
    plt.savefig(OUT / 'sen_spec_dumbbell.pdf',
                bbox_inches='tight', pad_inches=0.03)
    plt.close()
    print("✓ Saved sen_spec_dumbbell.png / .pdf")


# ======================================================================
#  Figure 3 — 2 × 2 small multiples
# ======================================================================
def make_smallmultiples(df: pd.DataFrame):
    d = df.sort_values('Dys-Sen', ascending=False)
    methods = [shorten(m) for m in d.index]
    x = np.arange(len(methods))

    fig, axes = plt.subplots(
        2, 2,
        figsize=(3.5, 3.6),
        sharex=True, sharey=True,
        constrained_layout=True,
    )

    panels = [
        (axes[0, 0], 'Dys-Sen',  'Dys — Sensitivity',  '#E74C3C', 1.00),
        (axes[0, 1], 'Typ-Sen',  'Typ — Sensitivity',  '#3498DB', 1.00),
        (axes[1, 0], 'Dys-Spec', 'Dys — Specificity',  '#E74C3C', 0.55),
        (axes[1, 1], 'Typ-Spec', 'Typ — Specificity',  '#3498DB', 0.55),
    ]

    for ax, col, title, color, alpha in panels:
        ax.bar(x, d[col], width=0.72, color=color, alpha=alpha,
               edgecolor='black', linewidth=0.5)
        for xi, v in enumerate(d[col]):
            ax.text(xi, v + 0.02, f'{v:.2f}',
                    ha='center', va='bottom',
                    fontsize=4.6, fontweight='bold')
        ax.set_title(title, fontsize=6.5, fontweight='bold', pad=2)
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis='y', labelsize=5.5, pad=1, length=2)
        ax.grid(True, alpha=0.30, axis='y', linewidth=0.4)

    for ax in axes[1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=55, ha='right', fontsize=5.2)
        ax.tick_params(axis='x', length=2, pad=1)

    fig.suptitle('Dys vs Typ Sen/Spec by Method (Classification)',
                 fontsize=7.5, fontweight='bold', y=1.02)

    plt.savefig(OUT / 'sen_spec_smallmultiples.png', dpi=300,
                bbox_inches='tight', pad_inches=0.03)
    plt.savefig(OUT / 'sen_spec_smallmultiples.pdf',
                bbox_inches='tight', pad_inches=0.03)
    plt.close()
    print("✓ Saved sen_spec_smallmultiples.png / .pdf")


# ======================================================================
#  Figure 4 — Cleveland dot plot
# ======================================================================
def make_dotplot(df: pd.DataFrame):
    d = df.sort_values('Dys-Sen', ascending=True)  # ascending → best on top
    methods = d.index.tolist()
    y = np.arange(len(methods))

    fig_h = max(2.8, len(methods) * 0.26)
    fig, ax = plt.subplots(figsize=(3.5, fig_h))

    off = 0.20
    series = [
        ('Dys-Sen',  'Dys – Sen',  '#E74C3C', 'o', +off * 1.5),
        ('Dys-Spec', 'Dys – Spec', '#E74C3C', 's', +off * 0.5),
        ('Typ-Sen',  'Typ – Sen',  '#3498DB', 'o', -off * 0.5),
        ('Typ-Spec', 'Typ – Spec', '#3498DB', 's', -off * 1.5),
    ]

    for col, label, color, marker, yoff in series:
        ax.scatter(d[col], y + yoff,
                   color=color, marker=marker, s=18,
                   edgecolor='black', linewidth=0.4,
                   label=label, zorder=3, alpha=0.95)

    for yi, m in enumerate(methods):
        vals = [d.loc[m, c] for c, *_ in series]
        ax.plot([min(vals), max(vals)], [yi, yi],
                color='lightgrey', linewidth=0.8, alpha=0.6, zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=6.2)
    ax.set_xlabel('Score', fontsize=7.5, labelpad=2)
    ax.set_xlim(0.4, 1.05)
    ax.set_ylim(-0.6, len(methods) - 0.4)
    ax.tick_params(axis='x', labelsize=6.5, pad=2, length=2.5)
    ax.tick_params(axis='y', length=0, pad=2)
    ax.grid(True, alpha=0.30, axis='x', linewidth=0.5)

    ax.set_title('Dys vs Typ Sen/Spec (Classification)',
                 fontsize=8, fontweight='bold', pad=22)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.005),
              fontsize=6, ncol=4, frameon=True, framealpha=0.95,
              handletextpad=0.3, columnspacing=0.8, borderpad=0.3,
              labelspacing=0.2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.86)
    plt.savefig(OUT / 'sen_spec_dotplot.png', dpi=300,
                bbox_inches='tight', pad_inches=0.03)
    plt.savefig(OUT / 'sen_spec_dotplot.pdf',
                bbox_inches='tight', pad_inches=0.03)
    plt.close()
    print("✓ Saved sen_spec_dotplot.png / .pdf")


# ======================================================================
#  Run all four
# ======================================================================
if __name__ == '__main__':
    print(f"Writing figures to: {OUT.resolve()}\n")
    make_heatmap(df)
    make_dumbbell(df)
    make_smallmultiples(df)
    make_dotplot(df)
    print("\nAll four figures generated.")
    print("Compare PNGs side-by-side and pick the one that fits your column.")