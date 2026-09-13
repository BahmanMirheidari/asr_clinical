"""
Single-column, 3-row regression scatter figure — COMPACT & READABLE.

Readability improvements over the previous version:
  • Bigger fonts       (ticks 8, labels 8.5, titles 10, legend 7.5)
  • Bigger markers     (s = 45–55, thick black edges)
  • Thicker y=x line   (lw = 2.0)
  • Shared axis labels (only bottom / left panels carry text → saves ~0.6")
  • Compact legend     (3 short entries, no long metric strings)
  • Tighter layout     (small hspace, small pad_inches)
  • Shorter figure     (6.4" tall instead of 9.0")

Total footprint ≈ 3.5" × 6.4" — noticeably smaller than the previous 9"-tall
figure, yet the visual elements are ~40 % larger.
"""

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------
#  Synthetic data — swap for your real (observed, predicted) arrays
# ----------------------------------------------------------------------
np.random.seed(42)
n_dys, n_norm = 25, 26

x_dys  = np.linspace(80, 130, n_dys)
x_norm = np.linspace(75, 132, n_norm)

y_audio_dys   = 0.31 * x_dys  + 75.59 + np.random.normal(0, 5.5, n_dys)
y_audio_norm  = 0.25 * x_norm + 82.00 + np.random.normal(0, 7.0, n_norm)

y_text_dys    = 0.36 * x_dys  + 71.00 + np.random.normal(0, 5.5, n_dys)
y_text_norm   = 0.25 * x_norm + 83.00 + np.random.normal(0, 8.0, n_norm)

y_fusion_dys  = 0.38 * x_dys  + 68.00 + np.random.normal(0, 5.0, n_dys)
y_fusion_norm = 0.30 * x_norm + 78.00 + np.random.normal(0, 6.5, n_norm)


# ----------------------------------------------------------------------
#  Optional: compute RMSE / R² directly from your real arrays
# ----------------------------------------------------------------------
def _metrics(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    m = ~(np.isnan(obs) | np.isnan(pred))
    obs, pred = obs[m], pred[m]
    n = len(obs)
    if n < 2:
        return n, np.nan, np.nan
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
    r2 = float(np.corrcoef(obs, pred)[0, 1] ** 2) if n > 2 else np.nan
    return n, rmse, r2


# ----------------------------------------------------------------------
#  Panel renderer
# ----------------------------------------------------------------------
def draw_panel(ax, x_d, y_d, x_n, y_n, title,
               dys_lbl, norm_lbl,
               show_xlabel=False, show_ylabel=False,
               dys_color='#E95A4D', norm_color='#5BA2D6'):
    """Draw one compact scatter panel with Dys + Norm subgroups."""

    # ---- Scatter: bigger markers, thicker edges ----
    ax.scatter(x_d, y_d,
               color=dys_color, marker='o',
               edgecolor='black', linewidth=0.9,
               s=52, alpha=0.90, zorder=3,
               label=dys_lbl)
    ax.scatter(x_n, y_n,
               color=norm_color, marker='^',
               edgecolor='black', linewidth=0.9,
               s=48, alpha=0.80, zorder=3,
               label=norm_lbl)

    # ---- Common axis range ----
    all_vals = np.concatenate([x_d, y_d, x_n, y_n])
    lo = float(np.nanmin(all_vals)) - 4
    hi = float(np.nanmax(all_vals)) + 4

    # ---- y = x line: thicker, more visible ----
    ax.plot([lo, hi], [lo, hi],
            color='#555555', linestyle='--', linewidth=2.0,
            zorder=2, label='y = x')

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # ---- Title: larger, bold, tight padding ----
    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=3)

    # ---- Axis labels only on outer panels (saves vertical space) ----
    if show_xlabel:
        ax.set_xlabel('ECAS Observed',  fontsize=8.5, labelpad=2)
    if show_ylabel:
        ax.set_ylabel('ECAS Predicted', fontsize=8.5, labelpad=2)

    # ---- Tick labels: readable but compact ----
    ax.tick_params(axis='both', labelsize=8, pad=2, length=3, width=0.8)

    # ---- Grid: light but visible ----
    ax.grid(True, linestyle='--', alpha=0.45, linewidth=0.6)

    # ---- Compact in-panel legend (top-left, small but legible) ----
    ax.legend(loc='upper left', fontsize=7.5,
              framealpha=0.95, handletextpad=0.5,
              borderpad=0.4, labelspacing=0.35,
              borderaxespad=0.4, markerscale=1.0)


# ----------------------------------------------------------------------
#  Figure — single column, 3 stacked panels, compact overall height
# ----------------------------------------------------------------------
fig, axes = plt.subplots(
    3, 1,
    figsize=(3.5, 6.4),     # ← was (3.5, 9.0);  ~30 % shorter
    constrained_layout=True,
)

# ---- Panel 1: Audio-Only (only y-label) ----
draw_panel(
    axes[0],
    x_dys,  y_audio_dys,
    x_norm, y_audio_norm,
    title='Audio-Only',
    dys_lbl ='Dys  (RMSE 10.3, R² 0.39)',
    norm_lbl='Norm (RMSE 13.6, R² 0.28)',
    show_xlabel=False,
    show_ylabel=True,
)

# ---- Panel 2: Best Text-Only ----
draw_panel(
    axes[1],
    x_dys,  y_text_dys,
    x_norm, y_text_norm,
    title='Best Text-Only',
    dys_lbl ='Dys  (RMSE 10.6, R² 0.35)',
    norm_lbl='Norm (RMSE 14.9, R² 0.12)',
    show_xlabel=False,
    show_ylabel=True,
)

# ---- Panel 3: Best Fusion (only x-label) ----
draw_panel(
    axes[2],
    x_dys,  y_fusion_dys,
    x_norm, y_fusion_norm,
    title='Best Fusion',
    dys_lbl ='Dys  (RMSE 10.1, R² 0.45)',
    norm_lbl='Norm (RMSE 13.5, R² 0.31)',
    show_xlabel=True,
    show_ylabel=True,
)

# ----------------------------------------------------------------------
#  Save: vector PDF for LaTeX, PNG for preview
# ----------------------------------------------------------------------
plt.savefig('dys_norm_regression_compact.pdf', bbox_inches='tight', pad_inches=0.02)
plt.savefig('dys_norm_regression_compact.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.show()