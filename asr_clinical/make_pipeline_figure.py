"""
make_pipeline_simple.py
Simplified overview of the multimodal fusion pipeline (text/LLM + clinical only).
Output:  figure/pipeline_simple.pdf  +  figure/pipeline_simple.png (300 dpi)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

EDGE = "#2C3E50"
C = {
    "in":     "#EBF1FA",
    "asr":    "#DCE8F5",
    "text":   "#FCE8D0",
    "clin":   "#DFF0D8",
    "merge":  "#F5F0DC",
    "fusion": "#FFF2CC",
    "meta":   "#F8D7D2",
    "post":   "#D9E2F3",
    "out":    "#CFE2F3",
}

fig, ax = plt.subplots(figsize=(7, 9))
ax.set_xlim(0, 100)
ax.set_ylim(0, 104)
ax.axis("off")


def box(cx, cy, w, h, text, fc, fs=10.5, bold=True):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.3,rounding_size=1.0",
        facecolor=fc, edgecolor=EDGE, linewidth=1.2, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=EDGE, zorder=4)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
        lw=1.4, color=EDGE, zorder=2, shrinkA=0, shrinkB=0))


H = 7          # standard box height
W = 56         # standard box width
gap = 12.5

y_in    = 97
y_asr   = y_in - gap
y_mod   = y_asr - gap - 1
y_feat  = y_mod - gap - 1
y_fus   = y_feat - gap
y_meta  = y_fus - gap
y_post  = y_meta - gap
y_out   = y_post - gap

box(50, y_in,  W, H, "Inputs\naudio  ·  transcripts  ·  labels", C["in"])

# audio -> clinical (right), transcripts -> ASR -> text (left)
box(26, y_asr, 36, H, "ASR\nspeech → transcript", C["asr"])
arrow(26, y_in - H/2 - 0.3, 26, y_asr + H/2 + 0.3)

mod_h = 8.5
arrow(26, y_asr - H/2 - 0.3, 26, y_mod + mod_h/2 + 0.3)
box(26, y_mod, 36, mod_h, "Text / LLM\nembeddings", C["text"])

arrow(74, y_in - H/2 - 0.3, 74, y_mod + mod_h/2 + 0.3)
box(74, y_mod, 36, mod_h, "Clinical\nfeatures", C["clin"])

# merge
arrow(26, y_mod - mod_h/2 - 0.3, 40, y_feat + H/2 + 0.3)
arrow(74, y_mod - mod_h/2 - 0.3, 60, y_feat + H/2 + 0.3)

box(50, y_feat, W, H, "Speaker-level\nfeature table", C["merge"])
arrow(50, y_feat - H/2 - 0.3, 50, y_fus + H/2 + 0.3)

box(50, y_fus, W, H, "Fusion", C["fusion"])
arrow(50, y_fus - H/2 - 0.3, 50, y_meta + H/2 + 0.3)

box(50, y_meta, W, H, "Meta-fusion", C["meta"])
arrow(50, y_meta - H/2 - 0.3, 50, y_post + H/2 + 0.3)

box(50, y_post, W, H, "Post-processing", C["post"])
arrow(50, y_post - H/2 - 0.3, 50, y_out + H/2 + 0.3)

box(50, y_out, W, H, "Outputs\npredictions  ·  metrics", C["out"])

out_dir = Path("figure")
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / "pipeline_simple.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(out_dir / "pipeline_simple.pdf", bbox_inches="tight", facecolor="white")
print("Saved figure/pipeline_simple.png and figure/pipeline_simple.pdf")
