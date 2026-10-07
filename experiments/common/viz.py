"""Figure style: one system for every plot in the paper.

Carried over unchanged from the companion project so that figures in the two papers read
as one system. Palette is the validated reference instance (categorical slots 1-3, light
mode), whose documented all-pairs separation under colour-vision deficiency covers scatter
and heatmap forms as well as lines. Slots are assigned in fixed order and never cycled; a
fourth series is not a fourth hue, it is a facet.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

__all__ = ["CAT", "SEQ", "DIV", "INK", "apply_style", "savefig", "label_line"]

# Categorical, fixed order: blue, orange, aqua.
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]

# Sequential: blue ramp, steps 100->700.
_SEQ_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]
SEQ = LinearSegmentedColormap.from_list("gig_seq", _SEQ_STEPS)

# Diverging: blue <-> red through neutral gray (never a hue at the midpoint).
DIV = LinearSegmentedColormap.from_list(
    "gig_div",
    ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#8f1d1c"],
)

INK = {
    "surface": "#fcfcfb",
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}


def apply_style():
    """Recessive chrome, thin marks, no top/right spines."""
    mpl.rcParams.update({
        "figure.facecolor": INK["surface"],
        "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.labelcolor": INK["secondary"],
        "axes.edgecolor": INK["axis"],
        "axes.linewidth": 0.8,
        "axes.titlecolor": INK["primary"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": INK["grid"],
        "grid.linewidth": 0.6,
        "xtick.color": INK["muted"],
        "ytick.color": INK["muted"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 4.5,
        "figure.dpi": 130,
        "figure.constrained_layout.use": True,
        # Embed TrueType (Type 42) rather than Type 3 fonts, as conference PDFs require.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    # Never rely on the default cycler beyond our three fixed slots.
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=CAT)


def label_line(ax, x, y, text, color, dx=0.0, dy=0.0, ha="left", va="center"):
    """Direct label on a series, in series colour, placed at the line's end."""
    ax.annotate(text, xy=(x, y), xytext=(x + dx, y + dy), color=color,
                fontsize=8, fontweight="bold", ha=ha, va=va, annotation_clip=False)


def savefig(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("  wrote {}".format(path))
