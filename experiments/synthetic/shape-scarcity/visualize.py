"""Figures for the shape-and-scarcity study.

Reads ``results/summary.csv``, writes ``figures/shape_scarcity.pdf``. Run after ``run.py``.

Two panels for one claim. Panel (a) is the result: the accuracy the exact criterion buys
over the two-moment surrogate, as a function of how far the budget has to stretch, at two
count levels. It crosses zero, and the crossing is the point of the figure as much as the
gain to its left is. Panel (b) is the mechanism: where each criterion actually sent the
budget, which is what the scores in the design gate predict and what the accuracy follows.

Run: python experiments/synthetic/shape-scarcity/visualize.py
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "common"))

import matplotlib.pyplot as plt                                       # noqa: E402

import viz                                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
viz.apply_style()

N_CONTEXTS = 12

#: The two count levels, in plotting order, with the slot each takes.
REGIME_STYLE = {
    "low":  dict(color=viz.CAT[0], label="low count"),
    "high": dict(color=viz.CAT[1], label="high count"),
}


def load(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k not in ("regime", "agent"):
                r[k] = float(v)
    return rows


def series(rows, regime, agent, field):
    sel = [r for r in rows if r["regime"] == regime and r["agent"] == agent]
    sel.sort(key=lambda r: r["budget"])
    return (np.array([r["budget"] / N_CONTEXTS for r in sel]),
            np.array([r[field] for r in sel]))


def paired(episodes, regime, field, ours="eig", theirs="d-optimality"):
    """Mean and standard error of ``theirs - ours`` over the episodes they share.

    The two arms are run on the same episodes, the same drawn rates and the same realised
    counts, so the comparison is paired and its error bar is the error of the difference.
    Treating the two as independent and adding their variances would inflate the band by
    about an order of magnitude here and hide an effect the paired ``t`` in
    Table~\\ref{tab:shape_scarcity} reports at many standard errors.
    """
    budgets = sorted({r["budget"] for r in episodes if r["regime"] == regime})
    xs, means, sems = [], [], []
    for b in budgets:
        byep = {}
        for r in episodes:
            if r["regime"] == regime and r["budget"] == b and r["agent"] in (ours, theirs):
                byep.setdefault(r["episode"], {})[r["agent"]] = r[field]
        d = np.array([v[theirs] - v[ours] for v in byep.values()
                      if ours in v and theirs in v])
        xs.append(b / N_CONTEXTS)
        means.append(float(d.mean()))
        sems.append(float(d.std(ddof=1) / np.sqrt(d.size)))
    return np.array(xs), np.array(means), np.array(sems)


def panel_gain(ax, episodes):
    """(a) Accuracy the exact criterion buys over D-optimality, against scarcity."""
    ends = {}
    for regime, style in REGIME_STYLE.items():
        x, gain, sem = paired(episodes, regime, "dex")
        band = 2.0 * sem
        ax.plot(x, gain, color=style["color"], zorder=5)
        ax.fill_between(x, gain - band, gain + band, color=style["color"],
                        alpha=0.18, linewidth=0, zorder=3)
        ends[regime] = (x[0], gain[0])
    # Direct labels at the scarce end, where the two regimes are furthest apart. Nudged
    # apart only if the curves happen to start within a hair of each other.
    order = sorted(ends, key=lambda r: ends[r][1])
    span = max(abs(v[1]) for v in ends.values()) or 1.0
    for i, regime in enumerate(order):
        x0, y0 = ends[regime]
        # Lift the label clear of its own curve; on a log axis the nudge is a factor.
        dy = 0.16 * span
        if len(order) == 2 and abs(ends[order[0]][1] - ends[order[1]][1]) < 0.20 * span:
            dy = (-0.18 if i == 0 else 0.18) * span
        ax.annotate(REGIME_STYLE[regime]["label"], xy=(x0 * 1.02, y0 + dy),
                    color=REGIME_STYLE[regime]["color"], fontsize=8, fontweight="bold",
                    ha="left", va="center", annotation_clip=False)
    ax.axhline(0.0, color=viz.INK["secondary"], linewidth=0.9, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("budget per context" + r"  $\longrightarrow$  more abundant")
    ax.set_ylabel("rate error saved (dex)")
    ax.set_title("(a) what the exact criterion buys")
    ax.margins(y=0.14)
    for text, va, frac in (("EIG ahead", "top", 0.96), ("D-optimality ahead", "bottom", 0.04)):
        ax.annotate(text, xy=(0.97, frac), xycoords="axes fraction", ha="right", va=va,
                    fontsize=7, color=viz.INK["muted"])


def panel_where(ax, rows):
    """(b) Share of the budget each criterion sent to the heavy-shape class."""
    for regime, style in REGIME_STYLE.items():
        for agent, dash in (("eig", None), ("d-optimality", (0, (3, 2)))):
            x, share = series(rows, regime, agent, "heavy_share")
            ax.plot(x, 100.0 * share, color=style["color"],
                    linestyle="-" if dash is None else dash,
                    linewidth=2.0 if dash is None else 1.6, zorder=5)
    ax.axhline(50.0, color=viz.INK["muted"], linewidth=0.8, linestyle=":", zorder=2)
    ax.set_xscale("log")
    ax.set_ylim(-5, 105)
    ax.set_xlabel("budget per context")
    ax.set_ylabel("budget to heavy-shape class (%)")
    ax.set_title("(b) where each criterion sent it")
    ax.annotate("even split", xy=(0.03, 52.0), xycoords=("axes fraction", "data"),
                fontsize=7, color=viz.INK["muted"], va="bottom", ha="left")
    handles = [plt.Line2D([], [], color=viz.INK["secondary"], linewidth=2.0),
               plt.Line2D([], [], color=viz.INK["secondary"], linewidth=1.6,
                          linestyle=(0, (3, 2)))]
    # Colours are keyed by the direct labels in panel (a), so this legend carries only
    # the line styles; four entries would sit on the curves at this height.
    ax.legend(handles, ["EIG (ours)", "D-optimality"],
              loc="upper right", bbox_to_anchor=(1.0, 0.99), handlelength=1.6,
              labelspacing=0.22, borderpad=0.2)


def main():
    rows = load(os.path.join(HERE, "results", "summary.csv"))
    episodes = load(os.path.join(HERE, "results", "episodes.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 1.40))
    panel_gain(axes[0], episodes)
    panel_where(axes[1], rows)
    viz.savefig(fig, os.path.join(HERE, "figures", "shape_scarcity.pdf"))


if __name__ == "__main__":
    main()
