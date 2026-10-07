"""Figure for the nested Monte Carlo cost study.

Reads ``results/``, writes ``figures/nmc_cost.pdf``. Run after ``run.py`` and ``refine.py``.

Three panels, one argument. (a) What accuracy costs: the estimator's error against its
wall-clock, with the quadrature marked at its own cost and its own measured error floor.
(b) Why paying more does not fix it: the bias is a function of the inner sample size, so
lines of fixed inner size run flat however far the outer size is pushed. (c) Whether any of
it reaches a decision: how often the estimator's preferred action differs from the exact one.

Run: python experiments/synthetic/nmc-cost/visualize.py
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

#: The cases drawn in panel (a): a representative one, the hard one, and the easy one.
CASES = ((("default", 4.0), viz.CAT[0], "-", "default, f=4"),
         (("heavy", 40.0), viz.CAT[1], "-", "heavy tail, f=40"),
         (("mild", 0.5), viz.CAT[2], "-", "mild, f=0.5"))


def load(name):
    with open(os.path.join(HERE, "results", name), newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                d[k] = v
        out.append(d)
    return out


#: Floors below this are drawn at it. The measured floors span 1e-16 to 2.6e-8 (see
#: ``results/floor.csv``); plotting the smallest as measured stretches the axis by six
#: decades to make a point about a case that happens to have cancelled exactly, and the
#: honest headline is the *worst* floor, which is well above this clip.
FLOOR_CLIP = 1e-10


def panel_cost(ax, acc, floor):
    floor_by = {(r["belief"], r["exposure"]): max(r["abs_diff"], FLOOR_CLIP)
                for r in floor}
    for key, colour, ls, label in CASES:
        pts = sorted([r for r in acc
                      if (r["belief"], r["exposure"]) == key and r["method"] == "nmc"],
                     key=lambda r: r["ms"])
        if not pts:
            continue
        x = np.array([r["ms"] for r in pts])
        y = np.array([max(r["rmse"], 1e-16) for r in pts])
        ax.plot(x, y, color=colour, linestyle=ls, zorder=5)
        dy = {0: 6, 1: 0, 2: -7}[CASES.index((key, colour, ls, label))]
        ax.annotate(label, xy=(x[-1], y[-1]), xytext=(4, dy),
                    textcoords="offset points", color=colour, fontsize=7.5,
                    fontweight="bold", va="center", annotation_clip=False)
        q = [r for r in acc
             if (r["belief"], r["exposure"]) == key and r["method"] == "quadrature"]
        if q:
            ax.plot([q[0]["ms"]], [floor_by.get(key, 1e-12)], marker="*",
                    markersize=11, color=colour, markeredgecolor="white",
                    markeredgewidth=0.6, linestyle="none", zorder=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.3 * FLOOR_CLIP)
    ax.set_xlabel("milliseconds per evaluation")
    ax.set_ylabel("error in the EIG (nats)")
    ax.set_title("(a) what accuracy costs")
    ax.annotate("quadrature ($\\bigstar$)", xy=(0.03, 0.06), xycoords="axes fraction",
                fontsize=7.5, color=viz.INK["secondary"], ha="left")


def panel_bias(ax, bias):
    inners = sorted({r["n_inner"] for r in bias})
    cmap = viz.SEQ(np.linspace(0.30, 0.92, len(inners)))
    for colour, m_in in zip(cmap, inners):
        pts = sorted([r for r in bias if r["n_inner"] == m_in and r["belief"] == "default"],
                     key=lambda r: r["n_outer"])
        if not pts:
            continue
        x = np.array([r["n_outer"] for r in pts])
        y = np.array([r["bias"] for r in pts])
        e = np.array([r["sem"] for r in pts])
        ax.errorbar(x, y, yerr=2 * e, color=colour, marker="o", markersize=3.2,
                    capsize=2, linewidth=1.6, zorder=5)
        ax.annotate("M={:g}".format(m_in), xy=(x[-1], y[-1]),
                    xytext=(4, 7 * (inners.index(m_in) - 1.5)),
                    textcoords="offset points", color=colour, fontsize=7,
                    fontweight="bold", va="center", annotation_clip=False)
    ax.axhline(0.0, color=viz.INK["secondary"], linewidth=0.9, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("outer samples $N$")
    ax.set_ylabel("bias (nats)")
    ax.set_title("(b) the bias is set by $M$, not $N$")


def panel_decisions(ax, dec, acc):
    ms_at = {}
    for r in acc:
        if r["method"] == "nmc":
            ms_at.setdefault(r["n_outer"], []).append(r["ms"])
    pts = sorted(dec, key=lambda r: r["n"])
    x = np.array([np.median(ms_at.get(r["n"], [np.nan])) for r in pts])
    y = np.array([100.0 * r["flip_rate"] for r in pts])
    n = pts[0]["n_beliefs"]
    e = 100.0 * np.sqrt(np.clip(y / 100.0, 0, 1) * (1 - np.clip(y / 100.0, 0, 1)) / n)
    ax.errorbar(x, y, yerr=2 * e, color=viz.CAT[0], marker="o", markersize=4,
                capsize=2, zorder=5)
    for xi, yi, r in zip(x, y, pts):
        ax.annotate("n={:g}".format(r["n"]), xy=(xi, yi), xytext=(0, 7),
                    textcoords="offset points", fontsize=6.5,
                    color=viz.INK["muted"], ha="center")
    ax.axhline(0.0, color=viz.CAT[2], linewidth=1.6, zorder=4)
    ax.annotate("quadrature", xy=(0.97, 0.03), xycoords="axes fraction", ha="right",
                va="bottom", fontsize=7.5, color=viz.CAT[2], fontweight="bold")
    ax.set_xscale("log")
    ax.set_ylim(bottom=-4)
    ax.set_xlabel("milliseconds per evaluation")
    ax.set_ylabel("chosen action differs (% of problems)")
    ax.set_title("(c) whether it reaches the decision")


def main():
    acc, bias, dec = load("accuracy.csv"), load("bias.csv"), load("decisions.csv")
    floor = load("floor.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.5))
    panel_cost(axes[0], acc, floor)
    panel_bias(axes[1], bias)
    panel_decisions(axes[2], dec, acc)
    viz.savefig(fig, os.path.join(HERE, "figures", "nmc_cost.pdf"))


if __name__ == "__main__":
    main()
