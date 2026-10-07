"""Figure for the scarcity sweep.

Reads ``results/scarcity_sweep.csv``, writes ``figures/scarcity.pdf``. Run after ``sweep.py``.

One panel, because there is one thing to see: the budget per context on the horizontal axis
and the predictive score of each agent relative to ours on the vertical. An agent on the zero
line is indistinguishable from ours at that budget. The systematic schedule sits on the line
while the budget is ample and falls away once it is not, which is the whole finding.

Run: python experiments/bulk-sampling/visualize_sweep.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "common"))

import matplotlib.pyplot as plt                                       # noqa: E402

import viz                                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
viz.apply_style()

STYLE = {
    "uniform": dict(color=viz.CAT[0], linewidth=2.4, marker="o", markersize=4.0),
    "eig@gamma-poisson": dict(color=viz.CAT[1], linewidth=2.0, marker="s",
                              markersize=3.6),
    "maxent": dict(color=viz.CAT[2], linewidth=2.0, marker="^", markersize=3.8),
    "d-optimality": dict(color=viz.INK["secondary"], linewidth=1.1,
                         linestyle=(0, (6, 2, 1, 2)), marker=".", markersize=4),
    "neyman": dict(color=viz.INK["secondary"], linewidth=1.1, linestyle=(0, (1, 2)),
                   marker=".", markersize=4),
    "eig@lognormal-poisson": dict(color=viz.INK["muted"], linewidth=1.1,
                                  linestyle=(0, (4, 2)), marker=".", markersize=4),
    "thompson": dict(color=viz.INK["muted"], linewidth=1.0,
                     linestyle=(0, (2, 2, 1, 2))),
    "random": dict(color=viz.INK["muted"], linewidth=1.0, linestyle=(0, (2, 3))),
}
LABEL = {"uniform": "systematic", "eig@gamma-poisson": "gamma-Poisson model",
         "maxent": "max-entropy", "d-optimality": "D-optimality",
         "neyman": "Neyman alloc.", "eig@lognormal-poisson": "lognormal model",
         "thompson": "Thompson", "random": "random"}


def main():
    d = np.genfromtxt(os.path.join(HERE, "results", "scarcity_sweep.csv"),
                      delimiter=",", names=True, dtype=None, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(4.6, 2.05))
    for name, st in STYLE.items():
        m = d["policy"] == name
        if not np.any(m):
            continue
        order = np.argsort(d["budget_per_block"][m])
        ax.plot(d["budget_per_block"][m][order], d["nlpd_delta_vs_eig"][m][order],
                markeredgewidth=0.0, **st)
    ax.axhline(0.0, color=viz.INK["primary"], linewidth=1.0)
    ax.annotate("indistinguishable from ours", xy=(0.97, 0.0), xycoords=("axes fraction",
                "data"), xytext=(0, 4), textcoords="offset points", fontsize=7.5,
                color=viz.INK["secondary"], ha="right")
    ax.set_xscale("log")
    ax.set_xlabel(r"budget per block (m$^3$)   $\longrightarrow$ more abundant")
    ax.set_ylabel("held-out NLPD, relative to ours (nats)")
    # Eight entries over a panel whose whole point is the flat region near zero, so the
    # legend goes below the axes rather than over the curves it explains. Four columns
    # rather than three, because two rows of legend cost less height than three.
    ax.legend(handles=[plt.Line2D([], [], **STYLE[k], label=LABEL[k]) for k in STYLE],
              loc="upper left", bbox_to_anchor=(0.0, -0.34), fontsize=6.2, ncol=4,
              handlelength=1.8, columnspacing=0.8, labelspacing=0.22,
              frameon=False, borderaxespad=0.0)
    viz.savefig(fig, os.path.join(HERE, "figures", "scarcity.pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
