"""Figures for the bulk-sampling study.

Reads ``results/``, writes ``figures/bulk.pdf``. Run after ``run.py``.

Three panels for two questions. Panel (a) holds the model fixed and varies the acquisition;
panel (b) holds the acquisition fixed and varies the mixing law. Keeping them apart matters,
because the two effects are an order of magnitude different in size and a single crowded
axis would hide that. Panel (c) shows the mechanism behind (a).

Run: python experiments/bulk-sampling/visualize.py
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

#: Panel (a), the acquisition axis. Ours takes the first slot, the two criteria that come
#: closest take the others, and the rest are facets in muted ink.
CRITERIA_STYLE = {
    "eig": dict(color=viz.CAT[0], linewidth=2.4, zorder=6),
    "epig": dict(color=viz.CAT[1], linewidth=2.0, zorder=5),
    "maxent": dict(color=viz.CAT[2], linewidth=2.0, zorder=5),
    "neyman": dict(color=viz.INK["primary"], linewidth=1.2, linestyle=(0, (5, 2))),
    "uniform": dict(color=viz.INK["secondary"], linewidth=1.2, linestyle=(0, (1, 2))),
    "d-optimality": dict(color=viz.INK["secondary"], linewidth=1.0,
                         linestyle=(0, (6, 2, 1, 2))),
    "epistemic": dict(color=viz.INK["muted"], linewidth=1.0),
    "predictive-variance": dict(color=viz.INK["muted"], linewidth=1.0,
                                linestyle=(0, (4, 2))),
    "thompson": dict(color=viz.INK["muted"], linewidth=1.0, linestyle=(0, (2, 2, 1, 2))),
    "random": dict(color=viz.INK["muted"], linewidth=1.0, linestyle=(0, (2, 3))),
}

#: Panel (b), the model axis. Three mixing laws, one acquisition.
MODEL_STYLE = {
    "eig": dict(color=viz.CAT[0], linewidth=2.4, zorder=6),
    "eig@gamma-poisson": dict(color=viz.CAT[1], linewidth=2.0, zorder=5),
    "eig@lognormal-poisson": dict(color=viz.CAT[2], linewidth=2.0, zorder=5),
}

LABEL = {"eig": "EIG (ours)", "epig": "EPIG", "maxent": "max-entropy",
         "neyman": "Neyman alloc.", "uniform": "systematic",
         "d-optimality": "D-optimality", "epistemic": "epistemic var.",
         "predictive-variance": "total var.", "thompson": "Thompson",
         "random": "random",
         "eig@gamma-poisson": "gamma-Poisson",
         "eig@lognormal-poisson": "lognormal-Poisson"}


def load(name):
    return np.atleast_1d(np.genfromtxt(os.path.join(HERE, "results", name),
                                       delimiter=",", names=True, dtype=None,
                                       encoding="utf-8"))


def curves(ax, seq, style, column, band):
    for name, st in style.items():
        m = seq["policy"] == name
        if not np.any(m):
            continue
        spent = np.unique(seq["spent"][m])
        mean = np.array([np.mean(seq[column][m & (seq["spent"] == s)]) for s in spent])
        sem = np.array([np.std(seq[column][m & (seq["spent"] == s)], ddof=1)
                        / np.sqrt(np.sum(m & (seq["spent"] == s))) for s in spent])
        ax.plot(spent, mean, **st)
        if name in band:
            ax.fill_between(spent, mean - sem, mean + sem, color=st["color"],
                            alpha=0.18, linewidth=0)
    ax.set_xlabel(r"gravel processed (m$^3$)")


def legend(ax, style, **kw):
    ax.legend(handles=[plt.Line2D([], [], **st, label=LABEL[k])
                       for k, st in style.items()], **kw)


def main():
    seq = load("sequential.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.2))

    ax = axes[0]
    curves(ax, seq, CRITERIA_STYLE, "nlpd", band=("eig", "epig"))
    ax.set_ylabel("held-out NLPD (nats)")
    ax.set_title("(a) acquisition, model fixed", loc="left")
    # Ten entries, and no free space inside the axes: every region of this panel carries a
    # curve. The legend therefore sits below the panel, cleared of the x-label.
    legend(ax, CRITERIA_STYLE, loc="upper left", bbox_to_anchor=(0.0, -0.22),
           fontsize=6.5, ncol=3, handlelength=2.0, columnspacing=1.0,
           labelspacing=0.35, frameon=False, borderaxespad=0.0)

    ax = axes[1]
    curves(ax, seq, MODEL_STYLE, "nlpd", band=tuple(MODEL_STYLE))
    ax.set_ylabel("held-out NLPD (nats)")
    ax.set_title("(b) mixing law, acquisition fixed", loc="left")
    legend(ax, MODEL_STYLE, loc="upper right", fontsize=7, handlelength=2.2)

    # (c) the mechanism behind (a): where the budget went, against what there was to learn.
    alloc = load("allocation.csv")
    ax = axes[2]
    for name in ("eig", "epig", "maxent", "neyman", "uniform"):
        m = alloc["policy"] == name
        g, v = alloc["grade_true"][m], alloc["volume_allocated"][m]
        order = np.argsort(g)
        edges = np.quantile(g, np.linspace(0, 1, 9))
        idx = np.clip(np.digitize(g[order], edges[1:-1]), 0, 7)
        ax.plot([np.median(g[order][idx == i]) for i in range(8)],
                [np.mean(v[order][idx == i]) for i in range(8)],
                marker="o", markersize=3.2, markeredgewidth=0.0,
                **CRITERIA_STYLE[name])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"true grade of the block (stones m$^{-3}$)")
    ax.set_ylabel(r"budget given to it (m$^3$)")
    ax.set_title("(c) where the budget went", loc="left")

    viz.savefig(fig, os.path.join(HERE, "figures", "bulk.pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
