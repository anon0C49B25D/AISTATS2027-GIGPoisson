"""Figures for the photon-counting study.

Reads ``results/``, writes ``figures/photon.pdf``. Run after ``fit.py`` and ``run.py``.

Two panels. (a) What the acquisition is worth once the budget is scarce enough that most
targets cannot be observed. (b) Where the telescope time went, against how bright each
target turned out to be, which is the mechanism behind (a).

The measured flux field against each fitted mixing law used to be a third panel here. It is
the first panel of the rate-fields figure, so it is shown there once instead of twice.

Run: python experiments/gamma-ray/visualize.py
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
    "eig": dict(color=viz.CAT[0], linewidth=2.4, zorder=6),
    "eig@gamma-poisson": dict(color=viz.CAT[1], linewidth=2.0, zorder=5),
    "eig@lognormal-poisson": dict(color=viz.CAT[2], linewidth=2.0, zorder=5),
    "uniform": dict(color=viz.INK["primary"], linewidth=1.4, linestyle=(0, (5, 2))),
    "d-optimality": dict(color=viz.INK["secondary"], linewidth=1.1,
                         linestyle=(0, (6, 2, 1, 2))),
    "neyman": dict(color=viz.INK["secondary"], linewidth=1.1, linestyle=(0, (1, 2))),
    "maxent": dict(color=viz.INK["muted"], linewidth=1.1),
    "thompson": dict(color=viz.INK["muted"], linewidth=1.1, linestyle=(0, (2, 2, 1, 2))),
    "random": dict(color=viz.INK["muted"], linewidth=1.1, linestyle=(0, (2, 3))),
    "epig": dict(color=viz.INK["muted"], linewidth=1.1, linestyle=(0, (4, 2))),
    "predictive-variance": dict(color=viz.INK["muted"], linewidth=0.9,
                                linestyle=(0, (1, 3))),
    "epistemic": dict(color=viz.INK["muted"], linewidth=0.9, linestyle=(0, (3, 3))),
}
LABEL = {"eig": "EIG, GIG-Poisson (ours)", "eig@gamma-poisson": "EIG, gamma-Poisson",
         "eig@lognormal-poisson": "EIG, lognormal-Poisson", "uniform": "systematic",
         "d-optimality": "D-optimality", "neyman": "Neyman alloc.",
         "maxent": "max-entropy", "thompson": "Thompson", "random": "random",
         "epig": "EPIG", "predictive-variance": "total var.",
         "epistemic": "epistemic var."}


def load(name):
    return np.atleast_1d(np.genfromtxt(os.path.join(HERE, "results", name),
                                       delimiter=",", names=True, dtype=None,
                                       encoding="utf-8"))


def panel_policy(ax, seq):
    for name, st in STYLE.items():
        m = seq["policy"] == name
        if not np.any(m):
            continue
        spent = np.unique(seq["spent"][m])
        mean = np.array([np.mean(seq["nlpd"][m & (seq["spent"] == s)]) for s in spent])
        sem = np.array([np.std(seq["nlpd"][m & (seq["spent"] == s)], ddof=1)
                        / np.sqrt(np.sum(m & (seq["spent"] == s))) for s in spent])
        ax.plot(spent, mean, **st)
        if name in ("eig", "eig@gamma-poisson"):
            ax.fill_between(spent, mean - sem, mean + sem, color=st["color"],
                            alpha=0.18, linewidth=0)
    ax.set_xlabel("telescope time spent (Ms)")
    ax.set_ylabel("held-out NLPD (nats)")
    ax.set_title("(a) 48 targets, 120 Ms", loc="left")
    ax.legend(handles=[plt.Line2D([], [], **STYLE[k], label=LABEL[k])
                       for k in ("eig", "eig@gamma-poisson", "eig@lognormal-poisson",
                                 "uniform", "maxent")],
              loc="upper right", fontsize=6.5, handlelength=2.2, labelspacing=0.3)


def panel_allocation(ax, alloc, seq):
    """(b) Time given to a target against how uncertain its class prior was."""
    for name in ("eig", "uniform", "maxent"):
        m = alloc["policy"] == name
        g, v = alloc["flux_true"][m], alloc["time_allocated"][m]
        order = np.argsort(g)
        edges = np.quantile(g, np.linspace(0, 1, 9))
        idx = np.clip(np.digitize(g[order], edges[1:-1]), 0, 7)
        ax.plot([np.median(g[order][idx == i]) for i in range(8)],
                [np.mean(v[order][idx == i]) for i in range(8)],
                marker="o", markersize=3.2, markeredgewidth=0.0, **STYLE[name])
    ax.set_xscale("log")
    ax.set_yscale("log")
    # A target given no time at all plots at minus infinity on a log axis, which turned the
    # maximum-entropy trace into a spike to 1e-7 Ms. Floor the axis at the shortest
    # integration actually on offer and mark the bins that fall below it.
    floor = 0.5
    lo, hi = ax.get_ylim()
    ax.set_ylim(0.5 * floor, hi)
    ax.axhline(floor, color=viz.INK["grid"], linewidth=0.8, zorder=0)
    ax.annotate("shortest integration on offer", xy=(0.02, floor),
                xycoords=("axes fraction", "data"), xytext=(0, 3),
                textcoords="offset points", fontsize=6.5, color=viz.INK["secondary"])
    ax.set_xlabel(r"true flux ($10^{-10}$ cm$^{-2}$ s$^{-1}$)")
    ax.set_ylabel("telescope time given (Ms)")
    ax.set_title("(b) where the time went", loc="left")


def main():
    seq = load("sequential.csv")
    alloc = load("allocation.csv")
    # Panel (a) used to show the measured flux field against each fitted mixing law. That
    # is the first panel of Figure 5, so it is carried there instead of twice.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    panel_policy(axes[0], seq)
    panel_allocation(axes[1], alloc, seq)
    viz.savefig(fig, os.path.join(HERE, "figures", "photon.pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
