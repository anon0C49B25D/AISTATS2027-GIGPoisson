"""The comparison figures the manuscript includes, written into the manuscript's ``figures/``.

    factorial_nlpd.pdf         the prior-by-policy factorial of Section 6: NLPD at the full
                               budget as the paired difference from the GIG under the EIG
    factorial_regret.pdf       the same factorial: area under the top-5 regret curve minus
                               that of the GIG under the EIG, one box per method over episodes
    budget.pdf                 NLPD against the budget spent, from the fitted priors alone:
                               the gamma mixture and the gamma, and the GIG under nested
                               estimates of the EIG, as paired differences from the GIG under
                               the EIG
    nlpd.pdf                  held-out NLPD at the full budget, one panel per setting: ours
                               and the four baselines of the main text
    regret.pdf                 top-1 and top-5 regret against budget spent, the same methods
    nlpd_acquisitions.pdf      the same NLPD panels for the allocation rules (appendix)
    regret_acquisitions.pdf    the same regret panels for the allocation rules (appendix)

Reads ``results/{bulk,gamma,hardxray}.csv`` and, for the factorial and the budget figure,
``results/fact_*.csv``, written by ``run.py``, and ``results/prior_only.csv``, written by
``prior_only.py``. The per-study figures, one set per group of methods, are drawn by each
study's ``visualize_comparison.py``.

Run: python experiments/comparison/paper_figures.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "common"))

import comparison_figs as cf                                          # noqa: E402
import natural_figs as nf                                             # noqa: E402
import tables                                                         # noqa: E402
import viz                                                            # noqa: E402

OUT = os.path.join(ROOT, "figures")

#: Tick and panel labels of the factorial figure, short enough for a column-width panel.
_POLICY_TICK = {"EIG": "EIG", "D-optimality": "D-opt.", "systematic": "syst."}
_SETTING_TITLE = {"bulk": "diamonds", "gamma": "$\\gamma$-ray survey", "hardxray": "hard X-ray"}


def _factorial_priors():
    """The rows of ``tables.FACTORIAL`` in blocks of three, one block per prior."""
    return [tables.FACTORIAL[k:k + 3] for k in range(0, len(tables.FACTORIAL), 3)]


def _factorial_metrics(setting, prefix="fact_"):
    """NLPD at the full budget and the area under the top-5 regret curve, per episode and
    agent, as ``tables.factorial`` computes them."""
    d = pd.read_csv(os.path.join(tables.RESULTS, prefix + setting + ".csv"))
    fin = d[np.isclose(d.spent, d.spent.max())]
    return (tables.wide(fin, "nlpd"),
            d.groupby(["episode", "agent"]).regret_top5.mean().unstack("agent"))


def _policy_axis(ax, rows, title):
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([_POLICY_TICK[r] for _, _, r in rows], fontsize=7)
    ax.set_title(title, loc="left", fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.tick_params(labelsize=7)


def factorial_nlpd(path, prefix="fact_"):
    """Column-width figure of the NLPD columns of the top block of ``tables.factorial``, one
    panel per setting.

    Every point is the mean paired difference from the GIG under the EIG over the episodes,
    with the half-width of a 95 per cent t-interval, so the figure and the table carry the
    same numbers. One line per prior joins its three policies, dodged so that the intervals
    do not overlap.
    """
    viz.apply_style()
    priors = _factorial_priors()
    fig, axes = plt.subplots(1, len(tables.SETTINGS), figsize=(3.3, 1.75), sharey=True)
    for ax, setting in zip(axes, tables.SETTINGS):
        w = _factorial_metrics(setting, prefix)[0]
        ax.axhline(0.0, color=viz.INK["secondary"], linewidth=0.8, zorder=1)
        for rows, dx in zip(priors, (-0.13, 0.0, 0.13)):
            st = cf.STYLE[rows[0][0]]
            x = np.arange(len(rows)) + dx
            m, h = np.array([(0.0, 0.0) if a == "eig" else tables._paired(w, a)
                             for a, _, _ in rows]).T
            ax.vlines(x, m - h, m + h, color=st["color"], linewidth=0.9, zorder=st["zorder"])
            ax.plot(x, m, color=st["color"], linestyle=st["linestyle"], linewidth=1.3,
                    marker=st["marker"], markersize=4, markeredgecolor=viz.INK["surface"],
                    markeredgewidth=0.6, zorder=st["zorder"], label=rows[0][1])
        _policy_axis(ax, priors[0], _SETTING_TITLE[setting])
        ax.set_xlim(-0.4, 2.4)
    axes[0].set_ylabel("$\\Delta$ NLPD (nats)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=7)
    viz.savefig(fig, path)


def factorial_regret(path, prefix="fact_"):
    """Column-width figure of the regret columns of the top block of ``tables.factorial``,
    one panel per setting.

    One box per method over the episodes, of its area under the top-5 regret curve minus
    that of the GIG under the EIG in the same episode. The GIG under the EIG is the line at
    zero. Boxes span the quartiles, whiskers reach 1.5 times the interquartile range, and
    dots mark the episodes beyond them.
    """
    viz.apply_style()
    priors = _factorial_priors()
    fig, axes = plt.subplots(1, len(tables.SETTINGS), figsize=(3.3, 1.9), sharey=True)
    for ax, setting in zip(axes, tables.SETTINGS):
        auc = _factorial_metrics(setting, prefix)[1]
        ax.axhline(0.0, color=viz.CAT[0], linewidth=1.2, zorder=1)
        for rows, dx in zip(priors, (-0.27, 0.0, 0.27)):
            colour = cf.STYLE[rows[0][0]]["color"]
            boxes = [(k + dx, (auc[a] - auc["eig"]).dropna().values)
                     for k, (a, _, _) in enumerate(rows) if a != "eig"]
            bp = ax.boxplot([b for _, b in boxes], positions=[x for x, _ in boxes],
                            widths=0.22, patch_artist=True, manage_ticks=False,
                            medianprops=dict(color=viz.INK["primary"], linewidth=1.0),
                            whiskerprops=dict(color=colour, linewidth=0.8),
                            capprops=dict(color=colour, linewidth=0.8),
                            flierprops=dict(marker="o", markersize=1.6, markeredgewidth=0,
                                            markerfacecolor=colour, alpha=0.6))
            for box in bp["boxes"]:
                box.set(facecolor=to_rgba(colour, 0.25), edgecolor=colour, linewidth=0.9)
        _policy_axis(ax, priors[0], _SETTING_TITLE[setting])
        ax.set_xlim(-0.42, 2.42)
    axes[0].set_ylabel("$\\Delta$ top-5 regret area", fontsize=8)
    colours = [cf.STYLE[rows[0][0]]["color"] for rows in priors]
    handles = [Patch(facecolor=to_rgba(c, 0.25), edgecolor=c) for c in colours]
    handles.append(plt.Line2D([], [], color=viz.CAT[0], linewidth=1.2))
    fig.legend(handles, [rows[0][1] for rows in priors] + ["GIG, EIG (reference)"],
               loc="outside upper center", ncol=2, fontsize=7)
    viz.savefig(fig, path)


#: The two rows of the budget figure, as (agent, prior-only model, label, style). The top row
#: sets the other two conjugate priors under the EIG against the GIG, in the styles of the
#: measured-count figures. The bottom row sets the GIG under nested estimates of the EIG
#: against the deterministic EIG; these share the GIG prior, so they have no prior-only model.
_BUDGET_ROWS = (
    (("eig@gammamix-poisson", "gammamix-poisson", "gamma mixture", nf.STYLE["gammamix"]),
     ("eig@gamma-poisson", "gamma-poisson", "gamma", nf.STYLE["gamma"])),
    (("eig@gig-pce", None, "GIG, EIG by PCE",
      dict(color=viz.CAT[0], marker="v", linestyle=(0, (4, 1.5)), zorder=5)),
     ("eig@gig-nmc", None, "GIG, EIG by nested MC",
      dict(color=viz.CAT[0], marker="x", linestyle=(0, (1, 1.2)), zorder=5))),
)


def _against_budget(d, po, agent, model):
    """Mean paired NLPD difference of ``agent`` from the GIG under the EIG, and the half-width
    of its 95 per cent interval, at zero budget and at every checkpoint, against the fraction
    of the budget spent. At zero budget the score is that of the fitted prior alone, from
    ``prior_only.csv``, which pairs with the same episodes; a ``model`` of None shares the
    GIG prior, so its difference there is zero.
    """
    x, m, h = [0.0], [0.0], [0.0]
    if model is not None:
        w = po.pivot(index="episode", columns="model", values="nlpd")
        m[0], h[0] = tables._paired(w, model, ref="gig-poisson")
    for spent in sorted(d.spent.unique()):
        mean, half = tables._paired(tables.wide(d[np.isclose(d.spent, spent)], "nlpd"), agent)
        x.append(spent / d.spent.max())
        m.append(mean)
        h.append(half)
    return np.array(x), np.array(m), np.array(h)


def budget(path, prefix="fact_"):
    """Column-width figure: NLPD against the fraction of the budget spent, as the paired
    difference from the GIG under the EIG, one column per setting, shaded 95 per cent
    intervals. The top row holds the gamma mixture and the gamma under the EIG and the bottom
    row the GIG under nested estimates of the EIG. At zero budget the score is the fitted
    prior's.
    """
    viz.apply_style()
    po = pd.read_csv(os.path.join(tables.RESULTS, "prior_only.csv"))
    fig, axes = plt.subplots(2, len(tables.SETTINGS), figsize=(3.3, 3.1), sharex=True,
                             sharey="row")
    for j, setting in enumerate(tables.SETTINGS):
        d = pd.read_csv(os.path.join(tables.RESULTS, prefix + setting + ".csv"))
        p = po[po.setting == setting]
        p = p[np.isclose(p["eval"], p["eval"].median())]
        for i, rows in enumerate(_BUDGET_ROWS):
            ax = axes[i, j]
            ax.axhline(0.0, color=viz.CAT[0], linewidth=1.4, zorder=2,
                       label="GIG, deterministic EIG")
            for agent, model, label, st in rows:
                x, m, h = _against_budget(d, p, agent, model)
                ax.fill_between(x, m - h, m + h, color=st["color"], alpha=0.14, linewidth=0,
                                zorder=st["zorder"] - 1)
                ax.plot(x, m, color=st["color"], linestyle=st["linestyle"], linewidth=1.3,
                        marker=st["marker"], markersize=3.2, zorder=st["zorder"], label=label)
            ax.grid(axis="x", visible=False)
            ax.tick_params(labelsize=7)
        axes[0, j].set_title(_SETTING_TITLE[setting], loc="left", fontsize=8)
        axes[1, j].set_xticks([0.0, 0.5, 1.0])
        axes[1, j].set_xticklabels(["0", "0.5", "1"])
    for ax in axes[:, 0]:
        ax.set_ylabel("$\\Delta$ NLPD (nats)", fontsize=8)
    fig.supxlabel("fraction of budget spent", fontsize=8, color=viz.INK["secondary"])
    top, top_labels = axes[0, 0].get_legend_handles_labels()
    bottom, bottom_labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(top + bottom[1:], top_labels + bottom_labels[1:], loc="outside upper center",
               ncol=2, fontsize=7)
    viz.savefig(fig, path)


def main():
    os.makedirs(OUT, exist_ok=True)
    factorial_nlpd(os.path.join(OUT, "factorial_nlpd.pdf"))
    factorial_regret(os.path.join(OUT, "factorial_regret.pdf"))
    budget(os.path.join(OUT, "budget.pdf"))
    cf.paper_nlpd_grid(os.path.join(OUT, "nlpd.pdf"), cf.PAPER_METHODS,
                       paired=("eig@gamma-poisson", "eig@gammamix-poisson"))
    cf.paper_regret_grid(os.path.join(OUT, "regret.pdf"), cf.PAPER_METHODS, ms=(1, 5))
    cf.paper_nlpd_grid(os.path.join(OUT, "nlpd_acquisitions.pdf"), cf.PAPER_ACQUISITIONS,
                       height=2.3)
    cf.paper_regret_grid(os.path.join(OUT, "regret_acquisitions.pdf"),
                         cf.PAPER_ACQUISITIONS, ms=(1, 5))


if __name__ == "__main__":
    main()
