"""The manuscript's figure and table for the two measured-count studies.

    figures/natural_counts.pdf            top: observed cell means against the means each
                                          fitted law predicts, near Fukushima Daiichi and in
                                          Osaka; bottom: each rival's NLPD minus the GIG's on
                                          a cell never measured, against the catalogue size,
                                          on one scale for both regions
    experiments/tables/natural_counts.tex the GIG's NLPD and every rival's paired difference
                                          from it at 20 and 400 cells per band

Reads ``experiments/safecast-{fukushima,osaka}/results/``, written by each study's
``run.py``.

Run: python experiments/common/natural_paper.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import natural_figs as nf                                             # noqa: E402
import safecast_study                                                 # noqa: E402
import viz                                                            # noqa: E402

STUDIES = [("safecast-fukushima", "Fukushima Daiichi, 60 km, 2024-2026"),
           ("safecast-osaka", "Osaka, 30 km, natural background")]
FIGURE = os.path.join(ROOT, "figures", "natural_counts.pdf")
TABLE = os.path.join(ROOT, "experiments", "tables", "natural_counts.tex")
#: Catalogue sizes the table reports; the figure shows every size run.
TABLE_CATALOGUES = (20, 400)
#: Linear range of the symmetric-log axis: Osaka's differences are of order 1e-5 nats.
LINTHRESH = 1e-4
TABLE_LAWS = [("gamma", "gamma"), ("gammamix", "gamma mixture"), ("lognormal", "lognormal"),
              ("logskewnormal", "skew-normal on $\\log \\lambda$, penalised"),
              ("poisson", "Poisson, one rate per band")]
#: Laws the studies fit but the manuscript does not report.
OMITTED = ("compoundgamma",)


def _load(study):
    r = os.path.join(ROOT, "experiments", study, "results")
    return (pd.read_csv(os.path.join(r, "cells.csv")),
            pd.read_csv(os.path.join(r, "population.csv")),
            pd.read_csv(os.path.join(r, "summary.csv")))


def figure():
    viz.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.0), height_ratios=[1.0, 1.05])
    data = [_load(s) for s, _ in STUDIES]
    gaps = []
    for j, ((study, title), (cells, pop, summ)) in enumerate(zip(STUDIES, data)):
        ax = axes[0, j]
        nf.means_panel(ax, cells, pop, safecast_study.PER_MINUTE, markersize=2.0,
                       legend=False)
        ax.set_title("({}) {}".format("ab"[j], title), loc="left", fontsize=8.5)
        d = summ[(summ.score == "reading") & (summ.condition == 0) & (summ.law != "gig")
                 & ~summ.law.isin(OMITTED)]
        gaps.append(d)
    vals = np.concatenate([d["diff"].values for d in gaps])
    ylim = (-(10.0 ** np.ceil(np.log10(max(-vals.min(), LINTHRESH)))),
            10.0 ** np.ceil(np.log10(vals.max())))
    for j, d in enumerate(gaps):
        ax = axes[1, j]
        nf.gap_panel(ax, d, linthresh=LINTHRESH, ylim=ylim, markersize=3.8)
        ax.set_title("({}) a cell never measured, one 5 s reading".format("cd"[j]),
                     loc="left", fontsize=8.5)
        ax.set_xlabel("cells per band in the catalogue")
    axes[1, 0].set_ylabel("NLPD minus the GIG's (nats)")
    laws = ["gig", "gamma", "logskewnormal", "gammamix", "lognormal", "poisson"]
    handles = nf.legend_handles(laws)
    obs = plt.Line2D([], [], linestyle="none", marker="o", markersize=3.5,
                     color=viz.INK["primary"], alpha=0.55)
    names = ["GIG", "gamma", "skew-normal on log rate", "gamma mixture", "lognormal",
             "Poisson, one rate per band"]
    fig.legend([obs] + handles, ["observed cell means"] + names, loc="outside lower center",
               ncol=4, fontsize=7)
    os.makedirs(os.path.dirname(FIGURE), exist_ok=True)
    viz.savefig(fig, FIGURE)


def _fmt(v, sig):
    """A difference in millinats, to the precision its size needs."""
    a = abs(v)
    if a >= 1000:
        txt = "{:+,.0f}".format(v).replace(",", "{,}")
    elif a >= 10:
        txt = "{:+.0f}".format(v)
    elif a >= 1:
        txt = "{:+.1f}".format(v)
    elif a >= 0.1:
        txt = "{:+.2f}".format(v)
    else:
        txt = "{:+.3f}".format(v)
    return "${}{}$".format(txt, "^{*}" if sig else "")


def table():
    cols = [(s, sc, n) for s, _ in STUDIES for sc in ("reading", "integration")
            for n in TABLE_CATALOGUES]
    summ = {s: _load(s)[2] for s, _ in STUDIES}

    def row(law):
        out = []
        for s, sc, n in cols:
            d = summ[s]
            x = d[(d.law == law) & (d.score == sc) & (d.condition == 0) & (d.catalogue == n)]
            x = x.iloc[0]
            if law == "gig":
                out.append("${:.3f}$".format(x.nlpd))
            else:
                out.append(_fmt(1e3 * x["diff"], str(x.get("significant")) == "True"))
        return out

    k = len(TABLE_CATALOGUES)
    lines = ["% Generated by experiments/common/natural_paper.py. Do not edit by hand.",
             "\\begin{tabular}{l " + " ".join(["c" * (2 * k)] * len(STUDIES)) + "}",
             "    \\toprule",
             "    & " + " & ".join("\\multicolumn{{{}}}{{c}}{{{}}}".format(2 * k, t)
                                   for t in ("Fukushima Daiichi, 2024--2026",
                                             "Osaka, natural background")) + " \\\\",
             "    " + " ".join("\\cmidrule(lr){{{}-{}}}".format(2 + 2 * k * i, 1 + 2 * k * (i + 1))
                               for i in range(len(STUDIES))),
             "    & " + " & ".join("\\multicolumn{{{}}}{{c}}{{{}}}".format(k, t)
                                   for _ in STUDIES
                                   for t in ("one 5 s reading", "50 s sum")) + " \\\\",
             "    " + " ".join("\\cmidrule(lr){{{}-{}}}".format(2 + k * i, 1 + k * (i + 1))
                               for i in range(2 * len(STUDIES))),
             "    cells per band & " + " & ".join("${}$".format(n) for _, _, n in cols)
             + " \\\\",
             "    \\midrule",
             "    GIG, NLPD (nats) & " + " & ".join(row("gig")) + " \\\\",
             "    \\midrule"]
    for law, name in TABLE_LAWS:
        lines.append("    {} & ".format(name) + " & ".join(row(law)) + " \\\\")
    lines += ["    \\bottomrule", "\\end{tabular}", ""]
    os.makedirs(os.path.dirname(TABLE), exist_ok=True)
    with open(TABLE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("  wrote {}".format(TABLE))


if __name__ == "__main__":
    figure()
    table()
