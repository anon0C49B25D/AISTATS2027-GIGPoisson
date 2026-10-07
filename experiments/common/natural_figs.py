"""Figures for the measured-count studies (``experiments/safecast-*``).

``data.pdf``    (a) the cell rates on a map, (b) the tail of the cell rates against each
                fitted mixing law, (c) within-cell variance against mean, the Poisson check.
``scores.pdf``  each rival's held-out surprisal minus the GIG's, against the catalogue size
                the class priors were fitted to; above zero, the GIG predicted better.

Encoding follows ``comparison_figs.py``: colour goes to the GIG and the two rivals the paper
is about (slot order fixed: GIG blue, gamma orange, log-skew-normal aqua), every other law is
a neutral ink told apart by marker and dash, and every line is labelled on the figure.
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import countfit
import viz

__all__ = ["STYLE", "LABEL", "ORDER", "means_panel", "gap_panel", "legend_handles",
           "fig_data", "fig_scores"]


def _direct_labels(ax, ends, gap_pt=8.5):
    """Label line ends at the right edge, pushed apart to at least ``gap_pt`` points."""
    if not ends:
        return
    fig = ax.figure
    fig.canvas.draw()
    to_disp = ax.transData.transform
    from_disp = ax.transData.inverted().transform
    x1 = ax.get_xlim()[1]
    ys = sorted((to_disp((x1, y))[1], law) for y, law in ends)
    gap = gap_pt * fig.dpi / 72.0
    pos = [ys[0][0]]
    for y, _ in ys[1:]:
        pos.append(max(y, pos[-1] + gap))
    # Centre the block on the line ends again if pushing moved it up as a whole.
    shift = (np.mean([y for y, _ in ys]) - np.mean(pos))
    pos = [p + shift for p in pos]
    for (y, law), p in zip(ys, pos):
        yd = from_disp((0.0, p))[1]
        ax.annotate(LABEL[law], xy=(1.0, yd), xycoords=ax.get_yaxis_transform(),
                    xytext=(4, 0), textcoords="offset points", fontsize=7,
                    color=viz.INK["secondary"], va="center", annotation_clip=False)

LABEL = {"gig": "GIG", "gamma": "gamma", "logskewnormal": "log-skew-normal",
         "gammamix": "gamma mixture", "lognormal": "lognormal",
         "compoundgamma": "compound gamma", "poisson": "Poisson (no mixing)"}


def _ink(colour, marker, dash, width=1.5, z=3):
    return dict(color=colour, marker=marker, linestyle=dash, linewidth=width, zorder=z)


STYLE = {
    "gig": dict(color=viz.CAT[0], marker="o", linestyle="-", linewidth=2.2, zorder=6),
    "gamma": dict(color=viz.CAT[1], marker="s", linestyle="-", linewidth=1.8, zorder=5),
    "logskewnormal": dict(color=viz.CAT[2], marker="D", linestyle="-", linewidth=1.8,
                          zorder=5),
    "gammamix": _ink(viz.INK["primary"], "X", (0, (5, 2)), 1.6, 4),
    "lognormal": _ink(viz.INK["secondary"], "v", (0, (5, 2))),
    "compoundgamma": _ink(viz.INK["secondary"], "^", (0, (1, 1.5))),
    "poisson": _ink(viz.INK["muted"], "x", (0, (2, 2))),
}
ORDER = ["gamma", "logskewnormal", "gammamix", "lognormal", "compoundgamma", "poisson"]


def _load(here):
    r = os.path.join(here, "results")
    info = json.load(open(os.path.join(r, "info.json")))
    return (info, pd.read_csv(os.path.join(r, "cells.csv")),
            pd.read_csv(os.path.join(r, "population.csv")),
            pd.read_csv(os.path.join(r, "summary.csv")))


# --------------------------------------------------------------------------
# Data figure
# --------------------------------------------------------------------------

def _draw_rates(law, th, n, rng):
    """``n`` rates from a fitted mixing law (for the predictive check of figure (b))."""
    from scipy.stats import geninvgauss, skewnorm
    th = np.asarray(th, float)
    if law == "poisson":
        return np.full(n, np.exp(th[0]))
    if law == "gamma":
        return rng.gamma(np.exp(th[0]), np.exp(-th[1]), n)
    if law == "gammamix":
        lw, r, be = countfit._mix_unpack(th)
        w = np.exp(lw)
        j = rng.choice(r.size, n, p=w / w.sum())
        return rng.gamma(r[j], 1.0 / be[j])
    if law == "lognormal":
        return np.exp(rng.normal(th[0], np.exp(th[1]), n))
    if law == "logskewnormal":
        return np.exp(skewnorm.rvs(th[2], loc=th[0], scale=np.exp(th[1]), size=n,
                                   random_state=rng))
    if law == "compoundgamma":
        r, A, B = countfit._cg_unpack(th)
        return B * rng.gamma(r, 1.0, n) / rng.gamma(A, 1.0, n)
    nu, a, b = countfit.gig_params(th)
    om, eta = np.sqrt(a * b), np.sqrt(a / b)
    # At the edges of the family the generic sampler is slow or fails; use the limit laws.
    if om < 1e-3 and nu < 0:
        return 0.5 * a / rng.gamma(-nu, 1.0, n)            # inverse gamma, b -> 0
    if om < 1e-3 and nu > 0:
        return rng.gamma(nu, 2.0 / b, n)                   # gamma, a -> 0
    return geninvgauss.rvs(nu, om, scale=eta, size=n, random_state=rng)


def means_panel(ax, cells, pop, per_minute, fontsize=7, markersize=2.6, legend=True):
    """Observed cell means against the cell means each fitted law predicts.

    A cell's mean count carries Poisson noise of relative size ``1 / sqrt(count)``, which on
    natural background is larger than the spread of the rates themselves, so the laws are
    compared with the observations where both carry it: for every cell, a rate is drawn
    from the law fitted to its class and counts at the cell's own exposure (ten draws per
    cell).
    """
    obs = np.sort(cells.rate_cpm.values)
    n_obs = obs.size
    ax.plot(obs, 1.0 - np.arange(n_obs) / n_obs, linestyle="none", marker="o",
            markersize=markersize, color=viz.INK["primary"], alpha=0.55, zorder=7,
            label="observed, all {} cells".format(n_obs))
    labels = sorted(pop.cls.unique(), key=lambda c: float(c.split("-")[0]))
    F = cells.readings.values.astype(float)
    rng = np.random.default_rng(0)
    for law in ("gamma", "lognormal", "gammamix", "logskewnormal", "gig"):
        sim = []
        for k, lab in enumerate(labels):
            th = json.loads(pop[(pop.cls == lab) & (pop.law == law)].params.iloc[0])
            idx = np.flatnonzero(cells.cls.values == k)
            for _ in range(10):
                lam = _draw_rates(law, th, idx.size, rng)
                sim.append(rng.poisson(F[idx] * lam) / F[idx] * per_minute)
        sim = np.sort(np.concatenate(sim))
        st = STYLE[law]
        lab_ = LABEL[law] + (" (= compound gamma)" if law == "gig" else "")
        ax.plot(sim, 1.0 - np.arange(sim.size) / sim.size, color=st["color"],
                linestyle=st["linestyle"], linewidth=st["linewidth"], zorder=st["zorder"],
                label=lab_)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(obs[0] * 0.8, obs[-1] * 1.5)
    ax.set_ylim(0.5 / n_obs, 1.2)
    ax.set_xlabel("cell mean, counts per minute")
    ax.set_ylabel("fraction of cells above")
    if legend:
        ax.legend(loc="best", fontsize=fontsize, frameon=True, facecolor=viz.INK["surface"],
                  edgecolor="none", framealpha=0.92)


def gap_panel(ax, d, offscale=(), linthresh=0.01, ylim=None, markersize=4.5, band=True):
    """Each rival's NLPD minus the GIG's against the catalogue size, symlog y.

    ``d`` is the rows of ``summary.csv`` for one score and one condition. Returns the line
    ends, for direct labels. Without ``ylim`` the limits are the next powers of ten, so the
    outermost decade carries a tick label.
    """
    ax.axhline(0.0, color=STYLE["gig"]["color"], linewidth=1.6, zorder=2)
    ends = []
    for law in ORDER:
        if law in offscale:
            continue
        g = d[d.law == law].sort_values("catalogue")
        if g.empty:
            continue
        st = STYLE[law]
        if band:
            ax.fill_between(g.catalogue, g["diff"] - g.diff_se, g["diff"] + g.diff_se,
                            color=st["color"], alpha=0.13, linewidth=0, zorder=1)
        ax.plot(g.catalogue, g["diff"], color=st["color"], linestyle=st["linestyle"],
                linewidth=st["linewidth"], marker=st["marker"], markersize=markersize,
                zorder=st["zorder"], markeredgecolor=viz.INK["surface"], markeredgewidth=0.6)
        ends.append((float(g["diff"].iloc[-1]), law))
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=linthresh, linscale=1.0)
    if ylim is None:
        vals = np.concatenate([[0.0]] + [d[d.law == l]["diff"].values for l in ORDER
                                         if l not in offscale])
        top, bot = float(vals.max()), float(vals.min())
        hi = 10.0 ** np.ceil(np.log10(top)) if top > linthresh else 1.5 * linthresh
        lo = -(10.0 ** np.ceil(np.log10(-bot))) if bot < -linthresh else -1.5 * linthresh
        ylim = (lo, hi)
    ax.set_ylim(*ylim)
    cats = sorted(d.catalogue.unique())
    ax.set_xticks(cats)
    ax.set_xticklabels([str(int(v)) for v in cats])
    ax.minorticks_off()
    return ends


def legend_handles(laws):
    return [plt.Line2D([], [], color=STYLE[l]["color"], linestyle=STYLE[l]["linestyle"],
                       linewidth=STYLE[l]["linewidth"], marker=STYLE[l]["marker"],
                       markersize=4.5) for l in laws]


def fig_data(here, title, per_minute, out="data.pdf"):
    viz.apply_style()
    info, cells, pop, _ = _load(here)
    cfg = info["config"]
    lat0, lon0 = cfg["centre"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5),
                             gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.0]))

    # (a) map, in km east and north of the centre.
    ax = axes[0]
    x = (cells.lon - lon0) * 111.0 * np.cos(np.radians(lat0))
    y = (cells.lat - lat0) * 111.0
    order = np.argsort(cells.rate_cpm.values)
    sc = ax.scatter(x.values[order], y.values[order], c=cells.rate_cpm.values[order],
                    cmap=viz.SEQ, norm=LogNorm(vmin=max(cells.rate_cpm.min(), 10.0),
                                               vmax=max(cells.rate_cpm.max(), 100.0)),
                    s=4, marker="s", linewidths=0)
    for r_ in cfg["bands"][1:]:
        t = np.linspace(0, 2 * np.pi, 400)
        ax.plot(r_ * np.cos(t), r_ * np.sin(t), color=viz.INK["axis"], linewidth=0.7,
                zorder=1)
    ax.plot(0, 0, marker="+", color=viz.INK["primary"], markersize=9, mew=1.5)
    ax.set_aspect("equal")
    lim = cfg["radius_km"] * 1.02
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("km east of centre")
    ax.set_ylabel("km north of centre")
    ax.set_title("(a) cell rates", loc="left")
    ax.grid(False)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("counts per minute", color=viz.INK["secondary"])
    cb.outline.set_visible(False)

    means_panel(axes[1], cells, pop, per_minute)
    axes[1].set_title("(b) cell means, observed and predicted", loc="left")

    # (c) within-cell dispersion.
    ax = axes[2]
    disp = pd.read_csv(os.path.join(here, "results", "dispersion.csv"))
    ax.scatter(disp["mean"], disp["var"], s=4, color=viz.CAT[0], alpha=0.35, linewidths=0,
               zorder=3)
    lo, hi = disp["mean"].min() * 0.8, disp["mean"].max() * 1.25
    ax.plot([lo, hi], [lo, hi], color=viz.INK["primary"], linewidth=1.2, zorder=4)
    ax.text(hi, hi, "  variance = mean\n  (Poisson)", fontsize=7.5,
            color=viz.INK["secondary"], va="center", ha="left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean of a cell's 5 s counts")
    ax.set_ylabel("variance of a cell's 5 s counts")
    ax.set_title("(c) within a cell: Poisson? ({} cells, >= 10 readings)".format(
        len(disp)), loc="left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=viz.INK["primary"])
    viz.savefig(fig, os.path.join(here, "figures", out))


# --------------------------------------------------------------------------
# Score figure
# --------------------------------------------------------------------------

def fig_scores(here, title, out="scores.pdf", conditions=(0, 5), offscale=None,
               linthresh=0.01):
    """Rival minus GIG, mean over episodes with one standard error, symlog y.

    ``offscale`` names rivals left out because their gaps are of another order (their range
    is stated under the figure instead), so the scale stays readable for the rest.
    """
    viz.apply_style()
    info, _, _, summ = _load(here)
    offscale = offscale or []
    summ = summ[summ.law != "gig"]
    rows, cols = ("reading", "integration"), conditions
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.3 * len(cols) + 1.2, 5.6),
                             sharex=True, squeeze=False)
    cond_title = {0: "new cell (nothing observed)", 1: "after 1 reading",
                  2: "after 2 readings", 5: "after 5 readings"}
    score_title = {"reading": "one 5 s reading", "integration": "sum of 10 readings (50 s)"}
    for i, s in enumerate(rows):
        for j, c in enumerate(cols):
            ax = axes[i, j]
            d = summ[(summ.score == s) & (summ.condition == c)]
            ends = gap_panel(ax, d, offscale, linthresh)
            if i == 0:
                ax.set_title(cond_title.get(c, "after {} readings".format(c)), loc="left")
            if j == 0:
                ax.set_ylabel("{}\nrival minus GIG, nats".format(score_title[s]))
            if i == len(rows) - 1:
                ax.set_xlabel("catalogue: cells per class the priors are fitted to")
            if j == len(cols) - 1:
                _direct_labels(ax, ends)
    drawn = ["gig"] + [l for l in ORDER if l not in offscale]
    handles = legend_handles(drawn)
    names = ["GIG (zero line)"] + [LABEL[l] for l in drawn[1:]]
    for law in offscale:
        # A law off the scale stays in the legend, with its range in place of a line.
        g = summ[(summ.law == law) & summ.condition.isin(cols)]
        h = legend_handles([law])[0]
        h.set_linestyle("none")
        handles.append(h)
        names.append("{}: {:.2g} to {:.2g} nats behind, off scale".format(
            LABEL[law], g["diff"].min(), g["diff"].max()))
    fig.legend(handles, names, loc="outside lower center", ncol=4, fontsize=7.5)
    fig.suptitle(title + "  (above zero: the GIG predicted better)", x=0.01, ha="left",
                 fontsize=10, color=viz.INK["primary"])
    viz.savefig(fig, os.path.join(here, "figures", out))
