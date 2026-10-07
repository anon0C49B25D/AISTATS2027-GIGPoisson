"""Figures for the GIG-Poisson data-generating process.

Reads ``data/`` and ``results/``, writes ``figures/``. Run after ``run.py``.

Four panels, one per property the process has to have: that the mixing law has a free
dispersion knob, that the analytic predictive is the law the simulator actually draws
from, that the overdispersion is a function of the action, and that the Bessel functions
have to be evaluated in log space.

Run: python experiments/synthetic/verification/visualize.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "common"))

import matplotlib.pyplot as plt                                       # noqa: E402

import viz                                                            # noqa: E402
from dgp import gig_from_mean, gig_logpdf, gig_moments                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
viz.apply_style()


def load(name):
    return np.atleast_1d(np.genfromtxt(os.path.join(HERE, name), delimiter=",",
                                       names=True, dtype=None, encoding="utf-8"))


def panel_mixing(ax):
    """(a) What the concentration does to the mixing law at a fixed mean."""
    lam = np.geomspace(1e-3, 60.0, 2000)
    for c, omega in zip(viz.CAT, (2.0, 0.5, 0.05)):
        prior = gig_from_mean(-0.5, 4.0, omega)
        m, v = gig_moments(prior)
        ax.plot(lam, np.exp(gig_logpdf(lam, prior)), color=c,
                label=r"$\omega={:g}$,  $\mathbb{{V}}/\mathbb{{E}}={:.1f}$"
                      .format(omega, v / m))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 5.0)
    ax.set_xlabel(r"rate $\lambda$")
    ax.set_ylabel("mixing density")
    ax.set_title(r"(a) dispersion knob at fixed mean $\mathbb{E}[\lambda]=4$", loc="left")
    ax.legend(loc="lower left")


def panel_predictive(ax, pmf, hist, site):
    """(b) The analytic Sichel law against the process, run end to end.

    The dots are the two-stage process, a fresh rate from the mixing law for every draw.
    That is the law the predictive claims to be, and the claim of correct specification is
    exactly that these two objects coincide.
    """
    dwells = np.unique(pmf["dwell"][pmf["site"] == site])
    chosen = [dwells[0], dwells[len(dwells) // 2], dwells[-1]]
    for c, t in zip(viz.CAT, chosen):
        m = (pmf["site"] == site) & (pmf["dwell"] == t)
        h = (hist["site"] == site) & (hist["dwell"] == t)
        ax.plot(hist["y"][h], hist["count"][h] / hist["n_draws"][h], "o", color=c,
                markersize=2.4, alpha=0.5, markeredgewidth=0.0)
        ax.plot(pmf["y"][m], np.exp(pmf["log_pmf"][m]), color=c, linewidth=1.4,
                label=r"$t={:.3g}$".format(t))
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 2.0)
    ax.set_xlabel("count $y$")
    ax.set_ylabel("probability")
    ax.set_title("(b) predictive law (lines) and the process (dots)", loc="left")
    ax.legend(loc="upper right", title="dwell", title_fontsize=8)


def panel_fano(ax, disp, env):
    """(c) Overdispersion as a function of the action, not a fixed property of the model.

    Lines are the analytic marginal Fano factor, dots the same quantity estimated from the
    process. The flat cloud along the bottom is the Fano factor of the survey trace, drawn
    at the rate each site was realised at: conditional on the rate the counts are Poisson,
    so the spread the sensor has to plan around is invisible to anyone who has already
    conditioned on a point estimate of it.
    """
    order = np.argsort(env["prior_var_over_mean"])
    norm = plt.Normalize(0.0, 1.0)
    for rank, k in enumerate(env["site"][order]):
        m = disp["site"] == k
        col = viz.SEQ(norm(0.15 + 0.8 * rank / max(len(order) - 1, 1)))
        ax.plot(disp["exposure"][m], disp["fano_analytic"][m], color=col, linewidth=1.4)
        ax.plot(disp["exposure"][m], disp["fano_marginal"][m], "o", color=col,
                markersize=2.6, markeredgewidth=0.0, alpha=0.75)
    ax.plot(disp["exposure"], disp["fano_conditional"], "x", color=viz.CAT[1],
            markersize=3.0, markeredgewidth=0.9, alpha=0.7)
    ax.axhline(1.0, color=viz.INK["secondary"], linewidth=1.0, linestyle=(0, (4, 3)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Leave room below the Poisson line so the conditional cloud and its label separate.
    ax.set_ylim(bottom=0.30)
    ax.set_xlabel(r"exposure $f(u)$")
    ax.set_ylabel(r"Fano factor $\mathbb{V}[y]/\mathbb{E}[y]$")
    ax.set_title("(c) overdispersion is set by the action", loc="left")
    ax.annotate("marginal, one line per site\nshaded by prior dispersion",
                xy=(0.03, 0.96), xycoords="axes fraction", fontsize=8,
                color=viz.INK["secondary"], va="top")
    ax.annotate("conditional on the realised rate", xy=(0.97, 0.02),
                xycoords="axes fraction", fontsize=8, color=viz.CAT[1],
                fontweight="bold", va="bottom", ha="right")


def panel_order(ax, growth):
    """(d) The posterior order outruns double precision after a handful of counts."""
    finite = growth["unlogged_finite"].astype(bool)
    ax.plot(growth["round"], growth["log_besselk"], color=viz.CAT[0],
            label=r"$\log K_\alpha(z)$ accumulated in log space")
    with np.errstate(divide="ignore", invalid="ignore"):
        naive = np.where(finite, np.log(growth["unlogged_kv"]), np.nan)
    ax.plot(growth["round"], naive, color=viz.CAT[1], linewidth=3.0,
            label=r"$\log K_\alpha(z)$ via a call to $K_\alpha(z)$")
    first = int(growth["round"][~finite][0]) if np.any(~finite) else None
    if first is not None:
        ax.axvline(first, color=viz.CAT[1], linewidth=1.0, linestyle=(0, (4, 3)))
        ax.annotate("overflow at round {},\norder $\\alpha = {:.0f}$".format(
            first, float(growth["order"][growth["round"] == first][0])),
            xy=(first, 0.62), xycoords=("data", "axes fraction"), xytext=(8, 0),
            textcoords="offset points", fontsize=8, color=viz.CAT[1],
            fontweight="bold", va="center")
    ax.set_xlabel("observation round")
    ax.set_ylabel("nats")
    ax.set_title("(d) the closed form is not an evaluation recipe", loc="left")
    ax.legend(loc="upper left")


def main():
    fd = os.path.join(HERE, "figures")
    env = load("data/environments.csv")
    hist = load("data/marginal_histogram.csv")
    disp = load("results/overdispersion.csv")
    pmf = load("results/predictive_pmf.csv")
    growth = load("results/order_growth.csv")

    site = int(pmf["site"][np.argmax(pmf["exposure"])])
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.6))
    panel_mixing(axes[0, 0])
    panel_predictive(axes[0, 1], pmf, hist, site)
    panel_fano(axes[1, 0], disp, env)
    panel_order(axes[1, 1], growth)
    viz.savefig(fig, os.path.join(fd, "dgp.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
