"""Figures for the prediction-error study.

Reads ``results/prediction_error.csv``, writes ``figures/prediction.pdf``. Run after
``run_prediction.py``.

Three panels, each testing one prediction of the theory rather than merely displaying a
curve: that prediction error falls as ``1/n``, that the rate error falls as ``1/sqrt(n)``,
and that the constant in the posterior variance is the true rate, so that the rescaled
variance collapses onto one across every regime.

Run: python experiments/synthetic/verification/visualize_prediction.py
"""

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

#: Three dispersion levels share the categorical slots; the fourth regime varies the mean
#: instead of the tail, so it is a facet rather than a fourth hue.
STYLE = {
    "light":    dict(color=viz.CAT[0], linestyle="-"),
    "moderate": dict(color=viz.CAT[1], linestyle="-"),
    "heavy":    dict(color=viz.CAT[2], linestyle="-"),
    "sparse":   dict(color=viz.INK["secondary"], linestyle=(0, (4, 2))),
}


def load():
    return np.atleast_1d(np.genfromtxt(
        os.path.join(HERE, "results", "prediction_error.csv"), delimiter=",",
        names=True, dtype=None, encoding="utf-8"))


def guide(ax, n, anchor, slope, text):
    """A reference power law, drawn behind the data and labelled at its right end."""
    ref = anchor * (n / n[0]) ** slope
    ax.plot(n, ref, color=viz.INK["muted"], linewidth=0.9, linestyle=(0, (2, 2)),
            zorder=0)
    ax.annotate(text, xy=(n[-1], ref[-1]), xytext=(3, -2), textcoords="offset points",
                fontsize=8, color=viz.INK["muted"], va="top")


def panel(ax, d, column, title, ylabel, sem_column=None):
    for name, st in STYLE.items():
        m = d["regime"] == name
        n, v = d["n"][m], d[column][m]
        ax.plot(n, v, marker="o", markersize=3.0, markeredgewidth=0.0, **st)
        if sem_column is not None:
            s = d[sem_column][m]
            ax.fill_between(n, v - 2 * s, v + 2 * s, color=st["color"], alpha=0.18,
                            linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("sample size $n$  (accumulated exposure $F_n$)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")


def main():
    d = load()
    n = np.unique(d["n"]).astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.1))

    ax = axes[0]
    panel(ax, d, "kl_mean", "(a) prediction error", "KL to the oracle predictive (nats)",
          sem_column="kl_sem")
    ax.set_yscale("log")
    guide(ax, n, 0.30, -1.0, r"$1/n$")
    ax.legend(handles=[plt.Line2D([], [], **st, marker="o", markersize=3.0,
                                  markeredgewidth=0.0, label=k)
                       for k, st in STYLE.items()], loc="lower left", fontsize=7.5)

    ax = axes[1]
    panel(ax, d, "rate_error_mean", "(b) rate error",
          r"$|\mathbb{E}[\lambda \,|\, \mathcal{D}_n] - \lambda^*|$",
          sem_column="rate_error_sem")
    ax.set_yscale("log")
    guide(ax, n, 1.5, -0.5, r"$1/\sqrt{n}$")

    ax = axes[2]
    for name, st in STYLE.items():
        m = d["regime"] == name
        ax.plot(d["n"][m], d["n"][m] * d["posterior_var_mean"][m] / d["prior_mean"][m],
                marker="o", markersize=3.0, markeredgewidth=0.0, **st)
    ax.axhline(1.0, color=viz.INK["secondary"], linewidth=1.0, linestyle=(0, (4, 3)))
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.6)
    ax.set_xlabel("sample size $n$  (accumulated exposure $F_n$)")
    ax.set_ylabel(r"$F_n \, \mathbb{V}[\lambda \,|\, \mathcal{D}_n] \,/\, \mathbb{E}[\lambda^*]$")
    ax.set_title("(c) the constant in the rate", loc="left")
    ax.annotate("Proposition 2", xy=(n[1], 1.0), xytext=(2, 5),
                textcoords="offset points", fontsize=8, color=viz.INK["secondary"])

    viz.savefig(fig, os.path.join(HERE, "figures", "prediction.pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
