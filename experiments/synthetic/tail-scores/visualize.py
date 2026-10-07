"""Figure for the tail-score illustration.

Reads ``results/summary.csv``, writes ``figures/tail_scores.pdf``. Run after ``run.py``.

Panel (a) is the phase diagram: the censored-likelihood gain of the Sichel over the matched
negative binomial, per exceedance of the 99th percentile, against the true order and the
exposure already spent at the context. Contours are the lemma's gap in geometric tail
rates, ``log(q_F / q_Gamma,F)``, which depends on the exposure and not on the counts.
Panel (b) is one column of (a), the order at minus one ceiling, at three thresholds, with
a ``1/F^2`` guide: the rate gap is O(1/F), and a divergence is quadratic in it.

Run: python experiments/synthetic/tail-scores/visualize.py
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "common"))

import matplotlib.pyplot as plt                                       # noqa: E402
import matplotlib.patheffects as pe                                   # noqa: E402
from matplotlib.colors import LogNorm                                 # noqa: E402

import viz                                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
viz.apply_style()

PHASE_LEVEL = 0.99
SLICE_ORDER = -1.0
LEVEL_STYLE = {0.95: viz.CAT[0], 0.99: viz.CAT[1], 0.999: viz.CAT[2]}


def load(path):
    with open(path, newline="") as fh:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def grid(rows, field, level):
    fr = sorted({r["order_over_max"] for r in rows})
    ex = sorted({r["exposure"] for r in rows})
    z = np.full((len(ex), len(fr)), np.nan)
    for r in rows:
        if r["level"] == level:
            z[ex.index(r["exposure"]), fr.index(r["order_over_max"])] = r[field]
    return fr, ex, z


def main():
    rows = load(os.path.join(HERE, "results", "summary.csv"))
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.8),
                                     gridspec_kw={"width_ratios": [1.25, 1.0]})

    # (a) Phase diagram on an index grid, since neither axis is evenly spaced.
    fr, ex, gain = grid(rows, "cl_gain_per_exceedance", PHASE_LEVEL)
    _, _, gap = grid(rows, "log_rate_gap", PHASE_LEVEL)
    floor = 1e-4
    ax_a.grid(False)
    mesh = ax_a.pcolormesh(np.arange(len(fr) + 1) - 0.5, np.arange(len(ex) + 1) - 0.5,
                           np.clip(gain, floor, None), cmap=viz.SEQ,
                           norm=LogNorm(vmin=floor, vmax=np.nanmax(gain)),
                           edgecolors=viz.INK["surface"], linewidth=1.0)
    cs = ax_a.contour(np.arange(len(fr)), np.arange(len(ex)), gap,
                      levels=[0.003, 0.01, 0.03, 0.1], colors=viz.INK["primary"],
                      linewidths=0.8)
    halo = [pe.withStroke(linewidth=2.2, foreground=viz.INK["surface"])]
    plt.setp(cs.collections if hasattr(cs, "collections") else [cs], path_effects=halo)
    for txt in ax_a.clabel(cs, fmt="%g", fontsize=7):
        txt.set_path_effects(halo)
    ax_a.set_xticks(range(len(fr)))
    ax_a.set_xticklabels(["%g" % f for f in fr], rotation=90)
    ax_a.set_yticks(range(len(ex)))
    ax_a.set_yticklabels(["%g" % e for e in ex])
    ax_a.set_xlabel(r"true order / ceiling $\nu / r$")
    ax_a.set_ylabel(r"exposure spent $F$")
    ax_a.set_title("(a) gain per exceedance of the 99th pct.", loc="left")
    cb = fig.colorbar(mesh, ax=ax_a, pad=0.02)
    cb.ax.grid(False)
    cb.set_label("nats", color=viz.INK["secondary"])
    cb.outline.set_visible(False)

    # (b) One column, three thresholds, against exposure.
    for level, color in LEVEL_STYLE.items():
        sel = sorted((r for r in rows if r["level"] == level
                      and r["order_over_max"] == SLICE_ORDER and r["exposure"] > 0),
                     key=lambda r: r["exposure"])
        F = np.array([r["exposure"] for r in sel])
        g = np.array([r["cl_gain_per_exceedance"] for r in sel])
        ax_b.plot(F, g, "o-", color=color, label="%g pct." % (100 * level))
        viz.label_line(ax_b, F[-1], g[-1], "%g" % (100 * level), color, dx=6.0)
        if level == PHASE_LEVEL:
            guide = g[0] * F[0] ** 2 / F ** 2
            ax_b.plot(F, guide, "--", color=viz.INK["muted"], linewidth=1.0)
            ax_b.text(1.05, guide[0] / 60.0, r"dashed: $\propto 1/F^2$",
                      color=viz.INK["muted"], fontsize=8, ha="left", va="top")
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel(r"exposure spent $F$")
    ax_b.set_ylabel("gain per exceedance (nats)")
    ax_b.set_title(r"(b) at $\nu / r = %g$" % SLICE_ORDER, loc="left")
    ax_b.legend(loc="lower left")

    viz.savefig(fig, os.path.join(HERE, "figures", "tail_scores.pdf"))


if __name__ == "__main__":
    main()
