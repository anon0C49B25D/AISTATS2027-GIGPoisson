"""Figure for the EIG accuracy study.

Reads ``results/eig_accuracy.csv``, writes ``figures/eig_accuracy.pdf``. Run after ``run.py``.

One panel: the error of the shipped EIG against the reference, against the predictive mass
the shipped support sum leaves outside, one point per setting, class, belief and action.
The dashed line is the tail tolerance of 1e-9. Points to its right are the evaluations
where the support cap of 2e6 counts binds before the tolerance is met.

Run: python experiments/eig-accuracy/visualize.py
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

import matplotlib.pyplot as plt                                       # noqa: E402

import viz                                                            # noqa: E402

viz.apply_style()

LABELS = {"bulk": "bulk sampling", "gamma": "gamma-ray survey", "hardxray": "hard X-ray"}
FLOOR = 1e-16


def main():
    with open(os.path.join(HERE, "results", "eig_accuracy.csv")) as fh:
        rows = list(csv.DictReader(fh))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for i, (setting, label) in enumerate(LABELS.items()):
        sel = [r for r in rows if r["setting"] == setting]
        x = np.maximum([float(r["outside"]) for r in sel], FLOOR)
        y = np.maximum([abs(float(r["error"])) for r in sel], FLOOR)
        ax.scatter(x, y, s=14, color=viz.CAT[i], label=label, alpha=0.85, linewidths=0)
    ax.axvline(1e-9, color=viz.INK["secondary"], linestyle="--", linewidth=0.8)
    ax.annotate("tolerance", xy=(1e-9, 1e-15), xytext=(4, 0), textcoords="offset points",
                color=viz.INK["secondary"], fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("predictive mass outside the support")
    ax.set_ylabel("|EIG - reference| (nats)")
    ax.legend(loc="upper left")
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    viz.savefig(fig, os.path.join(HERE, "figures", "eig_accuracy.pdf"))


if __name__ == "__main__":
    main()
