"""The comparison figures the manuscript includes, written into the manuscript's ``figures/``.

    nlpd.pdf                   held-out NLPD at the full budget, one panel per setting: ours
                               and the four baselines of the main text
    regret.pdf                 top-1 and top-5 regret against budget spent, the same methods
    nlpd_acquisitions.pdf      the same NLPD panels for the allocation rules (appendix)
    regret_acquisitions.pdf    the same regret panels for the allocation rules (appendix)

Reads ``results/{bulk,gamma,hardxray}.csv``, written by ``run.py``. The per-study figures,
one set per group of methods, are drawn by each study's ``visualize_comparison.py``.

Run: python experiments/comparison/paper_figures.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "common"))

import comparison_figs as cf                                          # noqa: E402

OUT = os.path.join(ROOT, "figures")


def main():
    os.makedirs(OUT, exist_ok=True)
    cf.paper_nlpd_grid(os.path.join(OUT, "nlpd.pdf"), cf.PAPER_METHODS,
                       paired=("eig@gamma-poisson", "eig@gammamix-poisson"))
    cf.paper_regret_grid(os.path.join(OUT, "regret.pdf"), cf.PAPER_METHODS, ms=(1, 5))
    cf.paper_nlpd_grid(os.path.join(OUT, "nlpd_acquisitions.pdf"), cf.PAPER_ACQUISITIONS,
                       height=2.3)
    cf.paper_regret_grid(os.path.join(OUT, "regret_acquisitions.pdf"),
                         cf.PAPER_ACQUISITIONS, ms=(1, 5))


if __name__ == "__main__":
    main()
