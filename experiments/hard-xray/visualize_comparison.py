"""Figures for the agent comparison on hard X-ray follow-up of Swift-BAT sources.

Reads ``experiments/comparison/results/hardxray.csv`` (192 sources, 1-99 per cent flux band,
NLPD scored on 20 ks integrations, class priors fitted to 20 catalogued sources per class)
and, where they exist, the catalogue-size runs ``pop40_hardxray.csv`` and
``popall_hardxray.csv``; writes into ``figures/``, each set under its prefix and once per
group of methods:

    comparison_<group>_nlpd_box.pdf      NLPD per Monte Carlo run at the full budget
    comparison_<group>_regret_topm.pdf   mean top-m regret against budget, m = 1..5
    comparison_<group>_nlpd_time.pdf     mean NLPD against ms per round (log scale)

The figures the manuscript includes are drawn by ``experiments/comparison/paper_figures.py``.

Run after ``python experiments/comparison/run.py hardxray``:
    python experiments/hard-xray/visualize_comparison.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

import comparison_figs                                                # noqa: E402

if __name__ == "__main__":
    for prefix, title in (("", "Hard X-ray follow-up, priors from 20 sources per class"),
                          ("pop40_", "Hard X-ray follow-up, priors from 40 sources per class"),
                          ("popall_", "Hard X-ray follow-up, priors from the full catalogue half")):
        if os.path.exists(os.path.join(comparison_figs.RESULTS, prefix + "hardxray.csv")):
            comparison_figs.make_all("hardxray", prefix, os.path.join(HERE, "figures"),
                                     title, "ks")
