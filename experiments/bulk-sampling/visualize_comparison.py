"""Figures for the agent comparison on bulk sampling over facies.

Reads ``experiments/comparison/results/bulk.csv`` (96 blocks in four facies, 240 m^3, NLPD
scored on 20 m^3 samples, facies priors fitted to 20 previously mined blocks per facies)
and, where they exist, the catalogue-size runs ``pop40_bulk.csv`` and ``pop200_bulk.csv``;
writes into ``figures/``, each set under its prefix and once per group of methods:

    comparison_<group>_nlpd_box.pdf      NLPD per Monte Carlo run at the full budget
    comparison_<group>_regret_topm.pdf   mean top-m regret against budget, m = 1..5
    comparison_<group>_nlpd_time.pdf     mean NLPD against ms per round (log scale)

The figures the manuscript includes are drawn by ``experiments/comparison/paper_figures.py``.

Run after ``python experiments/comparison/run.py bulk``:
    python experiments/bulk-sampling/visualize_comparison.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

import comparison_figs                                                # noqa: E402

if __name__ == "__main__":
    for prefix, title in (("", "Bulk sampling, priors from 20 blocks per facies"),
                          ("pop40_", "Bulk sampling, priors from 40 blocks per facies"),
                          ("pop200_", "Bulk sampling, priors from 200 blocks per facies")):
        if os.path.exists(os.path.join(comparison_figs.RESULTS, prefix + "bulk.csv")):
            comparison_figs.make_all("bulk", prefix, os.path.join(HERE, "figures"),
                                     title, "m$^3$")
