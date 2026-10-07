"""Figures of this study, from ``results/``: ``figures/data.pdf`` and ``figures/scores.pdf``.

Run after ``python run.py``: python visualize.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, HERE)

import natural_figs                                                   # noqa: E402
import safecast_study                                                 # noqa: E402
from config import CONFIG, FIGURES                                    # noqa: E402

if __name__ == "__main__":
    natural_figs.fig_data(HERE, CONFIG["name"], safecast_study.PER_MINUTE)
    natural_figs.fig_scores(HERE, CONFIG["name"], **FIGURES)
