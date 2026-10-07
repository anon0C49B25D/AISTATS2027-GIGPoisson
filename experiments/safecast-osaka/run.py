"""Fit the mixing laws to this study's measured counts and score them on held-out cells.

Writes ``results/``: ``describe.csv`` (the data), ``population.csv`` (every law fitted to every
cell of a class, AIC and BIC), ``scores.csv`` and ``fits.csv`` (per episode), ``summary.csv``
(paired comparison with the GIG), ``cells.csv`` and ``info.json``. Reproducible from seed 0.

Run: python run.py [episodes [workers]]
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, HERE)

import safecast_study                                                 # noqa: E402
from config import CONFIG                                             # noqa: E402

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    w = int(sys.argv[2]) if len(sys.argv) > 2 else None
    safecast_study.run(CONFIG, HERE, episodes=n, workers=w)
