"""Fetch the Safecast drive logs of this study and parse them into ``data/readings.npz``.

``data/imports.csv`` lists the logs. It is reused when present, so a rerun reads exactly the
same logs; ``--discover`` searches the measurement API again (slow, and the API's paging means
it may find a different set). Raw logs are cached outside the repository, see
``experiments/common/safecast.py``.

Run: python fetch.py [--discover]
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, HERE)

import safecast_study                                                 # noqa: E402
from config import CONFIG                                             # noqa: E402

if __name__ == "__main__":
    safecast_study.fetch(CONFIG, HERE, discover="--discover" in sys.argv)
