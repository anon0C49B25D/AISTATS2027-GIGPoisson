"""How scarce does the budget have to be before choosing where to sample matters?

The comparison in ``run.py`` is run at one budget and one number of blocks, and a single
such point turns out not to be a general statement about anything. When the budget is large
relative to the number of contexts, every block ends up well measured whatever the schedule
does, the prior differences wash out, and visiting each in turn is near-optimal: no
acquisition rule can distinguish itself, and none does. When it is small, most blocks cannot
be sampled at all and the allocation is the whole problem.

This sweep holds the budget at 240 cubic metres and varies the number of blocks that budget
has to cover, so the axis is cubic metres available per block. The question it answers is the
one a practitioner actually has, which is not "is this acquisition better" but "at what point
is it worth the trouble".

Writes ``results/scarcity_sweep.csv``. Run after ``run.py``.

Run: python experiments/bulk-sampling/sweep.py
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import agents                                                         # noqa: E402
import run as R                                                       # noqa: E402
from domain import BulkSampling                                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGET = 240.0
#: Blocks the same budget must cover. The last is scarce enough that most blocks go
#: unsampled; the first is abundant enough that all of them are well measured.
BLOCK_COUNTS = (6, 12, 24, 48, 96)
N_ENV = 24


def main():
    rows = []
    for n_blocks in BLOCK_COUNTS:
        env = BulkSampling(n_blocks=n_blocks, budget=BUDGET)
        per = BUDGET / n_blocks
        print("\n{} blocks, {:.1f} m^3 each on average".format(n_blocks, per))
        acc = {}
        for e in range(N_ENV):
            rng = np.random.default_rng(1000 + e)
            moments, truths, recovery = env.blocks(rng)
            fields = env.stone_fields(truths, recovery, rng)
            heldout = env.held_out(truths, recovery, rng, n=R.N_HELDOUT)
            for agent in agents.build_all(cross="models"):
                cps, _ = R.run_agent(env, agent, moments, truths, recovery, fields,
                                     heldout, np.random.default_rng(50_000 + e),
                                     checkpoints_at=(BUDGET,))
                acc.setdefault(agent.name, []).append(cps[-1][3:6])
        base = np.array(acc["eig"])
        for name, vals in acc.items():
            v = np.array(vals)
            # Paired against our agent on the same properties, which is the only
            # comparison the shared gravel makes meaningful.
            d = base[:, 0] - v[:, 0]
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std(ddof=1) > 0 else 0.0
            rows.append((n_blocks, per, name, *v.mean(0), float(d.mean()), float(t)))
        print("  {:<24} {:>9} {:>10} {:>9} {:>10} {:>7}".format(
            "agent", "NLPD", "grade RMSE", "regret", "dNLPD", "t"))
        for r in [x for x in rows if x[0] == n_blocks]:
            print("  {:<24} {:9.4f} {:10.4f} {:9.4f} {:10.4f} {:+7.1f}".format(r[2], *r[3:]))

    path = os.path.join(HERE, "results", "scarcity_sweep.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n_blocks", "budget_per_block", "policy", "nlpd", "grade_rmse",
                    "regret", "nlpd_delta_vs_eig", "paired_t"])
        w.writerows(rows)
    print("\n  wrote results/scarcity_sweep.csv ({} rows)".format(len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
