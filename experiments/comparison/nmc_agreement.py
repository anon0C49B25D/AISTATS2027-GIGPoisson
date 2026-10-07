"""How often a nested estimator of the EIG would choose a different design than the exact EIG.

Diagnostic for Section 6: the exact-EIG agent runs as in the study, and at every round it
also ranks every design by two nested estimators under the same GIG belief, plain nested
Monte Carlo and the PCE bound, at the sample sizes of ``agents.NMC_SIZES``. Like the agent's
own scores, a nested estimate is cached per context and redrawn only when that context is
observed, which is how the nested agents of ``run.py`` see them. Per round it records
whether each estimator's choice equals the exact one, and the exact EIG per unit cost that
the estimator's choice gives up, relative to the best. The trajectory is the exact agent's,
so the comparison is made on the same beliefs. Timings are not measured.

Writes ``results/nmc_agreement.csv`` (one row per setting, episode and estimator).
Run: python experiments/comparison/nmc_agreement.py [--reps 10]
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                                       # noqa: E402
import csv                                                            # noqa: E402
import sys                                                            # noqa: E402

import numpy as np                                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "common"))
sys.path.insert(0, HERE)

import run                                                            # noqa: E402
from agents import EIG, NMC_SIZES                                     # noqa: E402
from harness import run_episode                                       # noqa: E402


class EIGShadow(EIG):
    """Exact EIG that also records what nested estimators would have chosen."""

    criterion = "eig-shadow"

    def __init__(self, shadows, seed=0):
        super(EIGShadow, self).__init__()
        self.shadows = shadows
        self.seed = int(seed)

    def reset(self, env, prior_moments, recovery, rng=None):
        super(EIGShadow, self).reset(env, prior_moments, recovery, rng=rng)
        for i, (_, model) in enumerate(self.shadows):
            model.set_rng(np.random.default_rng(90_000 + 1000 * i + self.seed))
            env.initial_beliefs(model)          # registers the class priors for the memo
        self._shadow = [dict() for _ in self.shadows]
        self.log = []

    def observe(self, k, y, f):
        super(EIGShadow, self).observe(k, y, f)
        for cache in self._shadow:
            cache.pop(k, None)

    def _shadow_scores(self, i, env, k):
        cache = self._shadow[i]
        if k not in cache:
            model = self.shadows[i][1]
            cache[k] = [model.eig(self.beliefs[k], env.exposure((k, v), self.recovery))
                        / env.cost((k, v)) for v in env.action_values]
        return cache[k]

    def act(self, env, rng):
        k, v = super(EIGShadow, self).act(env, rng)
        best = max(max(self._block_scores(env, j)) for j in range(env.n_contexts))
        smallest = float(min(env.action_values))
        rec = []
        for i in range(len(self.shadows)):
            top = None
            for j in range(env.n_contexts):
                for a, s in enumerate(self._shadow_scores(i, env, j)):
                    if top is None or s > top[0]:
                        top = (s, j, a)
            _, j, a = top
            same = (j == k) and abs(float(env.action_values[a]) - v) < 1e-12
            loss = (best - self._block_scores(env, j)[a]) / best
            rec.append((same, loss, float(env.action_values[a]) > smallest))
        self.log.append(rec)
        return k, v


def one(setting, e):
    from methods.countmodels import GIGPoissonNMC
    run.ENV_KW = {}
    env, cps = run.make_env(setting)
    n = NMC_SIZES.get(setting, 1024)
    shadows = [("nmc", GIGPoissonNMC(n, n, pce=False)), ("pce", GIGPoissonNMC(n, n, pce=True))]
    agent = EIGShadow(shadows, seed=e)
    run_episode(env, agent, e, cps)
    out = []
    for i, (label, _) in enumerate(shadows):
        same = np.array([r[i][0] for r in agent.log], float)
        loss = np.array([r[i][1] for r in agent.log], float)
        longer = np.array([r[i][2] for r in agent.log], float)
        out.append((setting, e, label, n, len(agent.log), 1.0 - same.mean(), loss.mean(),
                    loss.max(), longer.mean()))
    return out


def main():
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    tasks = [(s, e) for s in ("bulk", "gamma", "hardxray") for e in range(args.reps)]
    res = Parallel(n_jobs=args.jobs, verbose=5)(delayed(one)(s, e) for s, e in tasks)
    rows = [r for chunk in res for r in chunk]
    path = os.path.join(HERE, "results", "nmc_agreement.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["setting", "episode", "estimator", "n", "rounds", "disagree",
                    "mean_relative_loss", "max_relative_loss", "frac_longer_than_shortest"])
        w.writerows(rows)
    for s in ("bulk", "gamma", "hardxray"):
        for label in ("nmc", "pce"):
            sub = [r for r in rows if r[0] == s and r[2] == label]
            print("{:9s} {}: disagrees in {:.1%} of rounds, gives up {:.2%} of the best EIG per "
                  "cost on average, picks a longer action in {:.1%}".format(
                      s, label, np.mean([r[5] for r in sub]), np.mean([r[6] for r in sub]),
                      np.mean([r[8] for r in sub])))
    print("wrote {}".format(os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()
