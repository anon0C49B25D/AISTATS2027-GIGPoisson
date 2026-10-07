"""The scores each prior earns before any budget is spent, on the episodes of the study.

The log score of Section 6 averages over every context, and with a budget of one or two of
the shortest observations per context most of it rests on the prior predictive. This script
measures that part directly: for every setting, every episode and every model, the held-out
log score and the top-m regret of the fitted class priors alone, with no counts. A row here
pairs with the same episode of ``run.py`` (same seed, same held-out counts), so the gap
between two priors here can be set beside the gap between the same priors at the full
budget.

By Lemma 2(iii) the difference between mixing laws shrinks with the exposure at which the
prediction is scored, so the scores are also computed at a quarter of and at four times the
paper's evaluation exposure.

Writes ``results/prior_only.csv``. Run: python experiments/comparison/prior_only.py
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                                       # noqa: E402
import csv                                                            # noqa: E402
import sys                                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "common"))
sys.path.insert(0, HERE)

import run                                                            # noqa: E402
from agents.base import Agent                                         # noqa: E402
from harness import TOP_M, prior_scores                               # noqa: E402

#: Evaluation exposure of the paper, per setting, in the setting's own unit.
EVAL = {"bulk": 20.0, "gamma": 25.0, "hardxray": 20.0}
MULTIPLIERS = (0.25, 1.0, 4.0)


def models():
    from methods.compoundgamma import CompoundGammaPoisson
    from methods.countmodels import GammaPoisson, GIGPoisson
    from methods.gammamixture import GammaMixturePoisson
    from methods.logskewnormal import LogSkewNormalQuad
    out = [GIGPoisson(), GammaPoisson(), GammaMixturePoisson(), CompoundGammaPoisson(),
           LogSkewNormalQuad(), LogSkewNormalQuad(penalised=True, name="logskewnormal-quad-pen")]
    for m in out:
        _cache_fits(m)
    return out


def _cache_fits(model):
    """The calibration catalogue of a class is the same in every episode, so is its fit."""
    fit, cache = model.fit_population, {}

    def cached(rates):
        key = tuple(float(x) for x in rates)
        if key not in cache:
            cache[key] = fit(rates)
        return cache[key]
    model.fit_population = cached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--settings", nargs="+", default=list(EVAL))
    args = ap.parse_args()
    rows = []
    for setting in args.settings:
        for mult in MULTIPLIERS:
            run.ENV_KW = {} if mult == 1.0 else {"eval": EVAL[setting] * mult}
            env, _ = run.make_env(setting)
            for model in models():
                agent = Agent(model=model)
                for e in range(args.reps):
                    rows.append((setting, EVAL[setting] * mult, model.name, e)
                                + tuple(prior_scores(env, agent, e)))
                print("  {} eval x{:g} {}: done".format(setting, mult, model.name), flush=True)
    path = os.path.join(HERE, "results", "prior_only.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["setting", "eval", "model", "episode", "nlpd"]
                   + ["regret_top{}".format(m) for m in TOP_M] + ["rate_rmse", "rate_dex"])
        w.writerows(rows)
    print("wrote {} ({} rows)".format(os.path.relpath(path, ROOT), len(rows)))


if __name__ == "__main__":
    main()
