"""Train the amortised design policy (``agents/dad.py``) for one setting.

Simulates experiments from the agent's own model (GIG-Poisson priors as the environment
supplies them, rates drawn from those priors) on seeds disjoint from the evaluation
episodes, and runs REINFORCE on the summed per-context sPCE bound. Writes the weights, the
training curve and the training wall-clock time to ``dad/<setting key>.json``, which the
agent loads at reset.

Run, with the same setting flags as run.py (none: the paper's settings):
    python experiments/comparison/train_dad.py bulk
    python experiments/comparison/train_dad.py gamma
    python experiments/comparison/train_dad.py hardxray
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                                       # noqa: E402
import json                                                           # noqa: E402
import sys                                                            # noqa: E402
import time                                                           # noqa: E402

import numpy as np                                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [ROOT, HERE, os.path.join(ROOT, "experiments", "common")]

import functools                                                      # noqa: E402
import importlib.util                                                 # noqa: E402

from agents.dad import WEIGHTS_DIR, env_key, simulate, train_policy  # noqa: E402
from methods.countmodels import GIGPoisson                            # noqa: E402

def _runner():
    """The comparison runner, loaded by path under its own name.

    experiments/gamma-ray has a run.py too, and once a process has that directory on its
    path, ``import run`` finds the wrong one; worker processes load it here themselves. The
    loaded module is kept only in ``sys.modules``, never in a global of this file, because
    worker jobs are pickled with this file's globals and a module cannot be.
    """
    mod = sys.modules.get("comparison_run")
    if mod is None:
        spec = importlib.util.spec_from_file_location("comparison_run",
                                                      os.path.join(HERE, "run.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["comparison_run"] = mod
        spec.loader.exec_module(mod)
    return mod


def make_env(setting, kw):
    R = _runner()
    R.ENV_KW = dict(kw)
    return R.make_env(setting)[0]


def draw_beliefs(env, model, i):
    moments, _, rec = env.blocks(np.random.default_rng(TRAIN_SEED0 + i))
    if hasattr(env, "initial_beliefs"):
        return env.initial_beliefs(model), moments, rec
    return [model.prior_from_moments(*mv) for mv in moments], moments, rec


def episode(setting, kw, w, i, greedy=False):
    """One training episode, built from plain arguments so it can run in a worker."""
    env = make_env(setting, kw)
    model = GIGPoisson()
    b, mom, rec = draw_beliefs(env, model, i)
    return simulate(w, env, model, b, mom, rec, np.random.default_rng(i), greedy=greedy)


#: Validation and held-out episodes: simulated, disjoint from training and from evaluation.
VALIDATION = range(5 * 10 ** 5, 5 * 10 ** 5 + 24)
HELDOUT = range(9 * 10 ** 5, 9 * 10 ** 5 + 24)


def score(setting, kw, w, episodes, pool):
    """Mean greedy sPCE of weights ``w`` over simulated ``episodes``."""
    from joblib import delayed
    out = pool(delayed(episode)(setting, kw, w.copy(), i, True) for i in episodes)
    return float(np.mean([o[0] for o in out]))


#: Training draws episodes from seeds far from the evaluation ones (1000 + e).
TRAIN_SEED0 = 10 ** 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("setting", choices=("bulk", "gamma", "hardxray"))
    ap.add_argument("--band", type=float, nargs=2, default=None)
    ap.add_argument("--sources", type=int, default=None)
    ap.add_argument("--eval-time", type=float, default=None)
    ap.add_argument("--pop-size", type=int, default=None)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    kw = {}
    if args.band is not None:
        kw["band"] = tuple(args.band)
    if args.sources is not None:
        kw["n_contexts"] = args.sources
    if args.eval_time is not None:
        kw["eval"] = args.eval_time
    if args.pop_size is not None:
        kw["pop_size"] = args.pop_size
    model = GIGPoisson()
    env = make_env(args.setting, kw)
    key = env_key(env)
    print("training the amortised policy for {}".format(key))
    t0 = time.time()
    w, curve, checks = train_policy(
        functools.partial(episode, args.setting, kw), n_iter=args.iters, batch=args.batch,
        n_jobs=args.jobs, validate=lambda ww, pool: score(args.setting, kw, ww, VALIDATION, pool))
    seconds = time.time() - t0

    # Held-out check on simulated episodes used neither for training nor for selection:
    # the selected policy against the D-optimality-per-cost policy it started from.
    from joblib import Parallel
    w0 = np.zeros_like(w)
    w0[3] = 1.0
    with Parallel(n_jobs=args.jobs) as pool:
        res = {"trained": score(args.setting, kw, w, HELDOUT, pool),
               "start": score(args.setting, kw, w0, HELDOUT, pool)}
    print("held-out sPCE, greedy: selected policy {:.2f}, D-opt start {:.2f}".format(
        res["trained"], res["start"]))

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    path = os.path.join(WEIGHTS_DIR, key + ".json")
    with open(path, "w") as fh:
        json.dump(dict(key=key, w=w.tolist(), curve=curve, train_seconds=seconds,
                       iters=args.iters, batch=args.batch,
                       validation=checks, heldout_spce_trained=res["trained"],
                       heldout_spce_start=res["start"]), fh, indent=1)
    print("wrote {} ({:.0f} s of training)".format(os.path.relpath(path, ROOT), seconds))


if __name__ == "__main__":
    main()
