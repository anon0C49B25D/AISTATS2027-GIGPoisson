"""Single-process cost of one EIG evaluation, per model, setting, class and action.

The per-round times of ``run.py`` are measured with twelve episodes running at once, which
inflates them several-fold and unevenly between vectorised and scalar code. This script times
the criteria alone, in one process with one BLAS thread, on an otherwise idle machine:

    eig           one evaluation of the expected information gain, memoisation bypassed,
                  at each class prior and after one observation at the shortest action
                  (the count set to its predictive mean, rounded)
    support       for the GIG and the compound gamma, the number of counts the predictive sum
                  runs over, which is what the cost of the exact EIG grows with
    d-optimality  one evaluation of the local D-optimality score under the GIG

Models: the GIG (ours), the gamma and the gamma mixture, the compound gamma and the
penalised skew-normal (both by quadrature), and the GIG with nested estimators of the EIG at
the sample sizes of ``agents.NMC_SIZES``.

Writes ``results/timing.csv`` and prints medians per model and setting, with the hardware.
Run with nothing else on the machine: python experiments/comparison/timing.py
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import csv                                                            # noqa: E402
import platform                                                       # noqa: E402
import subprocess                                                     # noqa: E402
import sys                                                            # noqa: E402
import time                                                           # noqa: E402

import numpy as np                                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import run                                                            # noqa: E402
from agents import NMC_SIZES                                          # noqa: E402


def hardware():
    cpu = platform.processor()
    if sys.platform.startswith("win"):
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
            cpu = out or cpu
        except Exception:
            pass
    import scipy
    return "{}; {} logical cores; {}; Python {}, NumPy {}, SciPy {}".format(
        cpu, os.cpu_count(), platform.platform(), platform.python_version(), np.__version__,
        scipy.__version__)


def models(setting):
    from methods.compoundgamma import CompoundGammaPoisson
    from methods.countmodels import GammaPoisson, GIGPoisson, GIGPoissonNMC
    from methods.gammamixture import GammaMixturePoisson
    from methods.logskewnormal import LogSkewNormalQuad
    n = NMC_SIZES.get(setting, 1024)
    return [("gig", GIGPoisson()), ("gamma", GammaPoisson()), ("mixture", GammaMixturePoisson()),
            ("compound-gamma", CompoundGammaPoisson()),
            ("skew-normal-pen", LogSkewNormalQuad(penalised=True)),
            ("gig-nmc", GIGPoissonNMC(n, n, pce=False)), ("gig-pce", GIGPoissonNMC(n, n, pce=True))]


def raw_eig(model, belief, f):
    """One evaluation by the agent's own routine, never served from a memo."""
    if model.name in ("gig-nmc", "gig-pce"):
        return model.nmc_eig(belief, f)
    for memo in ("_memo", "_prior_memo"):
        if isinstance(getattr(model, memo, None), dict):
            getattr(model, memo).clear()
    return model.eig(belief, f)


def support_size(model, belief, f):
    if model.name == "gig-poisson":
        return len(model.support(belief, f))
    return float("nan")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=["bulk", "gamma", "hardxray"])
    settings = ap.parse_args().settings
    hw = hardware()
    print(hw)
    rows = []
    for setting in settings:
        env, _ = run.make_env(setting)
        rec = (env.blocks(np.random.default_rng(1000))[2] if setting == "bulk"
               else np.ones(env.n_contexts))
        first = {c: env._classes.index(c) for c in env.CLASSES}
        for label, model in models(setting):
            if hasattr(model, "set_rng"):
                model.set_rng(np.random.default_rng(0))
            for c, k in first.items():
                prior = model.fit_population(env.population[c])
                f0 = env.exposure((k, float(env.action_values[0])), rec)
                y0 = float(np.round(f0 * model.rate_mean(prior)))
                for state, belief in (("prior", prior), ("posterior", model.update(prior, y0, f0))):
                    for v in env.action_values:
                        f = env.exposure((k, float(v)), rec)
                        t0 = time.perf_counter()
                        value = raw_eig(model, belief, f)
                        ms = 1e3 * (time.perf_counter() - t0)
                        rows.append((setting, label, c, state, float(v), f, value, ms,
                                     support_size(model, belief, f)))
                        if label == "gig":
                            t0 = time.perf_counter()
                            model.fisher_dopt(belief, f)
                            rows.append((setting, "d-optimality", c, state, float(v), f,
                                         float("nan"), 1e3 * (time.perf_counter() - t0),
                                         float("nan")))
            print("  {} {}: done".format(setting, label), flush=True)
    path = os.path.join(HERE, "results", "timing_{}.csv".format("_".join(settings)))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# " + hw])
        w.writerow(["setting", "model", "class", "state", "action", "exposure", "eig", "ms",
                    "support"])
        w.writerows(rows)
    print("\nmedian ms per evaluation (all classes, actions and states)")
    names = []
    for r in rows:
        if r[1] not in names:
            names.append(r[1])
    for setting in settings:
        cells = []
        for name in names:
            ms = [r[7] for r in rows if r[0] == setting and r[1] == name]
            cells.append("{} {:.2f}".format(name, float(np.median(ms))))
        print("  {:9s} ".format(setting) + " | ".join(cells))
    print("wrote {}".format(os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()
