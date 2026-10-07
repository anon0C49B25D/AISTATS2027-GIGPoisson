"""How well each family's class prior, fitted to a catalogue of n known members, predicts the
rest of the class: the priors alone, with no counts and no allocation.

For each telescope catalogue (Fermi-LAT 4FGL and Swift-BAT 105-month, 1-99 per cent flux
band, split in half within each class as the comparison splits them), each class and each
catalogue size n in {10, 20, 40, 80}, n sources are drawn from the population half, every
family is fitted to them exactly as the agents fit it (the GIG, the gamma and the gamma
mixture to the rates, the skew-normal to their logarithms), and each fitted density is
scored on every source of the other half. Twenty draws per size, and once more on the whole
population half (``n = 0`` in the output).

Writes ``results/prior_fit.csv`` (one row per catalogue, class, size and draw: summed held-out
log-density per family) and prints the difference per held-out source, GIG minus each rival.

Run: python experiments/comparison/prior_fit.py
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv                                                            # noqa: E402
import importlib.util                                                 # noqa: E402
import sys                                                            # noqa: E402
import warnings                                                       # noqa: E402

import numpy as np                                                    # noqa: E402
from scipy.special import gammaln, logsumexp                          # noqa: E402
from scipy.stats import gamma as gamma_dist, geninvgauss, skewnorm    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from methods.countmodels import GammaPoisson, GIGPoisson              # noqa: E402
from methods.gammamixture import GammaMixturePoisson                  # noqa: E402

#: Catalogue sizes; 0 stands for the whole population half, fitted once.
SIZES = (10, 20, 40, 80, 0)
DRAWS = 20
CATALOGUES = {"gamma": ("gamma-ray", "PhotonCounting"),
              "hardxray": ("hard-xray", "HardXRayFollowup")}
FAMILIES = ("gig", "gamma", "mixture", "skewnormal")


def environment(name):
    directory, cls = CATALOGUES[name]
    path = os.path.join(ROOT, "experiments", directory)
    if path not in sys.path:
        sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location("env_" + name,
                                                  os.path.join(path, "environment.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, cls)(band=(1.0, 99.0))


def log_gig(x, held):
    b = GIGPoisson().fit_population(x)
    return geninvgauss.logpdf(held, b["alpha"], np.sqrt(b["a"] * b["b"]),
                              scale=np.sqrt(b["a"] / b["b"]))


def log_gamma(x, held):
    b = GammaPoisson().fit_population(x)
    return gamma_dist.logpdf(held, b["a"], scale=1.0 / b["b"])


def log_mixture(x, held):
    b = GammaMixturePoisson().fit_population(x)
    a, r = b["a"], b["b"]
    lp = ((a * np.log(r) - gammaln(a))[None] + (a - 1.0)[None] * np.log(held)[:, None]
          - r[None] * held[:, None])
    return logsumexp(lp + b["logw"][None], axis=1)


def log_skewnormal(x, held):
    al, xi, om = skewnorm.fit(np.log(x))          # as LogSkewNormalPoisson.fit_population
    s = np.log(held)
    return skewnorm.logpdf(s, al, xi, om) - s       # density of the rate, not its logarithm


def one(name, klass, n, draw):
    warnings.filterwarnings("ignore")
    env = environment(name)
    pop, held = env.population[klass], env.targets[klass]
    x = pop if n == 0 else np.random.default_rng(1000 * draw + n).choice(
        pop, min(n, pop.size), replace=False)
    return (name, klass, n, draw, held.size) + tuple(
        float(np.sum(f(x, held))) for f in (log_gig, log_gamma, log_mixture, log_skewnormal))


def main():
    from joblib import Parallel, delayed
    tasks = [(name, c, n, r) for name in CATALOGUES for c in environment(name).CLASSES
             for n in SIZES for r in range(DRAWS if n else 1)]
    rows = Parallel(n_jobs=12)(delayed(one)(*t) for t in tasks)
    path = os.path.join(HERE, "results", "prior_fit.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["catalogue", "class", "n", "draw", "held_out"]
                   + ["loglik_" + f for f in FAMILIES])
        w.writerows(rows)
    print("wrote {}".format(os.path.relpath(path, ROOT)))
    names = np.array([r[0] for r in rows])
    num = np.array([r[2:] for r in rows], dtype=float)
    for label, keep in (("both", np.ones(len(rows), bool)),) + tuple(
            (name, names == name) for name in CATALOGUES):
        for n in SIZES:
            sel = num[keep & (num[:, 0] == n)]
            if not sel.size:
                continue
            held = sel[:, 2].sum()
            gaps = ", ".join("{} {:+.3f}".format(f, (sel[:, 3] - sel[:, 3 + i]).sum() / held)
                             for i, f in enumerate(FAMILIES) if i)
            print("{:<9} n = {:>3}: GIG minus {} nats per held-out source".format(
                label, n if n else "all", gaps))


if __name__ == "__main__":
    main()
