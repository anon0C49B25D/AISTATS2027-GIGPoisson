"""Numerical accuracy of the deterministic EIG at the priors the paper fits.

The EIG of the GIG-Poisson model is computed by a truncated support sum and a fixed
quadrature, both in log space (``methods/countmodels.py``, ``GIGPoisson.eig``):

    support sum   widened until the predictive mass outside is below 1e-9, capped at
                  2e6 counts (``methods/gigpoisson.py``, ``sichel_support``)
    second term   64-node Gauss-Legendre rule in log lam, over the window where the log
                  density of the belief is within 75 nats of its maximum

Neither error is bounded analytically, so this study measures it against a reference that
tightens both settings and shares only the Bessel routine:

    reference     the predictive summed in chunks until the mass outside is below 1e-12,
                  or to 6e7 counts, and the second term by adaptive quadrature (SciPy
                  ``quad``) over a 200-nat window

at the beliefs where the support is longest: the fitted class priors of the three settings
(calibration catalogues of 20 members, as in ``comparison/run.py``) and the belief after one
observation at the shortest action, at every action. These are the states that
``comparison/timing.py`` times.

Writes ``data/priors.csv`` (the fitted priors) and ``results/eig_accuracy.csv``, and prints a
summary. Deterministic. Run: python experiments/eig-accuracy/run.py   (~3 min)
"""

import csv
import os
import sys

import numpy as np
from scipy import integrate
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "comparison"))

import run as comparison                                              # noqa: E402
from methods.countmodels import GIGPoisson, poisson_entropy           # noqa: E402
from methods.gigpoisson import gig_logpdf, sichel_logpmf              # noqa: E402

SETTINGS = ("bulk", "gamma", "hardxray")

#: Reference settings: tail mass, longest support and chunk length of the predictive sum, and
#: the width of the window, in nats of log density, that the second term integrates over.
REF_TAIL = 1e-12
REF_CAP = 60_000_000
REF_CHUNK = 1_000_000
REF_WINDOW = 200.0


def predictive_default(model, belief, f):
    """Entropy, mass left outside, and length of the support the shipped EIG sums over."""
    y = model.support(belief, f)
    lp = model.logpmf(belief, f, y)
    return -float(np.sum(np.exp(lp) * lp)), 1.0 - float(np.exp(logsumexp(lp))), len(y)


def predictive_reference(belief, f):
    """Entropy summed in chunks until the mass outside is below ``REF_TAIL``."""
    entropy, mass, start = 0.0, 0.0, 0
    while start < REF_CAP:
        lp = sichel_logpmf(np.arange(start, start + REF_CHUNK, dtype=float), belief, f)
        p = np.exp(lp)
        entropy -= float(np.sum(p * lp))
        mass += float(np.sum(p))
        start += REF_CHUNK
        if 1.0 - mass < REF_TAIL:
            break
    return entropy, 1.0 - mass, start


def aleatoric_reference(belief, f):
    """Second term of the EIG by adaptive quadrature in s = log lam."""
    alpha, a, b = belief["alpha"], belief["a"], belief["b"]

    def h(s):
        return alpha * s - 0.5 * (a * np.exp(-s) + b * np.exp(s))

    mode = ((alpha - 1.0) + np.sqrt((alpha - 1.0) ** 2 + a * b)) / b
    s_mode = float(np.log(mode)) if mode > 0 else float(np.log(np.sqrt(a / b)))
    bounds = []
    for direction in (-1.0, 1.0):
        step, far = 1.0, s_mode
        while h(far) > h(s_mode) - REF_WINDOW and step < 1e7:
            far = s_mode + direction * step
            step *= 2.0
        bounds.append(far)

    def integrand(s):
        lam = np.exp(s)
        return float(np.exp(gig_logpdf(lam, belief)) * lam * poisson_entropy(f * lam))

    edges = sorted(set([bounds[0], s_mode, bounds[1]]
                       + list(np.linspace(bounds[0], bounds[1], 41)[1:-1])))
    return sum(integrate.quad(integrand, lo, hi, epsabs=1e-14, epsrel=1e-12, limit=400)[0]
               for lo, hi in zip(edges[:-1], edges[1:]))


def main():
    model = GIGPoisson()
    priors, rows = [], []
    for setting in SETTINGS:
        env, _ = comparison.make_env(setting)
        rec = (env.blocks(np.random.default_rng(1000))[2] if setting == "bulk"
               else np.ones(env.n_contexts))
        first = {c: env._classes.index(c) for c in env.CLASSES}
        for c, k in first.items():
            prior = model.fit_population(env.population[c])
            priors.append((setting, c, prior["alpha"], prior["a"], prior["b"]))
            f0 = env.exposure((k, float(env.action_values[0])), rec)
            y0 = float(np.round(f0 * model.rate_mean(prior)))
            for state, belief in (("prior", prior), ("posterior", model.update(prior, y0, f0))):
                for v in env.action_values:
                    f = env.exposure((k, float(v)), rec)
                    model.__dict__.get("_memo", {}).clear()
                    eig = model.eig(belief, f)
                    h_def, outside, support = predictive_default(model, belief, f)
                    a_def = model.aleatoric_entropy(belief, f)
                    h_ref, outside_ref, support_ref = predictive_reference(belief, f)
                    a_ref = aleatoric_reference(belief, f)
                    eig_ref = h_ref - a_ref
                    rows.append((setting, c, state, float(v), f, support, outside, eig, eig_ref,
                                 eig - eig_ref, h_def - h_ref, a_ref - a_def, support_ref,
                                 outside_ref))
                    print("{:8s} {:10s} {:9s} action {:5} support {:8d} outside {:9.2e} "
                          "EIG {:.6f} error {:+.1e}".format(setting, c, state, v, support,
                                                            outside, eig, eig - eig_ref),
                          flush=True)

    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "data", "priors.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["setting", "class", "order", "a", "b"])
        w.writerows(priors)
    path = os.path.join(HERE, "results", "eig_accuracy.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["setting", "class", "state", "action", "exposure", "support", "outside",
                    "eig", "eig_ref", "error", "error_support", "error_quadrature",
                    "support_ref", "outside_ref"])
        w.writerows(rows)

    err = np.array([r[9] for r in rows])
    capped = np.array([r[5] > 2_000_000 for r in rows])
    outside = np.array([r[6] for r in rows])
    print("\n{} evaluations".format(len(rows)))
    print("  largest |error|            {:.2e} nats".format(np.max(np.abs(err))))
    print("  largest |relative error|   {:.2e}".format(
        np.max(np.abs(err) / np.array([r[8] for r in rows]))))
    print("  largest |quadrature error| {:.2e} nats".format(max(abs(r[11]) for r in rows)))
    print("  cap binds in {} evaluations: mass outside up to {:.2e}, |error| up to {:.2e}".format(
        int(capped.sum()), outside[capped].max(), np.abs(err[capped]).max()))
    print("  elsewhere: mass outside up to {:.2e}, |error| up to {:.2e}".format(
        outside[~capped].max(), np.abs(err[~capped]).max()))
    print("  longest support after one observation: {}".format(
        max(r[5] for r in rows if r[2] == "posterior")))
    print("  reference: mass outside up to {:.2e}, support up to {}".format(
        max(r[13] for r in rows), max(r[12] for r in rows)))
    print("wrote {}".format(os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()
