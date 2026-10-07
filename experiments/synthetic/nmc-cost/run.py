"""What nested Monte Carlo costs to match a quadrature that is already exact.

Three questions, three result files.

**A. Accuracy against wall-clock.** For a geometric sweep of sample sizes, how close does
the estimator get to the exact value, and what does that cost? The exact quadrature is one
point on the same axes. ``results/accuracy.csv``.

**B. Where the error lives.** The nesting bias is a function of the inner sample size only.
Holding the inner size fixed and raising the outer size drives the variance down and leaves
the bias where it was; raising the inner size retires it at order ``1/M``. Run as a grid so
the two axes separate. ``results/bias.csv``.

**C. Whether it reaches the decision.** An estimator off by a constant still ranks
candidates correctly. Over many beliefs and a full grid of candidate actions, how often does
the estimator's argmax differ from the exact one, and what does that choice cost in exact
information per unit of budget? ``results/decisions.csv``.

Run: python experiments/synthetic/nmc-cost/run.py
"""

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from methods.countmodels import GIGPoisson                            # noqa: E402
from methods.gigpoisson import gig_from_mean, gig_moments             # noqa: E402
from nmc import nmc_eig                                               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0

#: Sample sizes on the diagonal ``N = M``. Cost grows as ``N M``, so the top of the sweep
#: already costs a second an evaluation against a quadrature that costs ten milliseconds.
SIZES = (64, 128, 256, 512, 1024, 2048, 4096, 8192)

#: Exposures spanning the regimes the studies run in: a fraction of a count expected, a
#: handful, and a few hundred.
EXPOSURES = (0.5, 4.0, 40.0)

#: Beliefs as ``(name, order, prior mean, index of dispersion)``. The default of
#: ``GIGPoisson``, and the heavy and mild shapes of the shape-scarcity study.
BELIEFS = (("default", -0.5, 1.0, 2.0),
           ("heavy", -2.0, 1.0, 4.0),
           ("mild", -0.5, 0.1, 1.0))


def repeats_for(n, budget=24 * 1024 * 1024):
    """Repeats at this size, holding the work per configuration roughly fixed."""
    return int(max(6, min(24, budget // (n * n))))


def save_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("  wrote {} ({} rows)".format(os.path.relpath(path, HERE), len(rows)))


def belief_with(order, mean, phi):
    """The GIG of this order whose mean is ``mean`` and whose ``V/E`` is ``phi``."""
    lo, hi = 1e-4, 1e4
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        m, v = gig_moments(gig_from_mean(order, mean, mid))
        if v / m > phi:
            lo = mid
        else:
            hi = mid
    law = gig_from_mean(order, mean, np.sqrt(lo * hi))
    m, v = gig_moments(law)
    if abs(v / m - phi) > 1e-3 * phi:
        raise ValueError("V/E={} unreachable at order {} mean {} (got {:.4f})"
                         .format(phi, order, mean, v / m))
    return law


def belief_at_most(order, mean, phi):
    """As above, backing the dispersion off until it is attainable at this order."""
    for _ in range(24):
        try:
            return belief_with(order, mean, phi)
        except ValueError:
            phi *= 0.7
    return gig_from_mean(order, mean, 1.0)


def timed(fn, repeats=3):
    """Median wall-clock of ``fn`` in milliseconds, after a warm-up call."""
    fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


# --------------------------------------------------------------------------

def study_accuracy():
    model = GIGPoisson()
    rows = []
    for name, order, mean, phi in BELIEFS:
        bel = belief_with(order, mean, phi)
        for f in EXPOSURES:
            exact = model.eig(bel, f)
            ms_exact = timed(lambda: model.eig(bel, f), repeats=15)
            rows.append((name, f, "quadrature", 0, 0, exact, exact, 0.0, 0.0, ms_exact))
            line = []
            for n in SIZES:
                reps = repeats_for(n)
                rng = np.random.default_rng(SEED)
                vals = np.asarray([nmc_eig(model, bel, f, rng, n)
                                   for _ in range(reps)], dtype=float)
                rng_t = np.random.default_rng(SEED + 1)
                ms = timed(lambda: nmc_eig(model, bel, f, rng_t, n), repeats=3)
                bias = float(vals.mean() - exact)
                sd = float(vals.std(ddof=1))
                rows.append((name, f, "nmc", n, n, exact, float(vals.mean()),
                             bias, float(np.sqrt(bias ** 2 + sd ** 2)), ms))
                line.append("%d:%.3f" % (n, np.sqrt(bias ** 2 + sd ** 2)))
            print("    {:<8} f={:<5g} exact={:.5f} ({:.1f} ms)   rmse {}"
                  .format(name, f, exact, ms_exact, " ".join(line)))
    return rows


def study_bias():
    model = GIGPoisson()
    outer = (256, 1024, 4096, 16384)
    inner = (64, 256, 1024, 4096)
    rows = []
    for name, order, mean, phi in BELIEFS[:2]:
        bel = belief_with(order, mean, phi)
        f = 4.0
        exact = model.eig(bel, f)
        for m_in in inner:
            got = []
            for n_out in outer:
                reps = repeats_for(max(n_out, m_in), budget=16 * 1024 * 1024)
                rng = np.random.default_rng(SEED)
                vals = np.asarray([nmc_eig(model, bel, f, rng, n_out, m_in)
                                   for _ in range(reps)], dtype=float)
                rows.append((name, f, n_out, m_in, exact, float(vals.mean()),
                             float(vals.mean() - exact),
                             float(vals.std(ddof=1) / np.sqrt(reps)),
                             float(vals.std(ddof=1))))
                got.append("%+.4f" % (vals.mean() - exact))
            print("    {:<8} inner={:<5d} bias at outer {}: {}"
                  .format(name, m_in, outer, " ".join(got)))
    return rows


def study_decisions(n_beliefs=40, n_contexts=5, sizes=(64, 256, 1024, 4096)):
    """Rank a grid of candidate actions by exact and by estimated EIG per unit cost.

    The belief sets and the exact ranking are built once and shared across sample sizes, so
    the comparison is paired and the exact quadratures are not paid for four times.
    """
    model = GIGPoisson()
    grid_f = np.array([0.5, 2.0, 8.0, 32.0])
    rng0 = np.random.default_rng(SEED)

    problems = []
    for _ in range(n_beliefs):
        beliefs = []
        for _ in range(n_contexts):
            order = float(rng0.uniform(-2.5, -0.3))
            mean = float(np.exp(rng0.uniform(np.log(0.1), np.log(3.0))))
            phi = float(rng0.uniform(0.5, 4.0))
            beliefs.append(belief_at_most(order, mean, phi))
        ex = np.array([[model.eig(b, f) / f for f in grid_f] for b in beliefs])
        problems.append((beliefs, ex, int(rng0.integers(1 << 30))))

    rows = []
    for n in sizes:
        flips, regrets = 0, []
        for beliefs, ex, seed in problems:
            rng = np.random.default_rng(seed)
            es = np.array([[nmc_eig(model, b, f, rng, n) / f for f in grid_f]
                           for b in beliefs])
            i_ex = np.unravel_index(np.argmax(ex), ex.shape)
            i_es = np.unravel_index(np.argmax(es), es.shape)
            if i_ex != i_es:
                flips += 1
                regrets.append(float(ex[i_ex] - ex[i_es]))
        rows.append((n, n_beliefs, n_contexts, len(grid_f), flips / float(n_beliefs),
                     float(np.mean(regrets)) if regrets else 0.0,
                     float(np.max(regrets)) if regrets else 0.0))
        print("    n={:<6d} argmax differs on {:5.1f}% of problems, mean cost "
              "{:.5f} nats per unit exposure"
              .format(n, 100.0 * flips / n_beliefs,
                      float(np.mean(regrets)) if regrets else 0.0))
    return rows


def main():
    rd = os.path.join(HERE, "results")
    os.makedirs(rd, exist_ok=True)
    t0 = time.time()

    print("A. accuracy against wall-clock")
    save_csv(os.path.join(rd, "accuracy.csv"),
             ["belief", "exposure", "method", "n_outer", "n_inner", "exact", "mean",
              "bias", "rmse", "ms"], study_accuracy())

    print("\nB. where the error lives  [{:.0f}s]".format(time.time() - t0))
    save_csv(os.path.join(rd, "bias.csv"),
             ["belief", "exposure", "n_outer", "n_inner", "exact", "mean", "bias",
              "sem", "sd"], study_bias())

    print("\nC. does the error reach the decision  [{:.0f}s]".format(time.time() - t0))
    save_csv(os.path.join(rd, "decisions.csv"),
             ["n", "n_beliefs", "n_contexts", "n_exposures", "flip_rate",
              "mean_regret", "max_regret"], study_decisions())

    print("\nDone in {:.0f}s.".format(time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
