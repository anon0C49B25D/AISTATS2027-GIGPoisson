"""Three refinements the first pass showed were needed.

**The quadrature needs an honest error floor.** Plotting it against the estimator on a log
error axis means claiming a number for its own accuracy. The two settings that control it
are the node count of the Gauss-Legendre rule over the mixing law and the tail tolerance of
the support sum, so the floor is measured by tightening both and seeing how far the value
moves. ``results/floor.csv``.

**The heavy tail needs a distribution, not a root-mean-square.** At order ``-2`` and an
exposure of forty the estimator's error is driven by rare outer draws that land where the
shared inner sample has almost no mass, so the mean and the RMSE are set by the tail of the
error rather than by its centre, and both bounce around with the seed. Reported here as
quantiles over many repeats. ``results/heavytail.csv``.

**The decision study needs more problems.** Forty leaves the flip rate with a standard error
of five points, which is the same size as the effect at the largest sample. ``results/
decisions.csv`` is rewritten with three times as many.

Run: python experiments/synthetic/nmc-cost/refine.py
"""

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

import methods.countmodels as cm                                      # noqa: E402
from methods.countmodels import GIGPoisson                            # noqa: E402
from nmc import nmc_eig                                               # noqa: E402
from run import BELIEFS, EXPOSURES, belief_with, save_csv, study_decisions  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0


def eig_at(model, belief, f, nodes, tail_tol):
    """``model.eig`` with the two quadrature settings overridden."""
    old_nodes, old_drop = cm.QUAD_NODES, cm.LOG_DENSITY_DROP
    cm.QUAD_NODES = nodes
    cm.LOG_DENSITY_DROP = 75.0 if nodes <= 64 else 120.0
    try:
        y = model.support(belief, f, tail_tol=tail_tol)
        lp = model.logpmf(belief, f, y)
        p = np.exp(lp)
        h_pred = float(-np.sum(p * lp))
        return h_pred - model.aleatoric_entropy(belief, f)
    finally:
        cm.QUAD_NODES, cm.LOG_DENSITY_DROP = old_nodes, old_drop


def study_floor():
    """How far does the exact value move when the quadrature is tightened?"""
    model = GIGPoisson()
    rows = []
    for name, order, mean, phi in BELIEFS:
        bel = belief_with(order, mean, phi)
        for f in EXPOSURES:
            base = model.eig(bel, f)
            ref = eig_at(model, bel, f, 256, 1e-15)
            mids = [eig_at(model, bel, f, 128, 1e-12),
                    eig_at(model, bel, f, 64, 1e-12)]
            rows.append((name, f, base, ref, abs(base - ref),
                         max(abs(m - ref) for m in mids)))
            print("    {:<8} f={:<5g} shipped={:.10f} tight={:.10f} |diff|={:.2e}"
                  .format(name, f, base, ref, abs(base - ref)))
    return rows


def study_heavytail(repeats=120):
    """Quantiles of the absolute error, where the mean is set by rare draws."""
    model = GIGPoisson()
    rows = []
    for name, order, mean, phi in (("heavy", -2.0, 1.0, 4.0),
                                   ("default", -0.5, 1.0, 2.0)):
        bel = belief_with(order, mean, phi)
        for f in (4.0, 40.0):
            exact = model.eig(bel, f)
            for n in (256, 1024, 4096):
                rng = np.random.default_rng(SEED)
                err = np.abs(np.asarray(
                    [nmc_eig(model, bel, f, rng, n) for _ in range(repeats)]) - exact)
                q = np.percentile(err, [50, 75, 90, 99])
                rows.append((name, f, n, repeats, exact, float(q[0]), float(q[1]),
                             float(q[2]), float(q[3]), float(err.max()),
                             float(err.mean())))
                print("    {:<8} f={:<4g} n={:<5d} median={:.4f} p90={:.4f} "
                      "p99={:.4f} max={:.4f}"
                      .format(name, f, n, q[0], q[2], q[3], err.max()))
    return rows


def main():
    rd = os.path.join(HERE, "results")
    t0 = time.time()

    print("Quadrature error floor")
    save_csv(os.path.join(rd, "floor.csv"),
             ["belief", "exposure", "shipped", "tightened", "abs_diff", "max_mid_diff"],
             study_floor())

    print("\nHeavy-tail error distribution  [{:.0f}s]".format(time.time() - t0))
    save_csv(os.path.join(rd, "heavytail.csv"),
             ["belief", "exposure", "n", "repeats", "exact", "p50", "p75", "p90",
              "p99", "max", "mean"], study_heavytail())

    print("\nDecisions, 120 problems  [{:.0f}s]".format(time.time() - t0))
    save_csv(os.path.join(rd, "decisions.csv"),
             ["n", "n_beliefs", "n_contexts", "n_exposures", "flip_rate",
              "mean_regret", "max_regret"], study_decisions(n_beliefs=120))

    print("\nDone in {:.0f}s.".format(time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
