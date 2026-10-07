"""Where does the third parameter pay in the tail? An illustration of Lemma (Sichel tail).

The lemma says that, at matched predictive mean and variance, the Sichel tail decays at a
slower geometric rate than the negative binomial, and that the gap between the two rates
depends on the accumulated exposure alone, shrinking as O(1/F). This study shows the same
picture in a tail-sensitive proper score.

Setting. One context. The true mixing law is a GIG at the predictive moments of Table 1
(mean 6, variance 48 at unit exposure), swept over its order up to the ceiling where it
becomes the gamma. Two models are given the truth's first two moments:

- ``sichel``: the GIG-Poisson with the true hyperparameters (well specified);
- ``negbinom``: the gamma-Poisson with the moment-matched gamma prior.

Every gamma prior is the same, whatever the true order, because the moments are held
fixed. A count ``T`` is observed at total exposure ``F`` (the posterior depends on nothing
else), both models are updated, and the next count at unit exposure is scored.

Score. The censored likelihood score of Diks, Panchenko and van Dijk (2011) on the region
``y >= t``,

    CL(p, y) = -[ 1{y >= t} log p(y) + 1{y < t} log P(Y < t) ] ,

which is proper and sees nothing of the forecast below ``t`` except its total mass. The
Sichel arm is the exact Bayesian predictive, so given ``T`` the next count is distributed
as its posterior predictive, and the expected score difference is exactly the divergence
between the two censored forecasts. That is summed over the support, so the only Monte
Carlo is over ``T`` (none at ``F = 0``). Scoring against a drawn rate instead gives the same
mean with a conditional variance that swamps a gain of a few millinats.

The gain is reported in nats per forecast and per exceedance, the latter divided by the
probability of ``y >= t``, since a 99th-percentile score is diluted by the 99% of forecasts
that land below it. The thresholds are quantiles of the matched negative binomial prior
predictive, which is the same at every order.

Illustration only: the Sichel arm knows the true hyperparameters, so this measures what
the third parameter can buy, not what an estimated one does.

Writes ``results/``. Figures come from ``visualize.py``.

Run: python experiments/synthetic/tail-scores/run.py
"""

import csv
import os
import sys
import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import nbinom

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from methods.gigpoisson import (                                      # noqa: E402
    gig_from_mean, gig_moments, gig_sample, sichel_logpmf, sichel_support)

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0
N_DRAWS = 1500          # accumulated counts per (order, exposure)

#: Predictive mass left outside the scored support.
TAIL_TOL = 1e-10

#: Mixing-law moments: predictive mean 6 and variance 48 at unit exposure, as in Table 1.
MEAN, VAR = 6.0, 42.0

#: Orders as fractions of the ceiling ``MEAN^2 / VAR``, the matched gamma's shape.
FRACTIONS = (-3.0, -2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.9)

#: Accumulated exposure before the scored count, in units of the scored count's exposure.
EXPOSURES = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)

#: Thresholds, as quantiles of the matched negative binomial prior predictive.
LEVELS = (0.95, 0.99, 0.999)


def gig_for_moments(order, mean, var, tol=1e-9):
    """The GIG of this order with the requested mean and variance, by bisection on omega."""
    lo, hi = 1e-12, 1e6
    if gig_moments(gig_from_mean(order, mean, lo))[1] < var * (1.0 - tol):
        raise ValueError("order %.4f cannot reach variance %.4f" % (order, var))
    for _ in range(300):
        mid = np.sqrt(lo * hi)
        if gig_moments(gig_from_mean(order, mean, mid))[1] > var:
            lo = mid
        else:
            hi = mid
    return gig_from_mean(order, mean, np.sqrt(lo * hi))


def censored_kl(ls, ln, t):
    """``KL`` between the censored forecasts, and the true exceedance probability.

    ``ls`` is the true (Sichel) log mass function and ``ln`` the negative binomial's, both
    on ``0..ymax``. The censored forecast keeps ``p(y)`` for ``y >= t`` and lumps the rest.
    """
    ps = np.exp(ls[t:])
    below_s = logsumexp(ls[:t])
    below_n = logsumexp(ln[:t])
    kl = np.dot(ps, ls[t:] - ln[t:]) + np.exp(below_s) * (below_s - below_n)
    return kl, ps.sum()


def scores_given_count(truth, r, beta, T, F, thresholds):
    """Per-threshold ``(gain, P(Y >= t))`` for the next count, given ``T`` at exposure ``F``."""
    post = {"alpha": truth["alpha"] + T, "a": truth["a"], "b": truth["b"] + 2.0 * F}
    y = sichel_support(post, 1.0, tail_tol=TAIL_TOL)
    if len(y) <= max(thresholds):
        y = np.arange(max(thresholds) + 1, dtype=float)
    ls = sichel_logpmf(y, post, 1.0)
    ln = nbinom.logpmf(y, r + T, (beta + F) / (beta + F + 1.0))
    return [censored_kl(ls, ln, t) for t in thresholds]


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    r = MEAN * MEAN / VAR
    beta = MEAN / VAR
    thresholds = [int(nbinom.ppf(lv, r, beta / (beta + 1.0))) for lv in LEVELS]
    print("matched gamma: shape %.4f rate %.4f; thresholds %s at %s"
          % (r, beta, thresholds, LEVELS))

    rows, t0 = [], time.time()
    for frac in FRACTIONS:
        order = frac * r
        truth = gig_for_moments(order, MEAN, VAR)
        lams = gig_sample(truth, N_DRAWS, rng)
        for F in EXPOSURES:
            counts = rng.poisson(F * lams) if F > 0 else np.zeros(1, dtype=int)
            cache = {}
            out = np.zeros((len(counts), len(LEVELS), 2))
            for i, T in enumerate(counts):
                if T not in cache:
                    cache[T] = scores_given_count(truth, r, beta, int(T), F, thresholds)
                out[i] = cache[T]
            q_s = 2.0 / (truth["b"] + 2.0 * F + 2.0)
            q_n = 1.0 / (beta + F + 1.0)
            n = len(counts)
            for j, (lv, t) in enumerate(zip(LEVELS, thresholds)):
                g, p = out[:, j, 0], out[:, j, 1]
                se = g.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
                rows.append([frac, order, truth["b"], F, lv, t, g.mean(), se, p.mean(),
                             g.mean() / p.mean(), np.log(q_s / q_n)])
        print("  order %+.3f done (%.0fs)" % (order, time.time() - t0))

    path = os.path.join(HERE, "results", "summary.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["order_over_max", "order", "b", "exposure", "level", "threshold",
                    "cl_gain", "cl_gain_se", "p_exceed", "cl_gain_per_exceedance",
                    "log_rate_gap"])
        w.writerows(rows)
    print("  wrote results/summary.csv (%d rows)" % len(rows))


if __name__ == "__main__":
    main()
