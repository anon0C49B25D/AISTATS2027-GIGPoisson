"""Nested Monte Carlo estimation of the expected information gain.

The estimator the design literature uses when the predictive is not available in closed
form \\citep{rainforth2018nesting}. Draw ``N`` outer pairs from the joint,

    lam_i ~ p(lam),   y_i ~ p(y | lam_i, u),

and estimate the evidence at each ``y_i`` by an inner average over ``M`` fresh rate draws:

    EIG ~= (1/N) sum_i [ log p(y_i | lam_i) - log( (1/M) sum_j p(y_i | lam'_j) ) ].

Two properties matter for the comparison in ``run.py``. The inner average sits inside a
logarithm, so by Jensen the second term is biased low and the estimate is biased **up**, at
order ``1/M``. And that bias is a function of ``M`` alone: raising ``N`` drives the variance
down and leaves the bias exactly where it was. Section 3.3 of the manuscript computes the
same quantity by two deterministic quadratures, which have neither.

The inner sample is shared across outer draws, as in the correctness gates of the two
allocation studies, because drawing a fresh inner set per outer draw costs ``N`` times more
for a variance reduction that does not change the picture. The outer loop is chunked so that
memory stays at ``chunk * M`` rather than ``N * M``; at ``N = M = 10^4`` the unchunked matrix
alone is eight hundred megabytes.
"""

import numpy as np
from scipy.special import gammaln, logsumexp, xlogy

__all__ = ["nmc_eig", "nmc_cost_model"]


#: Largest inner-by-outer block held at once, in floats. Four million doubles is about
#: thirty megabytes, so the chunk shrinks as the inner sample grows and the estimator stays
#: usable at inner sizes where the full matrix would not fit.
MAX_BLOCK = 4_000_000


def nmc_eig(model, belief, f, rng, n_outer, n_inner=None, chunk=None):
    """Nested Monte Carlo estimate of ``EIG(u)`` under ``belief`` at exposure ``f``.

    ``n_inner`` defaults to ``n_outer``, which is the usual diagonal setting. Passing the
    two separately is what isolates the nesting bias from the sampling noise.
    """
    if f <= 0.0:
        return 0.0
    m_in = int(n_outer if n_inner is None else n_inner)
    if chunk is None:
        chunk = int(max(32, min(4096, MAX_BLOCK // max(m_in, 1))))
    lam_in = np.asarray(model.sample_rate(belief, m_in, rng), dtype=float)
    mu_in = np.maximum(f * lam_in, 1e-300)
    log_mu_in = np.log(mu_in)
    log_m = np.log(float(m_in))

    total, done = 0.0, 0
    while done < n_outer:
        b = int(min(chunk, n_outer - done))
        lam = np.asarray(model.sample_rate(belief, b, rng), dtype=float)
        mu = np.maximum(f * lam, 1e-300)
        y = rng.poisson(mu).astype(float)
        # log p(y_i | lam_i), the outer term
        ll_out = -mu + xlogy(y, mu) - gammaln(y + 1.0)
        # log p(y_i | lam'_j) for every inner draw, then the inner average in log space
        ll_in = (-mu_in[None, :] + xlogy(y[:, None], mu_in[None, :])
                 - gammaln(y + 1.0)[:, None])
        total += float(np.sum(ll_out - (logsumexp(ll_in, axis=1) - log_m)))
        done += b
    return total / float(n_outer)


def nmc_cost_model(n_outer, n_inner=None):
    """Work done, in units of inner-outer products, for reporting alongside wall-clock."""
    m_in = n_outer if n_inner is None else n_inner
    return float(n_outer) * float(m_in)
