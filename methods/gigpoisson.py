"""Generalised inverse Gaussian mixing law, its Poisson mixture, and their numerics.

The mathematical core shared by every experiment in this project. Nothing here knows about
any particular sensing problem: a problem supplies an exposure map and a set of contexts,
and inherits every closed form below unchanged.

Conjugacy. Writing the mixing density as ``lam^(alpha-1) exp{-(a/lam + b lam)/2}``, one
observation of ``y ~ Poisson(f lam)`` sends

    GIG(alpha, a, b)  ->  GIG(alpha + y, a, b + 2 f) ,

so the exposure enters the second scale parameter additively. The marginal is the Sichel
law,

    log p(y | u) = y log f - lgamma(y + 1)
                   + logZ(alpha + y, a, b + 2 f) - logZ(alpha, a, b) ,
    logZ(alpha, a, b) = log 2 + (alpha/2)(log a - log b) + log K_alpha(sqrt(ab)) ,

whose normalising constants are modified Bessel functions of the second kind.

Why the family is worth the Bessel functions. Not free overdispersion: a gamma mixing law
already gives a negative binomial predictive with a free mean and a free variance. What
the GIG adds is a third parameter, so that with the predictive mean and variance both held
fixed there is still freedom to move mass between the centre of the distribution and its
tail. The gamma is the ``omega -> 0`` boundary of the parameterisation used here, so the
negative binomial sits inside this family as a limiting case and
:func:`negbinom_limit_error` measures the distance to it.

Numerics. ``K_v(z)`` overflows double precision at moderate order, and the posterior order
is ``alpha + sum(y)``, which grows without bound as data arrive. Everything is therefore
done in log space, and the support sum uses an upward recurrence on the order in which
every combined term is positive, so no cancellation is possible. See :func:`log_besselk`
and :func:`log_besselk_sequence`.
"""

import numpy as np
from scipy.special import gammaln, kve, logsumexp
from scipy.stats import geninvgauss, nbinom

__all__ = [
    "log_besselk", "log_besselk_sequence", "gig_logpdf", "gig_sample",
    "gig_moments", "gig_from_mean", "log_gig_norm", "sichel_logpmf",
    "sichel_support", "predictive_moments", "posterior", "negbinom_limit_error",
]

#: Order above which the exponentially scaled ``kve`` route overflows or loses digits and
#: the uniform asymptotic expansion takes over. Justified empirically by
#: :func:`_check_besselk`, which compares both routes against 40-digit references.
OLVER_FROM = 45.0

#: Predictive mass left outside a truncated support.
TAIL_TOL = 1e-13

#: Run length above which the orders of a support sum are evaluated directly rather than by
#: the recurrence. Neither route is uniformly better. Against 30-digit references the
#: recurrence is exact for the first few dozen orders, but its rounding compounds along the
#: run: at 10^3 terms it is out by 3e-12, at 5*10^4 by 6e-9 and at 2*10^5 by 3e-7, where the
#: uniform asymptotic used by :func:`log_besselk` is still exact to the last digit, being
#: a fixed-cost expansion at each order rather than an accumulation. The recurrence is also
#: a Python loop and so an order of magnitude slower at those lengths. Short runs therefore
#: take the recurrence, which is cancellation-free where it is used, and long runs take the
#: direct route, which does not accumulate.
SEQUENCE_MAX = 512


# --------------------------------------------------------------------------
# Modified Bessel function of the second kind, in log space
# --------------------------------------------------------------------------

def _log_besselk_scaled(v, z):
    """``log K_v(z)`` through the exponentially scaled ``kve``. Fails at large order."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(kve(v, z)) - z


def _olver_series(t, v):
    """``log`` of the Debye polynomial sum in the uniform asymptotic expansion.

    Terms alternate in sign, so this is the one place in the module where a cancellation
    occurs. It is harmless: the sum is dominated by its leading ``1`` for every order at
    which this route is used, the corrections being of size ``1/v``.
    """
    t2 = t * t
    u1 = t * (3.0 - 5.0 * t2) / 24.0
    u2 = t2 * (81.0 - 462.0 * t2 + 385.0 * t2 * t2) / 1152.0
    u3 = (t2 * t) * (30375.0 - 369603.0 * t2 + 765765.0 * t2 * t2
                     - 425425.0 * t2 * t2 * t2) / 414720.0
    u4 = (t2 * t2) * (4465125.0 - 94121676.0 * t2 + 349922430.0 * t2 * t2
                      - 446185740.0 * t2 * t2 * t2
                      + 185910725.0 * t2 * t2 * t2 * t2) / 39813120.0
    return np.log1p(-u1 / v + u2 / v ** 2 - u3 / v ** 3 + u4 / v ** 4)


def _log_besselk_olver(v, z):
    """``log K_v(z)`` by Olver's uniform asymptotic expansion in the order (DLMF 10.41.4).

    Uniform in the argument, so it holds for every ``z`` at large ``v``, which is what the
    posterior order ``alpha + sum(y)`` eventually requires.
    """
    x = z / v
    w = np.sqrt(1.0 + x * x)
    eta = w + np.log(x / (1.0 + w))
    return (0.5 * np.log(np.pi / (2.0 * v)) - v * eta - 0.5 * np.log(w)
            + _olver_series(1.0 / w, v))


def log_besselk(v, z):
    """``log K_v(z)`` for ``z > 0`` and any real order, without overflow.

    ``K`` is even in the order, so the sign of ``v`` is discarded first.
    """
    v = np.abs(np.asarray(v, dtype=float))
    z = np.asarray(z, dtype=float)
    v, z = np.broadcast_arrays(v, z)
    out = np.empty(v.shape, dtype=float)
    big = v > OLVER_FROM
    if np.any(~big):
        out[~big] = _log_besselk_scaled(v[~big], z[~big])
    if np.any(big):
        out[big] = _log_besselk_olver(v[big], z[big])
    # kve underflows to zero once the order is large relative to the argument but still
    # below the crossover; the asymptotic form covers that corner too.
    bad = ~np.isfinite(out)
    if np.any(bad):
        out[bad] = _log_besselk_olver(np.maximum(v[bad], 1e-12), z[bad])
    return out if out.ndim else float(out)


def log_besselk_sequence(v0, z, n):
    """``[log K_{v0}(z), ..., log K_{v0+n}(z)]`` by upward recurrence in log space.

    The recurrence ``K_{v+1} = K_{v-1} + (2v/z) K_v`` adds two positive quantities, so
    accumulating it through ``logaddexp`` introduces no cancellation at any order, however
    large. This is the arrangement the support sum of the Sichel law needs: the orders it
    requires are exactly ``v0 + y`` for ``y`` running over the support, and evaluating each
    independently would both cost more and, above the crossover, inherit the asymptotic
    expansion's error at every term instead of only at the first two.
    """
    z = float(z)
    out = np.empty(int(n) + 1, dtype=float)
    # Seed directly until the order is positive, since the recurrence coefficient 2v/z is
    # not usable at v <= 0. For any prior order above -1 this is the first one or two terms.
    j = 0
    while j <= n and (v0 + j) <= 0.0:
        out[j] = log_besselk(v0 + j, z)
        j += 1
    if j <= n:
        out[j] = log_besselk(v0 + j, z)
    if j + 1 <= n:
        out[j + 1] = log_besselk(v0 + j + 1, z)
    for k in range(j + 2, int(n) + 1):
        v = v0 + k - 1.0
        out[k] = np.logaddexp(out[k - 2], np.log(2.0 * v / z) + out[k - 1])
    return out


# --------------------------------------------------------------------------
# The generalised inverse Gaussian mixing law
# --------------------------------------------------------------------------

def log_gig_norm(alpha, a, b):
    """``log`` of ``int lam^(alpha-1) exp{-(a/lam + b lam)/2} dlam``."""
    return (np.log(2.0) + 0.5 * alpha * (np.log(a) - np.log(b))
            + log_besselk(alpha, np.sqrt(a * b)))


def gig_logpdf(lam, prior):
    alpha, a, b = prior["alpha"], prior["a"], prior["b"]
    lam = np.asarray(lam, dtype=float)
    return ((alpha - 1.0) * np.log(lam) - 0.5 * (a / lam + b * lam)
            - log_gig_norm(alpha, a, b))


def _scipy_shape(prior):
    """``(alpha, omega, eta)`` for ``scipy.stats.geninvgauss``.

    SciPy calls the order ``p`` and the concentration ``b``, so the call reads
    ``geninvgauss(alpha, b=omega, scale=eta)``. Its density is
    ``x^(alpha-1) exp{-omega(x + 1/x)/2}``, which matches the convention here after the
    scale ``eta = sqrt(a/b)``, with ``omega = sqrt(ab)``.
    """
    alpha, a, b = prior["alpha"], prior["a"], prior["b"]
    return alpha, float(np.sqrt(a * b)), float(np.sqrt(a / b))


def gig_sample(prior, n, rng):
    alpha, omega, eta = _scipy_shape(prior)
    return geninvgauss.rvs(alpha, omega, scale=eta, size=n, random_state=rng)


def gig_moments(prior):
    """``(mean, variance)`` of the mixing law, through ratios of Bessel functions.

    ``E[lam^n] = eta^n K_{alpha+n}(omega) / K_alpha(omega)``, evaluated as a difference
    of logs so that a large common factor never has to be represented.
    """
    alpha, omega, eta = _scipy_shape(prior)
    lk = log_besselk_sequence(alpha, omega, 2)
    m1 = eta * np.exp(lk[1] - lk[0])
    m2 = eta ** 2 * np.exp(lk[2] - lk[0])
    return float(m1), float(m2 - m1 ** 2)


def gig_from_mean(alpha, mean, omega):
    """Mixing law with the requested order, mean and concentration.

    The three parameters ``(alpha, a, b)`` are reparameterised as an order ``alpha``, a
    scale ``eta = sqrt(a/b)`` and a concentration ``omega = sqrt(ab)``. Fixing the mean
    then determines the scale in closed form,
    ``eta = mean * K_alpha(omega) / K_{alpha+1}(omega)``, so a site's brightness and its
    dispersion are set independently and no solver is needed.

    ``omega`` is the departure from the gamma: it vanishes in the gamma limit, where the
    predictive collapses to the negative binomial of the conjugate Poisson pair.
    """
    lk = log_besselk_sequence(alpha, float(omega), 1)
    eta = float(mean * np.exp(lk[0] - lk[1]))
    return {"alpha": float(alpha), "a": float(eta * omega), "b": float(omega / eta)}


# --------------------------------------------------------------------------
# The Sichel predictive and the conjugate update
# --------------------------------------------------------------------------

def posterior(prior, y, f):
    """One conjugate update: the order absorbs the count, the exposure the second scale."""
    return {"alpha": prior["alpha"] + float(y), "a": prior["a"],
            "b": prior["b"] + 2.0 * float(f)}


def sichel_support(prior, f, tail_tol=TAIL_TOL, cap=2_000_000):
    """Counts carrying all but ``tail_tol`` of the predictive mass.

    Sized from the predictive moments, then widened until the truncated mass is inside the
    tolerance, since a normal-tail estimate is not conservative for a heavy-tailed law.
    """
    mean, var = predictive_moments(prior, f)
    ymax = int(mean + 12.0 * np.sqrt(var) + 64.0)
    for _ in range(24):
        ymax = min(ymax, cap)
        y = np.arange(ymax + 1, dtype=float)
        if 1.0 - np.exp(logsumexp(sichel_logpmf(y, prior, f))) < tail_tol or ymax == cap:
            return y
        ymax *= 2
    return y


def sichel_logpmf(y, prior, f):
    """``log p(y | u)``, vectorised over the support.

    The run of orders ``alpha + y`` is obtained by whichever route is better at the length
    required, per :data:`SEQUENCE_MAX`.
    """
    alpha, a, b = prior["alpha"], prior["a"], prior["b"]
    y = np.atleast_1d(np.asarray(y, dtype=float))
    f = float(f)
    bf = b + 2.0 * f
    n = int(np.max(y))
    z = np.sqrt(a * bf)
    if n <= SEQUENCE_MAX:
        lk = log_besselk_sequence(alpha, z, n)[y.astype(np.int64)]
    else:
        lk = log_besselk(alpha + y, z)
    log_z_post = np.log(2.0) + 0.5 * (alpha + y) * (np.log(a) - np.log(bf)) + lk
    return (y * np.log(f) - gammaln(y + 1.0) + log_z_post - log_gig_norm(alpha, a, b))


def predictive_moments(prior, f):
    """``(mean, variance)`` of the count, by the law of total expectation and variance.

    The Fano factor is ``1 + f Var[lam] / E[lam]``, so the departure from equidispersion
    grows with the exposure: how overdispersed the data look is itself a function of the
    action, which is the reason the action matters here at all.
    """
    m, v = gig_moments(prior)
    return float(f * m), float(f * m + f * f * v)


def negbinom_limit_error(alpha, mean, omega, f, ymax=400):
    """Largest ``|log p_Sichel - log p_NB|`` over the support at concentration ``omega``.

    At ``omega -> 0`` with a positive order the mixing law tends to a gamma, so the
    predictive tends to the negative binomial of the conjugate Poisson pair. Reporting
    this distance shows that the boundary is where it is claimed to be, and how far a
    given site sits from it.
    """
    prior = gig_from_mean(alpha, mean, omega)
    y = np.arange(ymax + 1, dtype=float)
    lp = sichel_logpmf(y, prior, f)
    # Gamma limit: shape alpha, rate b/2, hence a negative binomial with shape alpha
    # and odds 2 f / b.
    s = 2.0 * f / prior["b"]
    lq = nbinom.logpmf(y, alpha, 1.0 / (1.0 + s))
    keep = np.exp(lp) > 1e-12
    return float(np.max(np.abs(lp[keep] - lq[keep])))


