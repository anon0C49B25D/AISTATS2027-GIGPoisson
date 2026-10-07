"""A compound-gamma prior on a Poisson rate, computed exactly on a grid.

    lam | beta ~ Gamma(r, beta),   beta ~ Gamma(A, B)   =>   lam ~ BetaPrime(r, A, scale B)

The hierarchical gamma-Poisson model with the gamma's rate hyperparameter integrated out
\\citep{christiansen1997hierarchical}: three parameters, like the GIG, and a power-law tail,
``p(lam) ~ lam^(-A-1)``, which no single gamma has. Its predictive is the generalised Waring
law. It is the in-regime rival to the GIG: if the GIG's advantage came from having a
power-law tail at all, this prior would share it.

**Computation.** It is not conjugate to the Poisson, but its density in ``s = log lam``,

    r s - (r + A) log(1 + e^s / B),

is concave, so the posterior ``prior(s) exp(T s - F e^s)`` is log-concave and unimodal, and
every quantity is a quadrature on a grid in ``s``, as for the exact log-skew-normal baseline,
whose predictive and entropy routines this class reuses (including its treatment of the far
tail of the predictive, which a power-law prior needs). The grid runs from the posterior mode
out to where the log density has dropped by :data:`DROP` nats, which for a power-law tail is
much further than a fixed number of standard deviations.

**Limits.** As ``A -> inf`` with ``r B / A`` fixed it tends to a gamma; as ``r -> inf`` with
``r B`` fixed it tends to an inverse gamma, ``lam ~ InvGamma(A, r B)``, which is the GIG of
order ``-A`` with ``b = 0``. Fitted to twenty catalogued rates, maximum likelihood can run to
that second limit, so the shape is capped at :data:`R_MAX`, where the prior is numerically
indistinguishable from its limit, and the tail index is held above one so that the prior mean
is finite. ``fits`` records every fit and whether a constraint was active.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, expit, logsumexp
from scipy.stats import gamma as gamma_dist

from .logskewnormal import LogSkewNormalQuad, _predictive

__all__ = ["CompoundGammaPoisson", "betaprime_logpdf"]

#: Largest shape: beyond it the prior is its inverse-gamma limit to within rounding.
R_MAX = 1e4
#: Largest tail index: beyond it the prior is its gamma limit.
A_MAX = 1e4
#: Nats below the mode at which the grid stops.
DROP = 60.0


def betaprime_logpdf(x, r, A, B):
    """Log density of the beta-prime law of shapes ``(r, A)`` and scale ``B`` at ``x > 0``."""
    lx = np.log(x)
    lb = np.log(B)
    return ((r - 1.0) * lx - r * lb - (r + A) * np.logaddexp(0.0, lx - lb)
            - betaln(r, A))


def _log_target(s, prior, T, F):
    r, A, B = prior
    return (r + T) * s - (r + A) * np.logaddexp(0.0, s - np.log(B)) - F * np.exp(s)


def _derivs(s, prior, T, F):
    r, A, B = prior
    p = expit(s - np.log(B))
    e = F * np.exp(s)
    return r + T - (r + A) * p - e, -(r + A) * p * (1.0 - p) - e


def _mode(prior, T, F):
    """Mode and curvature of the concave log posterior in ``s``: damped Newton."""
    r, A, B = prior
    s = float(np.log(B) + np.log((r + T) / A))
    if F > 0.0:
        s = min(s, float(np.log((r + T) / F)))
    for _ in range(200):
        g, h = _derivs(s, prior, T, F)
        step = -g / h
        f0 = _log_target(s, prior, T, F)
        t = 1.0
        while t > 1e-10 and not (_log_target(s + t * step, prior, T, F) >= f0 - 1e-12):
            t *= 0.5
        s += t * step
        if abs(t * step) < 1e-11:
            break
    return s, float(-_derivs(s, prior, T, F)[1])


def _edge(prior, T, F, mode, top, direction, sd):
    """Where the log posterior has dropped :data:`DROP` nats, on one side of the mode."""
    near, far, step = mode, mode + direction * sd, sd
    while _log_target(far, prior, T, F) > top - DROP:
        near, step = far, 2.0 * step
        far = mode + direction * step
    for _ in range(60):
        mid = 0.5 * (near + far)
        near, far = (mid, far) if _log_target(mid, prior, T, F) > top - DROP else (near, mid)
    return far


def compound_grid(prior, T, F, f, ymax):
    """``(s, log w)``: a uniform grid in ``s`` over the posterior, resolving the Poisson factor
    of a count up to ``ymax`` at exposure ``f``, and its normalised log weights."""
    mode, curv = _mode(prior, T, F)
    sd = 1.0 / np.sqrt(max(curv, 1e-300))
    top = _log_target(mode, prior, T, F)
    lo = _edge(prior, T, F, mode, top, -1.0, sd)
    hi = _edge(prior, T, F, mode, top, 1.0, sd)
    h = min(sd / 20.0, 0.25 / np.sqrt(float(ymax) + 1.0))
    n = int(np.clip(np.ceil((hi - lo) / h) + 1, 401, 400_001))
    s = np.linspace(lo, hi, n)
    lw = _log_target(s, prior, T, F)
    lw -= logsumexp(lw)
    keep = lw > lw.max() - DROP
    s, lw = s[keep], lw[keep]
    return s, lw - logsumexp(lw)


class CompoundGammaPoisson(LogSkewNormalQuad):
    """Poisson likelihood, compound-gamma (beta-prime) prior, every quantity by quadrature."""

    name = "compoundgamma-poisson"
    label = "compound-gamma-Poisson (exact quadrature)"
    conjugate = False

    def __init__(self, tail_tol=1e-6):
        super(CompoundGammaPoisson, self).__init__(tail_tol=tail_tol)
        #: One record per fit: the parameters and whether a constraint was active.
        self.fits = []

    def _belief(self, r, A, B):
        return {"prior": (float(r), float(A), float(B)), "T": 0.0, "F": 0.0}

    def prior_from_moments(self, mean, var):
        """The gamma limit matched to two moments (no third moment is available)."""
        r = mean * mean / var
        A = A_MAX
        return self._belief(r, A, mean * (A - 1.0) / r)

    def fit_population(self, rates):
        x = np.asarray(rates, dtype=float)

        def nll(th):
            r, A, B = np.exp(th[0]), 1.0 + np.exp(th[1]), np.exp(th[2])
            if r > R_MAX or A > A_MAX:
                return 1e300
            v = -float(np.sum(betaprime_logpdf(x, r, A, B)))
            return v if np.isfinite(v) else 1e300

        shape = gamma_dist.fit(x, floc=0.0)[0]
        best = None
        for r0 in (shape, 0.3, 1.0, 3.0, 30.0):
            for A0 in (1.5, 3.0, 10.0, 100.0):
                th0 = [np.log(r0), np.log(A0 - 1.0), np.log(x.mean() * (A0 - 1.0) / r0)]
                res = minimize(nll, th0, method="Nelder-Mead",
                               options=dict(maxiter=4000, xatol=1e-9, fatol=1e-11))
                if best is None or res.fun < best.fun:
                    best = res
        r, A, B = np.exp(best.x[0]), 1.0 + np.exp(best.x[1]), np.exp(best.x[2])
        self.fits.append(dict(r=float(r), A=float(A), B=float(B), loglik=-float(best.fun),
                              shape_capped=bool(r > 0.99 * R_MAX),
                              tail_capped=bool(A > 0.99 * A_MAX),
                              mean_bound=bool(A < 1.0 + 1e-3)))
        return self._belief(r, A, B)

    def _grid(self, belief, f=1.0, ymax=0.0):
        return compound_grid(belief["prior"], belief["T"], belief["F"], f, ymax)

    def logpmf(self, belief, f, y):
        y = np.atleast_1d(np.asarray(y, dtype=float))
        s, lw = self._grid(belief, f, float(y.max()))
        return _predictive(s, lw, f, y)

    def fisher_dopt(self, belief, f):
        m, v = self.rate_moments(belief)
        return float(np.log1p(f * v / max(m, 1e-12)))
