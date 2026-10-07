"""Fitting a mixing law to measured counts, by marginal likelihood.

The comparison studies fit each prior to a catalogue of *rates* (``fit_population``). A
natural count data set has no rates, only counts: context ``k`` returns counts ``y_ki`` at
known exposures ``f_ki``, and its rate is never observed. Every Poisson mixture then depends
on context ``k`` only through the accumulated count and exposure, ``T_k = sum_i y_ki`` and
``F_k = sum_i f_ki``:

    log p(y_k1, ..., y_kn) = sum_i [y_ki log f_ki - log y_ki!] + M(T_k, F_k),
    M(T, F) = log int lam^T exp(-F lam) p(lam) dlam .

The bracket is the same under every mixing law, so laws are fitted and compared through
``M`` alone (type-II maximum likelihood), and the posterior predictive of a new count ``y``
at exposure ``f``, after ``(T, F)`` has been observed, is

    log p(y | T, F) = y log f - log y! + M(T + y, F + f) - M(T, F) .

For the GIG this is the Sichel law of the updated belief and for the gamma the negative
binomial; :func:`gate` checks both against ``methods/``, so the predictive scored here is the
one every other study uses.

Laws, with their number of parameters:

    poisson         a point mass: every context of a class has the class rate      (1)
    gamma           the conjugate pair, negative binomial predictive               (2)
    lognormal       Poisson-lognormal                                              (2)
    gig             generalised inverse Gaussian, Sichel predictive                (3)
    compoundgamma   beta prime, a power-law tail (methods/compoundgamma.py)        (3)
    logskewnormal   skew normal on log lam, penalised as in methods/logskewnormal  (3)
    gammamix        up to four gammas, EM and BIC, shapes capped at 1e3        (3J - 1)

The three non-conjugate laws have log-concave densities in ``s = log lam``, so the integrand
of ``M`` is unimodal in ``s``. It is integrated by the trapezoidal rule after the substitution
``s = s* + sd sinh(u)`` about its mode ``s*``, where ``sd`` is the Laplace width. The sinh
makes an exponential tail in ``s``, which the compound gamma has, decay double-exponentially
in ``u``, so one fixed grid in ``u`` serves every context; :func:`gate` checks it against a
brute-force integral.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, digamma, expit, gammaln, log_ndtr, logsumexp

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from methods.gigpoisson import log_besselk                            # noqa: E402

__all__ = ["LAWS", "N_PARAMS", "marginal", "predictive", "fit", "fit_all", "mixing_logpdf",
           "rate_moments", "gate"]

LAWS = ("poisson", "gamma", "gammamix", "lognormal", "compoundgamma", "logskewnormal", "gig")

#: Shape cap of a gamma-mixture component, as in ``methods/gammamixture.py``: EM can otherwise
#: collapse a component onto a single context.
MIX_A_MAX = 1e3
#: Fewest contexts, in expected count, a mixture component must explain to be kept.
MIX_MIN_SUPPORT = 2.0
#: Caps of the compound gamma, as in ``methods/compoundgamma.py``.
CG_R_MAX, CG_A_MAX = 1e4, 1e4
#: Bounds short of the point-mass limit of the conjugate laws. A class whose catalogue looks
#: homogeneous pulls its maximum-likelihood prior towards a point mass, along a ridge the
#: optimiser follows without end: GIG concentrations of 1e34 and orders of -6500 were seen.
#: There the log normalising constants are of order 1e34 and their differences, which are
#: the predictive, lose every digit. At the bounds the law is already indistinguishable from
#: its limit for these data: a gamma of shape 1e6 has a coefficient of variation of 0.1 per
#: cent, a GIG of concentration 1e5 of about 0.3 per cent.
GAMMA_R_MAX = 1e6
GIG_NU_MAX, GIG_OMEGA_MAX = 1000.0, 1e5
GIG_BOUNDS = [(-GIG_NU_MAX, GIG_NU_MAX), (-40.0, float(np.log(GIG_OMEGA_MAX))), (-60.0, 60.0)]
#: Skew-normal penalty of Azzalini and Arellano-Valle (2013), as in methods/logskewnormal.py.
PEN_C1, PEN_C2 = 0.875913, 0.856250

#: Trapezoidal grid in ``u`` for the quadrature laws: ``s = s* + sd sinh(u)``, ``|u| <= 9``.
_U = np.linspace(-9.0, 9.0, 181)
_DU = _U[1] - _U[0]
_LOGCOSH = np.log(np.cosh(_U))
_SINH = np.sinh(_U)


# --------------------------------------------------------------------------
# Closed forms
# --------------------------------------------------------------------------

def _logZ_gig(nu, a, b):
    return np.log(2.0) + 0.5 * nu * (np.log(a) - np.log(b)) + log_besselk(nu, np.sqrt(a * b))


def gig_params(th):
    """``(alpha, a, b)`` of methods/gigpoisson.py from ``(order, log omega, log eta)``."""
    nu, om, eta = th[0], np.exp(th[1]), np.exp(th[2])
    return nu, eta * om, om / eta


def _M_gig(th, T, F):
    nu, a, b = gig_params(th)
    return _logZ_gig(nu + T, a, b + 2.0 * F) - _logZ_gig(nu, a, b)


def _M_gamma(th, T, F):
    r, be = np.exp(th[0]), np.exp(th[1])
    return gammaln(r + T) - gammaln(r) + r * np.log(be) - (r + T) * np.log(be + F)


def _M_poisson(th, T, F):
    lam = np.exp(th[0])
    return T * np.log(lam) - F * lam


def _mix_unpack(th):
    """``(log w, r, beta)`` arrays from a flat mixture parameter vector."""
    th = np.asarray(th, float)
    J = (th.size + 1) // 3
    lw = np.concatenate([[0.0], th[:J - 1]])
    return lw - logsumexp(lw), np.exp(th[J - 1:2 * J - 1]), np.exp(th[2 * J - 1:])


def _mix_components(th, T, F):
    lw, r, be = _mix_unpack(th)
    T = np.asarray(T, float)[..., None]
    F = np.asarray(F, float)[..., None]
    return lw + gammaln(r + T) - gammaln(r) + r * np.log(be) - (r + T) * np.log(be + F)


def _M_gammamix(th, T, F):
    return logsumexp(_mix_components(th, T, F), axis=-1)


# --------------------------------------------------------------------------
# Quadrature laws: log density of s = log lam, and its first two derivatives
# --------------------------------------------------------------------------

def _lp_lognormal(th, s):
    mu, sig = th[0], np.exp(th[1])
    z = (s - mu) / sig
    return (-0.5 * z * z - np.log(sig) - 0.5 * np.log(2 * np.pi),
            -z / sig, np.full_like(s, -1.0 / sig ** 2))


def _cg_unpack(th):
    return np.exp(th[0]), 1.0 + np.exp(th[1]), np.exp(th[2])


def _lp_compoundgamma(th, s):
    """Beta prime of shapes ``(r, A)`` and scale ``B``, as a density of ``s = log lam``."""
    r, A, B = _cg_unpack(th)
    x = s - np.log(B)
    p = expit(x)
    lp = r * x - (r + A) * np.logaddexp(0.0, x) - betaln(r, A)
    return lp, r - (r + A) * p, -(r + A) * p * (1.0 - p)


def _mills(x):
    return np.exp(-0.5 * x * x - 0.5 * np.log(2.0 * np.pi) - log_ndtr(x))


def _lp_logskewnormal(th, s):
    xi, om, al = th[0], np.exp(th[1]), th[2]
    z = (s - xi) / om
    m = _mills(al * z)
    lp = np.log(2.0) - 0.5 * z * z - np.log(om) - 0.5 * np.log(2 * np.pi) + log_ndtr(al * z)
    d1 = -z / om + (al / om) * m
    d2 = -1.0 / om ** 2 - (al / om) ** 2 * m * (al * z + m)
    return lp, d1, d2


def _dth_lognormal(th, s):
    sig = np.exp(th[1])
    z = (s - th[0]) / sig
    return np.stack([z / sig, z * z - 1.0], axis=-1)


def _dth_compoundgamma(th, s):
    r, A, B = _cg_unpack(th)
    x = s - np.log(B)
    sp = np.logaddexp(0.0, x)
    d_r = x - sp - digamma(r) + digamma(r + A)
    d_A = -sp - digamma(A) + digamma(r + A)
    d_lb = -(r - (r + A) * expit(x))
    return np.stack([r * d_r, (A - 1.0) * d_A, d_lb], axis=-1)


def _dth_logskewnormal(th, s):
    xi, om, al = th[0], np.exp(th[1]), th[2]
    z = (s - xi) / om
    m = _mills(al * z)
    return np.stack([(z - al * m) / om, -1.0 + z * z - al * z * m, z * m], axis=-1)


_LOGPRIOR = {"lognormal": _lp_lognormal, "compoundgamma": _lp_compoundgamma,
             "logskewnormal": _lp_logskewnormal}
#: ``d log p(s) / d theta``: the gradient of ``M`` is its expectation under the integrand,
#: which the quadrature already has, so the fits get exact gradients at no extra cost.
_DTHETA = {"lognormal": _dth_lognormal, "compoundgamma": _dth_compoundgamma,
           "logskewnormal": _dth_logskewnormal}


def _start_s(law, th):
    if law == "lognormal":
        return th[0]
    if law == "logskewnormal":
        return th[0]
    r, A, B = _cg_unpack(th)
    return np.log(B) + np.log(r / A)


def _mode(lpf, th, T, F, s):
    """Mode of the concave ``g(s) = T s - F e^s + lp(s)``: clipped Newton, then bisection on
    ``g'`` (decreasing, since ``g`` is concave) for any element Newton left unconverged."""
    for _ in range(60):
        lp, d1, d2 = lpf(th, s)
        e = F * np.exp(np.minimum(s, 700.0))
        step = np.clip(-(T - e + d1) / (d2 - e), -2.0, 2.0)
        s = s + step
        if np.max(np.abs(step)) < 1e-10:
            return s
    g1 = T - F * np.exp(np.minimum(s, 700.0)) + lpf(th, s)[1]
    bad = np.abs(g1) > 1e-6 * (1.0 + T)
    if np.any(bad):
        lo, hi = s[bad] - 60.0, s[bad] + 60.0
        Tb, Fb = T[bad], F[bad]
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            up = Tb - Fb * np.exp(np.minimum(mid, 700.0)) + lpf(th, mid)[1] > 0.0
            lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
        s = s.copy()
        s[bad] = 0.5 * (lo + hi)
    return s


def _M_quad(law, th, T, F, warm=None, grad=False):
    """``M`` by the sinh-substituted trapezoidal rule about the mode of the integrand.

    ``warm`` is a dict carried between calls with the same ``(T, F)`` (an optimiser's
    successive evaluations), so that each call starts Newton from the previous modes. With
    ``grad``, also returns ``dM / dtheta`` per element, the integrand-weighted mean of
    ``d log p(s) / dtheta`` over the same nodes.
    """
    lpf = _LOGPRIOR[law]
    T = np.asarray(T, float)
    F = np.asarray(F, float)
    T, F = np.broadcast_arrays(T, F)
    shape = T.shape
    T, F = T.ravel(), F.ravel()
    if (warm is not None and warm.get("s") is not None and warm["s"].shape == T.shape
            and np.all(np.isfinite(warm["s"]))):
        s = warm["s"]
    else:
        s = np.full(T.shape, float(_start_s(law, th)))
        pos = F > 0
        s[pos] = np.log((T[pos] + 0.5) / F[pos])
    s = _mode(lpf, th, T, F, s)
    if warm is not None and np.all(np.isfinite(s)):
        warm["s"] = s
    curv = F * np.exp(np.minimum(s, 700.0)) - lpf(th, s)[2]
    sd = 1.0 / np.sqrt(np.maximum(curv, 1e-300))
    nodes = s[:, None] + sd[:, None] * _SINH[None, :]
    # Clipping the exponent keeps a zero exposure from meeting an overflow (0 * inf); a
    # node that far out carries no mass whenever F > 0.
    g = (T[:, None] * nodes - F[:, None] * np.exp(np.minimum(nodes, 700.0))
         + lpf(th, nodes)[0] + np.log(sd)[:, None] + _LOGCOSH[None, :])
    lse = logsumexp(g, axis=1)
    out = (lse + np.log(_DU)).reshape(shape)
    if not grad:
        return out
    w = np.exp(g - lse[:, None])
    dM = np.einsum("kn,knp->kp", w, _DTHETA[law](th, nodes))
    return out, dM.reshape(shape + (dM.shape[-1],))


def marginal(law, th, T, F):
    """``M(T, F)``, elementwise over arrays of accumulated counts and exposures."""
    if law == "gig":
        return _M_gig(th, T, F)
    if law == "gamma":
        return _M_gamma(th, T, F)
    if law == "poisson":
        return _M_poisson(th, T, F)
    if law == "gammamix":
        return _M_gammamix(th, T, F)
    return _M_quad(law, th, T, F)


def predictive(law, th, T, F, y, f):
    """``log p(y | T, F)`` of a count ``y`` at exposure ``f``, after ``(T, F)`` observed."""
    y = np.asarray(y, float)
    f = np.asarray(f, float)
    return (marginal(law, th, np.asarray(T, float) + y, np.asarray(F, float) + f)
            - marginal(law, th, T, F) + y * np.log(f) - gammaln(y + 1.0))


def mixing_logpdf(law, th, lam):
    """Log density of the fitted mixing law at rates ``lam`` (for figures)."""
    lam = np.asarray(lam, float)
    s = np.log(lam)
    if law in _LOGPRIOR:
        return _LOGPRIOR[law](th, s)[0] - s
    if law == "gamma":
        r, be = np.exp(th[0]), np.exp(th[1])
        return r * np.log(be) - gammaln(r) + (r - 1.0) * s - be * lam
    if law == "gammamix":
        lw, r, be = _mix_unpack(th)
        comp = (r * np.log(be) - gammaln(r))[None] + (r - 1.0)[None] * s[:, None] \
            - be[None] * lam[:, None]
        return logsumexp(comp + lw[None], axis=1)
    if law == "gig":
        nu, a, b = gig_params(th)
        return (nu - 1.0) * s - 0.5 * (a / lam + b * lam) - _logZ_gig(nu, a, b)
    raise ValueError(law)


def rate_moments(law, th):
    """Mean and variance of the fitted mixing law (``inf`` where they do not exist)."""
    if law == "poisson":
        return float(np.exp(th[0])), 0.0
    if law == "gamma":
        r, be = np.exp(th[0]), np.exp(th[1])
        return float(r / be), float(r / be ** 2)
    if law == "gammamix":
        lw, r, be = _mix_unpack(th)
        w = np.exp(lw)
        m = float(np.sum(w * r / be))
        return m, float(np.sum(w * (r / be ** 2 + (r / be) ** 2)) - m * m)
    if law == "gig":
        nu, a, b = gig_params(th)
        om, eta = np.sqrt(a * b), np.sqrt(a / b)
        lk = log_besselk(np.array([nu, nu + 1.0, nu + 2.0]), om)
        m1, m2 = eta * np.exp(lk[1] - lk[0]), eta ** 2 * np.exp(lk[2] - lk[0])
        return float(m1), float(m2 - m1 * m1)
    # Quadrature laws: integrate on a wide grid in s.
    lpf = _LOGPRIOR[law]
    s0 = float(_start_s(law, th))
    s = np.linspace(s0 - 60.0, s0 + 60.0, 240001)
    lp = lpf(th, s)[0]
    w = np.exp(lp - lp.max())
    w /= w.sum()
    m = float(np.sum(w * np.exp(s)))
    return m, float(np.sum(w * np.exp(2 * s)) - m * m)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

N_PARAMS = {"poisson": 1, "gamma": 2, "lognormal": 2, "gig": 3, "compoundgamma": 3,
            "logskewnormal": 3}


def _minimise(f, starts, jac=None, polish=800, bounds=None):
    """Best of L-BFGS runs from each start, each polished by a short simplex search (which
    also copes with the barrier values :func:`_safe` returns outside the parameter space).
    With ``bounds``, starts are clipped into them and the simplex sees a barrier outside."""
    lo = hi = None
    if bounds is not None:
        lo = np.array([b[0] for b in bounds], float)
        hi = np.array([b[1] for b in bounds], float)
    value = f if jac is None else (lambda th: f(th)[0])
    if bounds is not None:
        inner = value

        def value(th):
            return inner(th) if np.all(th >= lo) and np.all(th <= hi) else 1e300
    best = None
    for x0 in starts:
        x0 = np.asarray(x0, float)
        if bounds is not None:
            x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
        r = minimize(f, x0, jac=jac, method="L-BFGS-B", bounds=bounds,
                     options=dict(maxiter=400))
        if polish:
            r2 = minimize(value, r.x, method="Nelder-Mead",
                          options=dict(maxfev=polish, xatol=1e-8, fatol=1e-10, adaptive=True))
            if r2.fun < r.fun - 1e-12:
                r = r2
        if best is None or r.fun < best.fun:
            best = r
    return best


def _safe(fun):
    def g(th):
        if not np.all(np.isfinite(th)):
            return 1e300
        with np.errstate(all="ignore"):
            v = fun(th)
        return float(v) if np.isfinite(v) else 1e300
    return g


def _moments(T, F, w=None):
    """Pooled rate and the variance of the rates net of the Poisson noise in ``T / F``."""
    w = np.ones_like(T) if w is None else w
    sw = w.sum()
    m = np.sum(w * T) / np.sum(w * F)
    rate = T / F
    v = np.sum(w * (rate - m) ** 2) / sw - m * np.sum(w / F) / sw
    return m, max(v, 1e-4 * m * m)


def _fit_gamma(T, F, w=None, starts=None, cap=None):
    """Weighted gamma fit on counts, by L-BFGS with the analytic gradient in
    ``(log r, log beta)``. ``cap`` bounds the shape (mixture components)."""
    w = np.ones_like(T) if w is None else w

    def f(th):
        r, be = np.exp(th[0]), np.exp(th[1])
        lbf = np.log(be + F)
        M = gammaln(r + T) - gammaln(r) + r * th[1] - (r + T) * lbf
        dr = digamma(r + T) - digamma(r) + th[1] - lbf
        db = r / be - (r + T) / (be + F)
        return -np.sum(w * M), -np.array([np.sum(w * dr) * r, np.sum(w * db) * be])

    if starts is None:
        m, v = _moments(T, F, w)
        starts = [[np.log(m * m / v), np.log(m / v)], [0.0, -np.log(m)]]
    bounds = [(-15.0, float(np.log(cap if cap else GAMMA_R_MAX))), (-60.0, 60.0)]
    lo = np.array([bd[0] for bd in bounds])
    hi = np.array([bd[1] for bd in bounds])
    best = None
    for x0 in starts:
        x0 = np.clip(np.asarray(x0, float), lo, hi)
        r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    return best.x, -float(best.fun)


def _fit_gammamix(T, F, J, restarts=3, iters=150, seed=0):
    """EM for a mixture of ``J`` gammas on counts; components follow methods/gammamixture.py:
    shapes capped at :data:`MIX_A_MAX`, a component explaining fewer than
    :data:`MIX_MIN_SUPPORT` contexts dropped. Restarts split the log-rates into ``J``
    quantile bands, jittered after the first, as the rate-based fit does."""
    rng = np.random.default_rng(seed)
    x = np.log((T + 0.5) / F)
    best = None
    for rep in range(restarts):
        q = np.quantile(x, np.linspace(0, 1, J + 1))
        if rep:
            q[1:-1] += rng.normal(0, 0.15 * (q[-1] - q[0]) / J, J - 1)
            q.sort()
        lab = np.clip(np.searchsorted(q[1:-1], x), 0, J - 1)
        resp = np.eye(J)[lab] + 1e-3
        resp /= resp.sum(axis=1, keepdims=True)
        par = [None] * J
        prev, ll, th = -np.inf, -np.inf, None
        for it in range(iters):
            keep = resp.sum(axis=0) >= MIX_MIN_SUPPORT
            if not np.any(keep):
                break
            if not np.all(keep):
                resp = resp[:, keep]
                resp /= resp.sum(axis=1, keepdims=True)
                par = [p_ for p_, k_ in zip(par, keep) if k_]
            for j in range(resp.shape[1]):
                par[j], _ = _fit_gamma(T, F, resp[:, j],
                                       starts=None if par[j] is None else [par[j]],
                                       cap=MIX_A_MAX)
            lw = np.log(resp.mean(axis=0))
            lr = np.array([p_[0] for p_ in par])
            lb = np.array([p_[1] for p_ in par])
            th = np.concatenate([lw[1:] - lw[0], lr, lb])
            comp = _mix_components(th, T, F)
            norm = logsumexp(comp, axis=1)
            ll = float(norm.sum())
            resp = np.exp(comp - norm[:, None])
            if ll - prev < 1e-9 * abs(ll):
                break
            prev = ll
        if th is not None and (best is None or ll > best[1]):
            best = (th.copy(), ll)
    return best


def _ig_moments(m, v):
    """Shape and scale of the inverse gamma with mean ``m`` and variance ``v``."""
    al = 2.0 + m * m / v
    return al, m * (al - 1.0)


def fit(law, T, F, init=None):
    """Maximum (penalised, for ``logskewnormal``) marginal likelihood. Returns ``(theta, M)``
    with ``M = sum_k M(T_k, F_k)`` at the optimum, penalty excluded.

    ``init``, a previous fit of the same law (to a sub-catalogue), is tried as an extra start
    to every default start of the three-parameter laws, never instead of them.
    """
    T = np.asarray(T, float)
    F = np.asarray(F, float)
    m, v = _moments(T, F)
    if law == "poisson":
        th = np.array([np.log(m)])
        return th, float(np.sum(_M_poisson(th, T, F)))
    if law == "gamma":
        return _fit_gamma(T, F)
    if law == "gammamix":
        best = None
        for J in range(1, 5):
            if J == 1:
                th, ll = _fit_gamma(T, F)
            else:
                if T.size < 2 * MIX_MIN_SUPPORT * J:
                    break
                res = _fit_gammamix(T, F, J)
                if res is None or res[0].size != 3 * J - 1:   # a component was dropped
                    continue
                th, ll = res
            bic = -2.0 * ll + (3 * J - 1) * np.log(T.size)
            if best is None or bic < best[0]:
                best = (bic, th, ll)
        return best[1], best[2]

    if law == "gig":
        def obj(th):
            return -np.sum(_M_gig(th, T, F))
        # The gamma is the omega -> 0 edge at order r and b = 2 beta, and the inverse gamma
        # the b -> 0 edge at order -shape and a = 2 scale: one start next to each, so the
        # optimum can end below neither, and two in the interior.
        g, _ = _fit_gamma(T, F)
        r, be = np.exp(g[0]), np.exp(g[1])
        al, sc = _ig_moments(m, v)
        starts = [[r, np.log(1e-3), np.log(1e-3) - np.log(2.0 * be)],
                  [-al, np.log(1e-3), np.log(2.0 * sc) - np.log(1e-3)],
                  [-1.0, np.log(0.3), np.log(m)], [0.5, np.log(1.0), np.log(m)]]
        if init is not None:
            starts = [list(init)] + starts
        res = _minimise(_safe(obj), starts, bounds=GIG_BOUNDS)
        return res.x, float(np.sum(_M_gig(res.x, T, F)))

    warm = {}

    def value_grad(th, pen=False):
        if not np.all(np.isfinite(th)):
            return 1e300, np.zeros_like(th)
        with np.errstate(all="ignore"):
            M, dM = _M_quad(law, th, T, F, warm, grad=True)
            v, g = -float(np.sum(M)), -np.sum(dM, axis=0)
            if pen:
                v += PEN_C1 * np.log1p(PEN_C2 * th[2] * th[2])
                g[2] += PEN_C1 * 2.0 * PEN_C2 * th[2] / (1.0 + PEN_C2 * th[2] * th[2])
        if not (np.isfinite(v) and np.all(np.isfinite(g))):
            return 1e300, np.zeros_like(th)
        return v, g

    pen = False
    if law == "lognormal":
        s2 = np.log1p(v / m / m)
        starts = [[np.log(m) - 0.5 * s2, 0.5 * np.log(s2)]]
        bounds = [(-60.0, 60.0), (-12.0, 5.0)]
    elif law == "compoundgamma":
        g, _ = _fit_gamma(T, F)
        r0, b0 = float(np.exp(g[0])), float(np.exp(g[1]))
        al, sc = _ig_moments(m, v)
        # At the gamma limit (the fitted gamma, tail index at half its cap: the likelihood is
        # flat in the tail index there, so a start further in can stall short of the limit),
        # next to the inverse-gamma limit (large shape), and two interior points.
        a_lim = 0.5 * CG_A_MAX
        starts = [[np.log(r0), np.log(a_lim - 1.0), np.log((a_lim - 1.0) / b0)],
                  [np.log(1000.0), np.log(max(al - 1.0, 0.05)), np.log(sc / 1000.0)],
                  [0.0, np.log(0.5), np.log(m * 0.5)],
                  [np.log(10.0), np.log(2.0), np.log(m * 2.0 / 10.0)]]
        bounds = [(-10.0, np.log(CG_R_MAX)), (-10.0, np.log(CG_A_MAX - 1.0)), (-80.0, 80.0)]
    elif law == "logskewnormal":
        ln, _ = fit("lognormal", T, F)
        pen = True
        starts = [[ln[0], ln[1], a_] for a_ in (-2.0, 0.0, 2.0)]
        bounds = [(-60.0, 60.0), (-12.0, 5.0), (-60.0, 60.0)]
    else:
        raise ValueError(law)
    if init is not None:
        starts = [list(init)] + starts
    lo = np.array([bd[0] for bd in bounds])
    hi = np.array([bd[1] for bd in bounds])
    best = None
    for x0 in starts:
        x0 = np.clip(np.asarray(x0, float), lo + 1e-9, hi - 1e-9)
        warm.clear()
        r = minimize(lambda th: value_grad(th, pen), x0, jac=True, method="L-BFGS-B",
                     bounds=bounds, options=dict(maxiter=500))
        if best is None or r.fun < best.fun:
            best = r
    return best.x, float(np.sum(_M_quad(law, best.x, T, F)))


def fit_all(T, F, laws=LAWS):
    return {law: fit(law, T, F) for law in laws}


def n_params(law, th):
    return 3 * ((len(th) + 1) // 3) - 1 if law == "gammamix" else N_PARAMS[law]


# --------------------------------------------------------------------------
# Correctness gate
# --------------------------------------------------------------------------

def gate(verbose=True):
    """Check the predictive of every law before any data are scored. Returns failures.

    1. GIG: ``M``-based predictive against ``sichel_logpmf`` of the conjugate posterior.
    2. Gamma: against ``GammaPoisson.logpmf`` of its posterior.
    3. Quadrature laws: ``M`` against a brute-force trapezoid on 400 001 points in ``s``.
    4. Quadrature laws: the exact gradient of ``M`` against central differences.
    5. Every law, fitted to a catalogue with no heterogeneity: finite, non-negative surprisals.
    6. Every law: the predictive sums to one over its support.
    """
    from methods.gigpoisson import posterior, sichel_logpmf
    from methods.countmodels import GammaPoisson
    fails = []
    rng = np.random.default_rng(0)
    cases = [(0.0, 0.0), (3.0, 1.0), (40.0, 7.0), (2500.0, 12.0), (0.0, 25.0)]
    y = np.array([0.0, 1.0, 4.0, 17.0, 120.0])
    for th in ([-2.3, np.log(0.4), np.log(3.0)], [1.7, np.log(2.0), np.log(0.8)]):
        nu, a, b = gig_params(th)
        for T, F in cases:
            ours = predictive("gig", th, T, F, y, 1.5)
            ref = sichel_logpmf(y, posterior({"alpha": nu, "a": a, "b": b}, T, F), 1.5)
            err = float(np.max(np.abs(ours - ref)))
            if err > 1e-8:
                fails.append(("gig vs sichel", T, F, err))
    gp = GammaPoisson()
    for th in ([np.log(0.6), np.log(0.2)], [np.log(30.0), np.log(9.0)]):
        for T, F in cases:
            ours = predictive("gamma", th, T, F, y, 1.5)
            post = gp.update({"a": float(np.exp(th[0])), "b": float(np.exp(th[1]))}, T, F)
            err = float(np.max(np.abs(ours - gp.logpmf(post, 1.5, y))))
            if err > 1e-8:
                fails.append(("gamma vs negbin", T, F, err))
    quad = {"lognormal": [[1.2, np.log(0.9)], [-2.0, np.log(2.5)]],
            "compoundgamma": [[np.log(0.4), np.log(0.8), np.log(2.0)],
                              [np.log(30.0), np.log(1.5), np.log(0.1)]],
            "logskewnormal": [[1.0, np.log(1.4), 4.0], [0.5, np.log(0.6), -7.0]]}
    s = np.linspace(-80.0, 40.0, 400001)
    ds = s[1] - s[0]
    for law, ths in quad.items():
        for th in ths:
            lp = _LOGPRIOR[law](th, s)[0]
            for T, F in cases:
                ref = logsumexp(T * s - F * np.exp(s) + lp) + np.log(ds)
                err = abs(float(marginal(law, th, T, F)) - float(ref))
                if err > 1e-6:
                    fails.append(("%s quadrature" % law, T, F, err))
    # The fits use exact gradients of M: check them against central differences.
    Tg, Fg = np.array([0.0, 3.0, 40.0, 2500.0]), np.array([0.0, 1.0, 7.0, 12.0])
    for law, ths in quad.items():
        for th in ths:
            th = np.asarray(th, float)
            _, dM = _M_quad(law, th, Tg, Fg, grad=True)
            for i in range(th.size):
                e = np.zeros_like(th)
                e[i] = 1e-5
                fd = (_M_quad(law, th + e, Tg, Fg) - _M_quad(law, th - e, Tg, Fg)) / 2e-5
                err = float(np.max(np.abs(fd - dM[:, i])))
                if err > 1e-5 * (1.0 + float(np.max(np.abs(fd)))):
                    fails.append(("%s gradient" % law, i, err))
    # A catalogue with no heterogeneity at all pulls every law towards its point mass. The fit
    # must stop short of it in a numerically sound place: finite parameters, and a predictive
    # whose surprisals are finite and non-negative, as they must be for a count.
    rng_h = np.random.default_rng(7)
    Fh = np.full(20, 12.0)
    Th = rng_h.poisson(3.0 * Fh).astype(float)
    yh, fh = rng_h.poisson(3.0 * 10.0, 50).astype(float), 10.0
    for law in LAWS:
        th, _ = fit(law, Th, Fh)
        for T0, F0 in ((0.0, 0.0), (40.0, 12.0)):
            nl = -predictive(law, th, T0, F0, yh, fh)
            if not (np.all(np.isfinite(th)) and np.all(np.isfinite(nl))
                    and float(np.min(nl)) > -1e-9 and float(np.max(nl)) < 50.0):
                fails.append(("%s on a homogeneous catalogue" % law, T0, F0,
                              float(np.min(nl)), float(np.max(nl))))
    mix = np.array([0.4, np.log(0.7), np.log(20.0), np.log(1.0), np.log(4.0)])
    norm_cases = {"poisson": [np.log(3.0)], "gamma": [np.log(0.6), np.log(0.2)],
                  "gammamix": mix, "lognormal": quad["lognormal"][0],
                  "compoundgamma": quad["compoundgamma"][0],
                  "logskewnormal": quad["logskewnormal"][0],
                  "gig": [-2.3, np.log(0.4), np.log(3.0)]}
    yy = np.arange(0.0, 20000.0)
    for law, th in norm_cases.items():
        for T, F in ((0.0, 0.0), (12.0, 3.0)):
            tot = float(np.exp(logsumexp(predictive(law, np.asarray(th, float), T, F, yy,
                                                    1.0))))
            # A power-law prior leaves a heavy predictive tail when nothing is observed: the
            # compound gamma at (0, 0) has tail index 1.2 here, so the support is not enough
            # to collect all the mass and only a lower bound is checked.
            heavy = law in ("compoundgamma",) and T == 0.0
            if (abs(tot - 1.0) > 1e-6 and not heavy) or tot > 1.0 + 1e-6 or tot < 0.95:
                fails.append(("%s normalisation" % law, T, F, tot))
    if verbose:
        print("countfit gate: {} failures".format(len(fails)))
        for f in fails:
            print("   ", f)
    return fails


if __name__ == "__main__":
    import time
    t = time.time()
    gate()
    print("  {:.1f} s".format(time.time() - t))
