"""A finite mixture of gammas as the mixing law: flexible and still conjugate.

The obvious rival to a three-parameter conjugate family. A mixture of ``J`` gammas can
approximate any mixing law on the positive half-line as ``J`` grows, and it stays conjugate
to the Poisson: an observation ``y`` at exposure ``f`` updates every component exactly,

    Gamma(a_j, b_j)  ->  Gamma(a_j + y, b_j + f),

and reweights the components by their own predictive mass,

    w_j  ->  w_j NB(y | a_j, f / b_j) / sum_i w_i NB(y | a_i, f / b_i).

The predictive is a mixture of negative binomials, and the expected information gain is
exact: the predictive entropy is a sum over its support, and the aleatoric term is a
weighted sum of each component's, by the same quadrature the gamma model uses.

**Fitting.** To a population of rates, by EM with the shape of each component solved by
Newton's method on the weighted gamma likelihood equation, restarted from quantile splits,
with ``J in {1, ..., 4}`` chosen by BIC. Where only two moments are given (bulk sampling),
there is nothing to fit a mixture to, and the model is a single gamma matched to them,
which is the gamma-Poisson model exactly.
"""

import numpy as np
from scipy.special import digamma, gammaln, logsumexp, polygamma
from scipy.stats import nbinom

from .countmodels import GammaPoisson

__all__ = ["GammaMixturePoisson", "fit_gamma_mixture"]

_GAMMA = GammaPoisson()


def _weighted_gamma_mle(x, w, a0):
    """Shape and rate of a gamma maximising ``sum w log Gamma(x | a, b)``."""
    sw = w.sum()
    m = float((w * x).sum() / sw)
    c = np.log(m) - float((w * np.log(x)).sum() / sw)       # >= 0 by Jensen
    a = float(a0) if np.isfinite(a0) and a0 > 0 else 0.5 / max(c, 1e-8)
    for _ in range(100):
        g = np.log(a) - digamma(a) - c
        h = 1.0 / a - polygamma(1, a)
        step = g / h
        a_new = a - step
        if a_new <= 0:
            a_new = a / 2.0
        if abs(a_new - a) < 1e-10 * a:
            a = a_new
            break
        a = a_new
    return a, a / m


def _mixture_loglik(x, logw, a, b):
    lp = (a * np.log(b) - gammaln(a))[None, :] + (a - 1.0)[None, :] * np.log(x)[:, None] \
        - b[None, :] * x[:, None]
    return logsumexp(lp + logw[None, :], axis=1)


#: Largest shape a component may take: a coefficient of variation of about 3 per cent. EM can
#: otherwise collapse a component onto a single rate (shapes of 1e14 were seen on a class of
#: twenty), which is overfitting rather than structure, and which breaks double precision in
#: ``a log b - lgamma(a)``.
A_MAX = 1e3
#: Fewest rates, in expected count, a component must explain to be kept.
MIN_SUPPORT = 2.0


def fit_gamma_mixture(x, J, restarts=4, iters=300, seed=0):
    """EM for a gamma mixture of at most ``J`` components. Returns ``(logw, a, b, loglik)``.

    Regularised in the two standard ways, both inactive unless violated: each component's
    shape is capped at :data:`A_MAX` (its mean is kept), and a component whose expected count
    falls below :data:`MIN_SUPPORT` is dropped and EM continues with the rest. The returned
    arrays have as many entries as components survived.
    """
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        # Start from a split of the log-rates into J quantile bands, jittered after the first.
        q = np.quantile(np.log(x), np.linspace(0, 1, J + 1))
        if r:
            q[1:-1] += rng.normal(0, 0.15 * (q[-1] - q[0]) / J, J - 1)
            q.sort()
        lab = np.clip(np.searchsorted(q[1:-1], np.log(x)), 0, J - 1)
        resp = np.eye(J)[lab] + 1e-3
        resp /= resp.sum(axis=1, keepdims=True)
        a = np.full(J, np.nan)
        b = np.zeros(J)
        prev = -np.inf
        for _ in range(iters):
            keep = resp.sum(axis=0) >= MIN_SUPPORT
            if not np.all(keep) and np.any(keep):
                resp, a, b = resp[:, keep], a[keep], b[keep]
                resp /= resp.sum(axis=1, keepdims=True)
            for j in range(a.size):
                a[j], b[j] = _weighted_gamma_mle(x, resp[:, j], a[j])
                if a[j] > A_MAX:
                    b[j] *= A_MAX / a[j]          # cap the shape, keep the mean a / b
                    a[j] = A_MAX
            logw = np.log(resp.mean(axis=0))
            lp = (a * np.log(b) - gammaln(a))[None, :] + (a - 1.0)[None, :] * \
                np.log(x)[:, None] - b[None, :] * x[:, None] + logw[None, :]
            norm = logsumexp(lp, axis=1)
            ll = float(norm.sum())
            resp = np.exp(lp - norm[:, None])
            if ll - prev < 1e-9 * abs(ll):
                break
            prev = ll
        if best is None or ll > best[3]:
            best = (logw.copy(), a.copy(), b.copy(), ll)
    return best


class GammaMixturePoisson(object):
    """Poisson likelihood, a finite gamma mixture on the rate. Conjugate; NB-mixture predictive."""

    name = "gammamix-poisson"
    label = "gamma-mixture-Poisson"
    conjugate = True

    def __init__(self, max_components=4):
        self.max_components = int(max_components)

    # -- belief -------------------------------------------------------------

    def prior_from_moments(self, mean, var):
        g = _GAMMA.prior_from_moments(mean, var)
        return {"logw": np.zeros(1), "a": np.array([g["a"]]), "b": np.array([g["b"]])}

    def fit_population(self, rates):
        x = np.asarray(rates, dtype=float)
        best = None
        for J in range(1, self.max_components + 1):
            logw, a, b, ll = fit_gamma_mixture(x, J)
            bic = -2.0 * ll + (3 * J - 1) * np.log(x.size)
            if best is None or bic < best[0]:
                best = (bic, logw, a, b)
        return {"logw": best[1], "a": best[2], "b": best[3]}

    def update(self, belief, y, f):
        y = float(y)
        lp = self._component_logpmf(belief, f, np.array([y]))[0]
        logw = belief["logw"] + lp
        return {"logw": logw - logsumexp(logw), "a": belief["a"] + y, "b": belief["b"] + f}

    def _w(self, belief):
        return np.exp(belief["logw"])

    def rate_mean(self, belief):
        return float(np.sum(self._w(belief) * belief["a"] / belief["b"]))

    def rate_moments(self, belief):
        w, a, b = self._w(belief), belief["a"], belief["b"]
        m = float(np.sum(w * a / b))
        return m, float(np.sum(w * (a / b ** 2 + (a / b) ** 2)) - m * m)

    def sample_rate(self, belief, n, rng):
        j = rng.choice(belief["a"].size, size=n, p=self._w(belief))
        return rng.gamma(belief["a"][j], 1.0 / belief["b"][j])

    def _density_grid(self, belief, n=4001):
        """The mixture density of ``log lam`` on a grid spanning every component."""
        a, b = belief["a"], belief["b"]
        lo = np.min(np.log(np.maximum(a, 1e-3) / b) - 12.0 / np.sqrt(np.maximum(a, 1e-3))
                    - np.where(a < 1, 30.0 / np.maximum(a, 1e-3), 0.0))
        hi = np.max(np.log(np.maximum(a, 1e-3) / b) + 12.0 / np.sqrt(np.maximum(a, 1e-3)) + 2.0)
        s = np.linspace(lo, hi, n)
        lp = (a * np.log(b) - gammaln(a))[None, :] + a[None, :] * s[:, None] \
            - b[None, :] * np.exp(s)[:, None]
        return s, logsumexp(lp + belief["logw"][None, :], axis=1)

    def rate_map(self, belief):
        s, ld = self._density_grid(belief)
        return float(np.exp(s[np.argmax(ld - s)]))       # mode of the density in lam

    def rate_quantile(self, belief, q):
        s, ld = self._density_grid(belief)
        w = np.exp(ld - ld.max())
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]))])
        return np.exp(np.interp(q, cdf / cdf[-1], s))

    # -- predictive ---------------------------------------------------------

    def _component_logpmf(self, belief, f, y):
        a, sc = belief["a"][None, :], (f / belief["b"])[None, :]
        y = np.asarray(y, dtype=float)[:, None]
        l1 = np.log1p(sc)
        return (gammaln(a + y) - gammaln(a) - gammaln(y + 1.0) - a * l1
                + y * (np.log(sc) - l1))

    def logpmf(self, belief, f, y):
        y = np.atleast_1d(np.asarray(y, dtype=float))
        return logsumexp(self._component_logpmf(belief, f, y) + belief["logw"][None, :], axis=1)

    def support(self, belief, f, tail_tol=1e-9):
        ymax = 0
        for a, b in zip(belief["a"], belief["b"]):
            s = f / b
            ymax = max(ymax, int(nbinom.isf(tail_tol, a, 1.0 / (1.0 + s))) + 16)
        return np.arange(min(ymax, 4_000_000) + 1, dtype=float)

    def predictive_entropy(self, belief, f):
        if f <= 0.0:
            return 0.0
        lp = self.logpmf(belief, f, self.support(belief, f))
        return -float(np.sum(np.exp(lp) * lp))

    def aleatoric_entropy(self, belief, f):
        if f <= 0.0:
            return 0.0
        w = self._w(belief)
        return float(sum(wj * _GAMMA.aleatoric_entropy({"a": aj, "b": bj}, f)
                         for wj, aj, bj in zip(w, belief["a"], belief["b"]) if wj > 1e-12))

    def eig(self, belief, f):
        """Memoised on the belief and the exposure, as for the GIG: contexts of a class share
        their prior, so the first round needs one evaluation per class and action."""
        key = (belief["logw"].tobytes(), belief["a"].tobytes(), belief["b"].tobytes(),
               float(f))
        memo = self.__dict__.setdefault("_memo", {})
        if key not in memo:
            if len(memo) > 200000:
                memo.clear()
            memo[key] = self.predictive_entropy(belief, f) - self.aleatoric_entropy(belief, f)
        return memo[key]

    def fisher_dopt(self, belief, f):
        m, v = self.rate_moments(belief)
        return float(np.log1p(f * v / max(m, 1e-12)))

