"""Count models a sensing agent might hold, behind one interface.

Every model here is a Poisson likelihood mixed over an unknown rate. They differ only in
the mixing law, which is the whole of the modelling question this paper asks:

    GIGPoisson        generalised inverse Gaussian mixing, three parameters. Conjugate, so
                      the posterior stays in the family and the predictive is Sichel.
    GammaPoisson      gamma mixing, two parameters. Conjugate, predictive negative binomial.
                      This is the ``omega -> 0`` boundary of the family above, so it is the
                      nearest rival rather than a different tradition.
    LognormalPoisson  lognormal mixing, two parameters. Not conjugate: the posterior is
                      carried on a grid and every quantity is a quadrature. Included because
                      it is the mixing law the species-abundance literature most often
                      prefers, and because it prices what leaving the conjugate class costs.

**Fair priors.** A comparison between models is only about the models if each starts from
the same prior information. Every model is therefore initialised by matching the first two
moments of the mixing law, through :meth:`CountModel.prior_from_moments`. No model is given
the true parameters, and no model is given a prior another could not express.

**Fair scoring.** Every model supplies a probability mass function on the non-negative
integers, so held-out predictive scores are comparable across models rather than across
supports.
"""

import numpy as np
from scipy.special import digamma, gammaln, logsumexp
from scipy.stats import nbinom

from .gigpoisson import (gig_logpdf, gig_moments, gig_sample, posterior as gig_posterior,
                         sichel_logpmf, sichel_support)

__all__ = ["CountModel", "GIGPoisson", "GIGPoissonNMC", "GammaPoisson", "LognormalPoisson",
           "LogNormalLikelihood", "poisson_entropy", "MODELS", "EXTRA_MODELS",
           "build_model"]

POISSON_SERIES_FROM = 400.0
QUAD_NODES = 64
LOG_DENSITY_DROP = 75.0
#: Total log-likelihood by which a GIG population fit may trail the gamma fit before it is
#: treated as a failed optimisation and restarted. Rounding in SciPy's single-start fit
#: leaves a few thousandths of a nat either way; a failed fit trails by tens of nats.
FIT_SLACK = 0.1
_GAUSS = {}


def poisson_entropy(mu):
    """``H[Poisson(mu)]`` in nats, vectorised: direct sum below the crossover, series above."""
    mu = np.asarray(mu, dtype=float)
    scalar = mu.ndim == 0
    mu = np.atleast_1d(mu)
    out = np.zeros_like(mu)
    small = (mu < POISSON_SERIES_FROM) & (mu > 0.0)
    if np.any(small):
        ms = mu[small]
        ymax = int(ms.max() + 12.0 * np.sqrt(ms.max()) + 40.0)
        y = np.arange(ymax + 1, dtype=float)
        lp = -ms[:, None] + y[None, :] * np.log(ms)[:, None] - gammaln(y + 1.0)[None, :]
        out[small] = -np.sum(np.exp(lp) * lp, axis=1)
    big = mu >= POISSON_SERIES_FROM
    if np.any(big):
        m = mu[big]
        out[big] = (0.5 * np.log(2.0 * np.pi * np.e * m) - 1.0 / (12.0 * m)
                    - 1.0 / (24.0 * m ** 2) - 19.0 / (360.0 * m ** 3))
    return float(out[0]) if scalar else out


def _gauss_nodes(n):
    if n not in _GAUSS:
        _GAUSS[n] = np.polynomial.legendre.leggauss(n)
    return _GAUSS[n]


class CountModel(object):
    """A belief about one rate, and everything an acquisition might ask of it."""

    name = "model"
    label = "model"
    conjugate = True

    # -- belief -------------------------------------------------------------

    def prior_from_moments(self, mean, var):
        """Initialise a belief with the given mixing mean and variance."""
        raise NotImplementedError

    def fit_population(self, rates):
        """Initialise a belief by maximum likelihood on a sample of rates.

        Where a population of comparable rates has been observed, this is a better prior
        than two matched moments and a fairer comparison between families: each model is
        allowed the best description of the population its own parameters can express,
        rather than being forced through a summary chosen to suit somebody else's family.
        The default falls back to the moments, which is what a family with no closed-form
        fit would do.
        """
        rates = np.asarray(rates, dtype=float)
        return self.prior_from_moments(float(rates.mean()), float(rates.var()))

    def update(self, belief, y, f):
        raise NotImplementedError

    def rate_mean(self, belief):
        raise NotImplementedError

    def rate_moments(self, belief):
        """``(mean, variance)`` of the rate under the belief."""
        raise NotImplementedError

    def rate_map(self, belief):
        """Mode of the posterior density over the rate, in the rate's own coordinate."""
        raise NotImplementedError

    def sample_rate(self, belief, n, rng):
        raise NotImplementedError

    # -- predictive ---------------------------------------------------------

    def support(self, belief, f, tail_tol=1e-9):
        raise NotImplementedError

    def logpmf(self, belief, f, y):
        raise NotImplementedError

    # -- acquisition ingredients -------------------------------------------

    def predictive_entropy(self, belief, f):
        if f <= 0.0:
            return 0.0
        y = self.support(belief, f)
        lp = self.logpmf(belief, f, y)
        return -float(np.sum(np.exp(lp) * lp))

    def aleatoric_entropy(self, belief, f):
        raise NotImplementedError

    def eig(self, belief, f):
        return self.predictive_entropy(belief, f) - self.aleatoric_entropy(belief, f)

    def variance_decomposition(self, belief, f):
        m, v = self.rate_moments(belief)
        return float(f * m), float(f * f * v)

    def fisher_dopt(self, belief, f):
        """Local D-optimality in the variance-stabilising coordinate of a Poisson rate."""
        m, v = self.rate_moments(belief)
        return float(np.log1p(f * v / max(m, 1e-12)))


# --------------------------------------------------------------------------

class GIGPoisson(CountModel):
    """Generalised inverse Gaussian mixing. Conjugate; the predictive is the Sichel law."""

    name = "gig-poisson"
    label = "GIG-Poisson"
    conjugate = True

    def __init__(self, order=-0.5):
        self.order = float(order)
        self._memo = {}

    def prior_from_moments(self, mean, var):
        """Solve for the concentration that gives the requested variance at that mean.

        The order is held at its default and the scale follows from the mean, so one
        parameter is left to carry the variance. The equation is monotone in ``omega`` over
        the range used here, so a bisection is enough and no starting value is needed.
        """
        from .gigpoisson import gig_from_mean
        lo, hi = 1e-3, 1e3

        def vom(omega):
            m, v = gig_moments(gig_from_mean(self.order, mean, omega))
            return v / m

        target = var / mean
        for _ in range(80):
            mid = np.sqrt(lo * hi)
            if vom(mid) > target:
                lo = mid
            else:
                hi = mid
        return gig_from_mean(self.order, mean, np.sqrt(lo * hi))

    def fit_population(self, rates):
        """Maximum likelihood, guarded against the optimiser stopping short.

        SciPy's single-start fit can stop at a local optimum whose likelihood is below the
        gamma's, which cannot be the maximum: the gamma is the ``omega -> 0`` limit of this
        family, so the GIG maximum is at least the gamma maximum. When that happens the fit is
        restarted from several orders and concentrations and the best is kept. A fit within
        :data:`FIT_SLACK` nats of the gamma, or above it, is returned unchanged.
        """
        from scipy.optimize import minimize
        from scipy.stats import gamma as gamma_dist, geninvgauss
        x = np.asarray(rates, dtype=float)

        def loglik(p, omega, eta):
            return float(np.sum(geninvgauss.logpdf(x, p, omega, scale=eta)))

        p, omega, _, eta = geninvgauss.fit(x, floc=0.0)
        shape, _, scale = gamma_dist.fit(x, floc=0.0)
        floor = float(np.sum(gamma_dist.logpdf(x, shape, scale=scale)))
        if not loglik(p, omega, eta) >= floor - FIT_SLACK:
            def nll(th):
                v = -loglik(th[0], np.exp(th[1]), np.exp(th[2]))
                return v if np.isfinite(v) else 1e300
            best = None
            for p0 in (-3.0, -1.0, 0.5, 2.0, 5.0):
                for omega0 in (0.01, 0.3, 3.0):
                    r = minimize(nll, [p0, np.log(omega0), np.log(x.mean())],
                                 method="Nelder-Mead",
                                 options=dict(maxiter=4000, xatol=1e-8, fatol=1e-10))
                    if best is None or r.fun < best.fun:
                        best = r
            p, omega, eta = best.x[0], np.exp(best.x[1]), np.exp(best.x[2])
        # SciPy's (p, b, scale) is our (order, omega, eta); invert eta = sqrt(a/b).
        return {"alpha": float(p), "a": float(eta * omega), "b": float(omega / eta)}

    def update(self, belief, y, f):
        return gig_posterior(belief, y, f)

    def rate_mean(self, belief):
        return gig_moments(belief)[0]

    def rate_moments(self, belief):
        return gig_moments(belief)

    def rate_map(self, belief):
        # Stationary point of (alpha - 1) log lam - (a / lam + b lam) / 2.
        alpha, a, b = belief["alpha"], belief["a"], belief["b"]
        return float(((alpha - 1.0) + np.sqrt((alpha - 1.0) ** 2 + a * b)) / b)

    def sample_rate(self, belief, n, rng):
        return gig_sample(belief, n, rng)

    def rate_quantile(self, belief, q):
        """Posterior quantiles of the rate, from the log-concave density of ``log lam``.

        SciPy's ``geninvgauss.ppf`` takes about a millisecond per call and fails outright
        once the order has grown by a few thousand counts, so the density
        ``alpha s - (a e^{-s} + b e^{s}) / 2`` is integrated on a grid of twelve curvature
        widths either side of its mode, which a log-concave density cannot escape, and the
        cumulative distribution is inverted by interpolation.
        """
        alpha, a, b = belief["alpha"], belief["a"], belief["b"]
        mode = self.rate_map(belief)
        s0 = float(np.log(mode)) if mode > 0 else float(np.log(np.sqrt(a / b)))
        sd = 1.0 / np.sqrt(0.5 * (a * np.exp(-s0) + b * np.exp(s0)))
        s = s0 + sd * np.linspace(-12.0, 12.0, 4001)
        h = alpha * s - 0.5 * (a * np.exp(-s) + b * np.exp(s))
        w = np.exp(h - h.max())
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]))])
        return np.exp(np.interp(q, cdf / cdf[-1], s))


    def support(self, belief, f, tail_tol=1e-9):
        return sichel_support(belief, f, tail_tol=tail_tol)

    def eig(self, belief, f):
        """Expected information gain, memoised on the belief and the exposure.

        Every context of a class starts from the same prior, so the first round needs one
        evaluation per class and action rather than one per context, as for the exact
        log-skew-normal baseline. The value is the same either way.
        """
        key = (belief["alpha"], belief["a"], belief["b"], float(f))
        memo = self.__dict__.setdefault("_memo", {})
        if key not in memo:
            if len(memo) > 200000:
                memo.clear()
            memo[key] = self.predictive_entropy(belief, f) - self.aleatoric_entropy(belief, f)
        return memo[key]

    def logpmf(self, belief, f, y):
        return sichel_logpmf(np.atleast_1d(np.asarray(y, dtype=float)), belief, f)

    def aleatoric_entropy(self, belief, f):
        if f <= 0.0:
            return 0.0
        alpha, a, b = belief["alpha"], belief["a"], belief["b"]

        def h(s):
            return alpha * s - 0.5 * (a * np.exp(-s) + b * np.exp(s))

        mode = ((alpha - 1.0) + np.sqrt((alpha - 1.0) ** 2 + a * b)) / b
        s_mode = float(np.log(mode)) if mode > 0 else float(np.log(np.sqrt(a / b)))
        target = h(s_mode) - LOG_DENSITY_DROP
        bounds = []
        for direction in (-1.0, 1.0):
            step, far = 1.0, s_mode
            while h(far) > target and step < 1e6:
                far = s_mode + direction * step
                step *= 2.0
            near = s_mode
            for _ in range(60):
                mid = 0.5 * (near + far)
                near, far = (mid, far) if h(mid) > target else (near, mid)
            bounds.append(far)
        s0, s1 = bounds
        x, w = _gauss_nodes(QUAD_NODES)
        s = 0.5 * (s1 - s0) * x + 0.5 * (s1 + s0)
        lam = np.exp(s)
        dens = np.exp(gig_logpdf(lam, belief)) * lam
        return float(0.5 * (s1 - s0) * np.sum(w * dens * poisson_entropy(f * lam)))


class GIGPoissonNMC(GIGPoisson):
    """The GIG-Poisson model with its EIG estimated by nested Monte Carlo instead of computed.

    The ablation that isolates the estimator: the same prior, the same exact conjugate
    posterior, and rates drawn from that posterior exactly (SciPy's ``geninvgauss``), so the
    only difference from :class:`GIGPoisson` is how the EIG is obtained. Two estimators:

    ``pce=False``  outer pairs ``(lam_n, y_n)`` and an inner sample of ``M`` rates disjoint
                   from them, ``mean_n [log p(y_n | lam_n) - log mean_m p(y_n | lam_m)]``.
                   Consistent as ``M`` grows, biased upwards at ``O(1/M)``
                   \\citep{rainforth2018nesting}.
    ``pce=True``   the same with each outer rate included in its own inner average, the
                   prior contrastive estimation bound of Foster et al. (2020): a lower bound
                   on the EIG that never exceeds ``log(M + 1)``.

    Memoised at the class priors only, as the sampled log-skew-normal baseline is, so the
    first round needs one estimate per class and action; every later call draws afresh.
    """

    conjugate = True

    def __init__(self, n_outer=1024, n_inner=1024, pce=False, order=-0.5):
        super(GIGPoissonNMC, self).__init__(order=order)
        self.n_outer, self.n_inner, self.pce = int(n_outer), int(n_inner), bool(pce)
        self.rng = np.random.default_rng(0)
        self._priors = set()
        self.name = "gig-pce" if self.pce else "gig-nmc"
        self.label = "GIG-Poisson ({}, N = {}, M = {})".format(
            "PCE" if self.pce else "nested MC", self.n_outer, self.n_inner)

    def set_rng(self, rng):
        self.rng = rng

    def fit_population(self, rates):
        prior = super(GIGPoissonNMC, self).fit_population(rates)
        self._priors.add((prior["alpha"], prior["a"], prior["b"]))
        return prior

    def eig(self, belief, f):
        key = (belief["alpha"], belief["a"], belief["b"])
        if key not in self._priors:
            return self.nmc_eig(belief, f)
        memo = self.__dict__.setdefault("_prior_memo", {})
        if key + (float(f),) not in memo:
            memo[key + (float(f),)] = self.nmc_eig(belief, f)
        return memo[key + (float(f),)]

    def nmc_eig(self, belief, f):
        if f <= 0.0:
            return 0.0
        n, m = self.n_outer, self.n_inner
        lam = gig_sample(belief, n + m, self.rng)
        mu_out, mu_in = f * lam[:n], f * lam[n:]
        y = self.rng.poisson(mu_out).astype(float)
        ly = gammaln(y + 1.0)
        ll_out = -mu_out + y * np.log(mu_out) - ly
        log_mu_in = np.log(mu_in)
        lse = np.empty(n)
        # In blocks of outer draws, so a large inner sample never needs an N x M table.
        for i in range(0, n, 256):
            j = slice(i, min(i + 256, n))
            ll_in = -mu_in[None, :] + y[j, None] * log_mu_in[None, :] - ly[j, None]
            if self.pce:
                ll_in = np.concatenate([ll_in, ll_out[j, None]], axis=1)
            lse[j] = logsumexp(ll_in, axis=1)
        return float(np.mean(ll_out - (lse - np.log(m + 1.0 if self.pce else m))))


# --------------------------------------------------------------------------

class GammaPoisson(CountModel):
    """Gamma mixing. Conjugate; the predictive is negative binomial.

    Two parameters, so the predictive mean and variance are both free but the tail is then
    determined. This is the boundary of the GIG family, and the model whose extra parameter
    the paper is arguing for.
    """

    name = "gamma-poisson"
    label = "gamma-Poisson"
    conjugate = True

    def prior_from_moments(self, mean, var):
        # Gamma(shape, rate): mean = a/b, var = a/b^2.
        return {"a": float(mean * mean / var), "b": float(mean / var)}

    def fit_population(self, rates):
        from scipy.stats import gamma as gamma_dist
        shape, _, scale = gamma_dist.fit(np.asarray(rates, dtype=float), floc=0.0)
        return {"a": float(shape), "b": float(1.0 / scale)}

    def update(self, belief, y, f):
        return {"a": belief["a"] + float(y), "b": belief["b"] + float(f)}

    def rate_mean(self, belief):
        return float(belief["a"] / belief["b"])

    def rate_moments(self, belief):
        a, b = belief["a"], belief["b"]
        return float(a / b), float(a / (b * b))

    def rate_map(self, belief):
        # A shape below one puts the mode at the origin.
        return float(max(belief["a"] - 1.0, 0.0) / belief["b"])

    def sample_rate(self, belief, n, rng):
        return rng.gamma(belief["a"], 1.0 / belief["b"], size=n)

    def rate_quantile(self, belief, q):
        from scipy.stats import gamma as gamma_dist
        return gamma_dist.ppf(q, belief["a"], scale=1.0 / belief["b"])


    def support(self, belief, f, tail_tol=1e-9):
        a, s = belief["a"], f / belief["b"]
        if s <= 0:
            return np.zeros(1, dtype=float)
        ymax = int(nbinom.isf(tail_tol, a, 1.0 / (1.0 + s))) + 16
        return np.arange(min(ymax, 4_000_000) + 1, dtype=float)

    def logpmf(self, belief, f, y):
        a, s = belief["a"], f / belief["b"]
        y = np.atleast_1d(np.asarray(y, dtype=float))
        l1 = np.log1p(s)
        return (gammaln(a + y) - gammaln(a) - gammaln(y + 1.0)
                - a * l1 + y * (np.log(s) - l1))

    def aleatoric_entropy(self, belief, f):
        """Quadrature over the gamma prior, in log-rate to match the other members."""
        if f <= 0.0:
            return 0.0
        a, b = belief["a"], belief["b"]
        # Gamma is log-concave in log-lam: h(s) = a s - b e^s.
        s_mode = float(np.log(a / b))
        target = a * s_mode - b * np.exp(s_mode) - LOG_DENSITY_DROP

        def h(s):
            return a * s - b * np.exp(s)

        bounds = []
        for direction in (-1.0, 1.0):
            step, far = 1.0, s_mode
            while h(far) > target and step < 1e6:
                far = s_mode + direction * step
                step *= 2.0
            near = s_mode
            for _ in range(60):
                mid = 0.5 * (near + far)
                near, far = (mid, far) if h(mid) > target else (near, mid)
            bounds.append(far)
        s0, s1 = bounds
        x, w = _gauss_nodes(QUAD_NODES)
        s = 0.5 * (s1 - s0) * x + 0.5 * (s1 + s0)
        lam = np.exp(s)
        log_dens = a * np.log(b) - gammaln(a) + a * s - b * lam
        return float(0.5 * (s1 - s0) * np.sum(w * np.exp(log_dens)
                                              * poisson_entropy(f * lam)))

    def eig(self, belief, f):
        """Per-term route: average the exact gamma-to-gamma KL over the predictive.

        Every cancellation stays inside one summand, which is the arrangement Remark~1
        recommends and the one the companion project found necessary at large predictive
        mean.
        """
        a, b = belief["a"], belief["b"]
        y = self.support(belief, f)
        a1 = a + y
        kl = ((a1 - a) * digamma(a1) - gammaln(a1) + gammaln(a)
              + a * (np.log(b + f) - np.log(b)) + a1 * (b - (b + f)) / (b + f))
        return float(np.sum(np.exp(self.logpmf(belief, f, y)) * kl))


# --------------------------------------------------------------------------

class LognormalPoisson(CountModel):
    """Lognormal mixing, carried on a grid. Not conjugate.

    The posterior has no closed form, so the belief is a set of weights on a fixed grid in
    log-rate, updated by multiplying by the Poisson likelihood and renormalising. Every
    predictive quantity is then a quadrature over that grid. This is the honest cost of
    leaving the conjugate class: nothing is intractable, but nothing is free either, and the
    grid has to be wide enough to hold the posterior for the whole campaign.
    """

    name = "lognormal-poisson"
    label = "lognormal-Poisson"
    conjugate = False

    def __init__(self, n_grid=128, width=8.0):
        self.n_grid = int(n_grid)
        self.width = float(width)

    def prior_from_moments(self, mean, var):
        sig2 = float(np.log1p(var / (mean * mean)))
        mu = float(np.log(mean) - 0.5 * sig2)
        sig = np.sqrt(sig2)
        s = np.linspace(mu - self.width * sig, mu + self.width * sig, self.n_grid)
        logw = -0.5 * ((s - mu) / sig) ** 2
        logw -= logsumexp(logw)
        return {"s": s, "logw": logw}

    def fit_population(self, rates):
        from scipy.stats import lognorm
        sig, _, scale = lognorm.fit(np.asarray(rates, dtype=float), floc=0.0)
        mu = float(np.log(scale))
        s = np.linspace(mu - self.width * sig, mu + self.width * sig, self.n_grid)
        logw = -0.5 * ((s - mu) / sig) ** 2
        return {"s": s, "logw": logw - logsumexp(logw)}

    def update(self, belief, y, f):
        lam = np.exp(belief["s"])
        ll = -f * lam + y * np.log(f * lam) - gammaln(y + 1.0)
        logw = belief["logw"] + ll
        return {"s": belief["s"], "logw": logw - logsumexp(logw)}

    def _weights(self, belief):
        return np.exp(belief["logw"])

    def rate_mean(self, belief):
        return float(np.sum(self._weights(belief) * np.exp(belief["s"])))

    def rate_moments(self, belief):
        w, lam = self._weights(belief), np.exp(belief["s"])
        m = float(np.sum(w * lam))
        return m, float(np.sum(w * lam * lam) - m * m)

    def rate_map(self, belief):
        """Grid mode of the density in ``lam``, refined by a parabola through its neighbours.

        The weights are masses on a uniform grid in log-rate, so the density in the rate
        itself carries the Jacobian ``1 / lam``. Without the refinement the mode moves in
        steps of the grid spacing, and two contexts with close rates would tie or swap.
        """
        s = belief["s"]
        g = belief["logw"] - s
        i = int(np.argmax(g))
        if 0 < i < s.size - 1:
            den = g[i - 1] - 2.0 * g[i] + g[i + 1]
            if den < 0.0:
                return float(np.exp(s[i] + 0.5 * (s[1] - s[0]) * (g[i - 1] - g[i + 1]) / den))
        return float(np.exp(s[i]))

    def sample_rate(self, belief, n, rng):
        idx = rng.choice(belief["s"].size, size=n, p=self._weights(belief))
        return np.exp(belief["s"][idx])

    def support_and_logpmf(self, belief, f, tail_tol=1e-9):
        """Support and its log mass in one pass, grown until the tail is inside tolerance.

        A moment-based guess is not conservative for a heavy-tailed mixture, and a support
        that is too short biases this model's predictive moments low, which would make it
        look worse for a reason that is ours rather than its own. Returning the mass
        alongside the support avoids evaluating the mixture twice, which is most of the cost
        of an acquisition value for this model.
        """
        m, v = self.rate_moments(belief)
        ymax = int(f * m + 12.0 * np.sqrt(max(f * m + f * f * v, 0.0)) + 64.0)
        for _ in range(20):
            ymax = min(ymax, 200_000)
            y = np.arange(ymax + 1, dtype=float)
            lp = self.logpmf(belief, f, y)
            if 1.0 - np.exp(logsumexp(lp)) < tail_tol or ymax == 200_000:
                return y, lp
            ymax *= 2
        return y, lp

    def support(self, belief, f, tail_tol=1e-9):
        return self.support_and_logpmf(belief, f, tail_tol)[0]

    def predictive_entropy(self, belief, f):
        if f <= 0.0:
            return 0.0
        _, lp = self.support_and_logpmf(belief, f)
        return -float(np.sum(np.exp(lp) * lp))

    def logpmf(self, belief, f, y):
        """Mixture of Poissons over the grid, in log space."""
        y = np.atleast_1d(np.asarray(y, dtype=float))
        lam = np.exp(belief["s"])
        mu = f * lam
        ll = (-mu[None, :] + y[:, None] * np.log(mu)[None, :]
              - gammaln(y + 1.0)[:, None])
        return logsumexp(ll + belief["logw"][None, :], axis=1)

    def aleatoric_entropy(self, belief, f):
        if f <= 0.0:
            return 0.0
        lam = np.exp(belief["s"])
        return float(np.sum(self._weights(belief) * poisson_entropy(f * lam)))


# --------------------------------------------------------------------------

def _discrete_lognormal_logpmf(y, mu, sigma):
    """``log P(y)`` for a lognormal on the counts, binned onto the integers.

    The bins ``[0, 0.5], [0.5, 1.5], [1.5, 2.5], ...`` tile ``[0, inf)`` exactly, so this is
    a proper mass function on the non-negative integers and sums to one without any
    renormalisation. It is also what makes a zero count representable at all: a continuous
    lognormal puts no density at ``y = 0``, and the usual fix of modelling ``log(y + c)``
    buries an arbitrary offset in the likelihood. Here the zero bin is just ``[0, 0.5]``.

    ``mu`` is the *mean* count, so the log-scale location carries the usual ``-sigma^2/2``.
    Differences of the normal CDF cancel badly in either tail, so each side is evaluated with
    whichever of ``ndtr`` and its reflection keeps the subtraction away from one.
    """
    from scipy.special import ndtr

    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-300)
    m = np.log(mu) - 0.5 * sigma * sigma
    hi = (np.log(y + 0.5) - m) / sigma
    with np.errstate(divide="ignore"):
        lo = np.where(y >= 1.0, (np.log(np.maximum(y - 0.5, 1e-300)) - m) / sigma, -np.inf)
    upper = lo > 0.0
    p = np.where(upper, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))
    return np.log(np.maximum(p, 1e-300))


class LogNormalLikelihood(CountModel):
    """A lognormal *likelihood* on the counts, not a lognormal mixing law.

    The other remedy for overdispersion in the standard list, and the one practitioners reach
    for most often: stop modelling the count as a count, take logs, and fit a Gaussian. Here
    the count is lognormal about its mean,

        y | u, lam  ~  LogNormal(log(f(u) lam) - s^2/2, s^2),

    so ``E[y | lam] = f(u) lam`` as in every other model, and ``s`` is a free observation
    scale that absorbs the overdispersion. Note what this is *not*: :class:`LognormalPoisson`
    keeps the Poisson likelihood and puts a lognormal on the rate. This replaces the
    likelihood itself.

    **It is a baseline, and it is included because it should fail in an informative way.**
    Two structural differences from anything built on a Poisson.

    First, the variance is ``(f lam)^2 (e^{s^2} - 1)``, quadratic in the exposure, where a
    Poisson-based model has ``f lam + f^2 V[lam]``. There is no counting floor. Matching a
    count model's dispersion therefore fixes ``s`` at *one* exposure and gets it wrong at
    every other: too little variance where counts are few, too much where they are many.
    :data:`f_ref` is that exposure and it is explicit rather than buried.

    Second, and in its favour, ``s`` is free, so the Fano factor is unconstrained. The
    GIG-Poisson family has ``F(u) = 1 + f(u) V/E > 1`` always and cannot represent
    underdispersion at all; this can.

    The belief over the rate is carried on a grid in log-rate, as in
    :class:`LognormalPoisson`, because binning the lognormal onto the integers costs the
    conjugacy that the continuous version would have had.
    """

    name = "lognormal-likelihood"
    label = "log-normal lik."
    conjugate = False

    def __init__(self, n_grid=128, width=8.0, f_ref=1.0):
        self.n_grid = int(n_grid)
        self.width = float(width)
        #: Exposure at which the observation scale is matched to a Poisson-based model.
        self.f_ref = float(f_ref)

    # -- setup -------------------------------------------------------------

    def _sigma_for(self, mean, var):
        """The observation scale that reproduces a count model's variance at ``f_ref``.

        A Poisson-based model at exposure ``f`` has ``V[y] = f E[lam] + f^2 V[lam]``; this
        model has ``V[y] = f^2 ((e^{s^2} - 1) E[lam^2] + V[lam])``. Equating them at
        ``f_ref`` leaves ``e^{s^2} - 1 = E[lam] / (f_ref E[lam^2])``, which is the whole of
        the calibration and also the whole of the problem: it depends on ``f_ref``.
        """
        e1 = float(mean)
        e2 = float(var + mean * mean)
        return float(np.sqrt(np.log1p(e1 / max(self.f_ref * e2, 1e-300))))

    def _grid(self, mu, sig):
        s = np.linspace(mu - self.width * sig, mu + self.width * sig, self.n_grid)
        logw = -0.5 * ((s - mu) / sig) ** 2
        return {"s": s, "logw": logw - logsumexp(logw)}

    def prior_from_moments(self, mean, var):
        sig2 = float(np.log1p(var / (mean * mean)))
        b = self._grid(float(np.log(mean) - 0.5 * sig2), np.sqrt(sig2))
        b["sigma"] = self._sigma_for(mean, var)
        return b

    def fit_population(self, rates):
        from scipy.stats import lognorm
        x = np.asarray(rates, dtype=float)
        sig, _, scale = lognorm.fit(x, floc=0.0)
        b = self._grid(float(np.log(scale)), float(sig))
        b["sigma"] = self._sigma_for(float(x.mean()), float(x.var()))
        return b

    # -- belief ------------------------------------------------------------

    def update(self, belief, y, f):
        lam = np.exp(belief["s"])
        ll = _discrete_lognormal_logpmf(float(y), f * lam, belief["sigma"])
        logw = belief["logw"] + ll
        return {"s": belief["s"], "logw": logw - logsumexp(logw),
                "sigma": belief["sigma"]}

    def _weights(self, belief):
        return np.exp(belief["logw"])

    def rate_mean(self, belief):
        return float(np.sum(self._weights(belief) * np.exp(belief["s"])))

    def rate_moments(self, belief):
        w, lam = self._weights(belief), np.exp(belief["s"])
        m = float(np.sum(w * lam))
        return m, float(np.sum(w * lam * lam) - m * m)

    def sample_rate(self, belief, n, rng):
        idx = rng.choice(belief["s"].size, size=n, p=self._weights(belief))
        return np.exp(belief["s"][idx])

    def rate_map(self, belief):
        """Grid mode of the density in ``lam``, refined by a parabola through its neighbours.

        Same construction as :meth:`LognormalPoisson.rate_map`, and for the same reason: the
        weights are masses on a uniform grid in log-rate, so the density in the rate carries
        the Jacobian ``1 / lam``, and without the refinement the mode moves in grid steps and
        two contexts with close rates would tie.
        """
        s = belief["s"]
        g = belief["logw"] - s
        i = int(np.argmax(g))
        if 0 < i < s.size - 1:
            den = g[i - 1] - 2.0 * g[i] + g[i + 1]
            if den < 0.0:
                return float(np.exp(s[i] + 0.5 * (s[1] - s[0]) * (g[i - 1] - g[i + 1]) / den))
        return float(np.exp(s[i]))

    # -- predictive --------------------------------------------------------

    def logpmf(self, belief, f, y):
        y = np.atleast_1d(np.asarray(y, dtype=float))
        lam = np.exp(belief["s"])
        ll = _discrete_lognormal_logpmf(y[:, None], (f * lam)[None, :], belief["sigma"])
        return logsumexp(ll + belief["logw"][None, :], axis=1)

    def support_and_logpmf(self, belief, f, tail_tol=1e-9):
        m, v = self.rate_moments(belief)
        s2 = belief["sigma"] ** 2
        var = f * f * ((np.expm1(s2)) * (v + m * m) + v)
        ymax = int(f * m + 12.0 * np.sqrt(max(var, 0.0)) + 64.0)
        for _ in range(20):
            ymax = min(ymax, 200_000)
            y = np.arange(ymax + 1, dtype=float)
            lp = self.logpmf(belief, f, y)
            if 1.0 - np.exp(logsumexp(lp)) < tail_tol or ymax == 200_000:
                return y, lp
            ymax *= 2
        return y, lp

    def support(self, belief, f, tail_tol=1e-9):
        return self.support_and_logpmf(belief, f, tail_tol)[0]

    def predictive_entropy(self, belief, f):
        if f <= 0.0:
            return 0.0
        _, lp = self.support_and_logpmf(belief, f)
        return -float(np.sum(np.exp(lp) * lp))

    def aleatoric_entropy(self, belief, f):
        """``E_lam H[p(y | lam)]``, the binned lognormal's entropy averaged over the grid.

        Summed over the support at every grid point, because the binned lognormal has no
        closed-form entropy where the Poisson does. Subsampling the grid and interpolating
        was tried and rejected: at a stride of eight it bought a factor of 1.2 in time and
        cost up to 0.045 nats at small exposures, which is larger than several of the
        differences this model is meant to be measured against.
        """
        if f <= 0.0:
            return 0.0
        lam = np.exp(belief["s"])
        y = self.support(belief, f)
        lp = _discrete_lognormal_logpmf(y[:, None], (f * lam)[None, :], belief["sigma"])
        h = -np.sum(np.exp(lp) * lp, axis=0)
        return float(np.sum(self._weights(belief) * h))


MODELS = (GIGPoisson, GammaPoisson, LognormalPoisson)

#: Kept out of :data:`MODELS` so that every study already in the repository writes exactly
#: what it wrote before: the gamma-ray correctness gate iterates :data:`MODELS`, and adding a
#: member would silently change its output. :func:`build_model` still finds it by name.
EXTRA_MODELS = (LogNormalLikelihood,)


def build_model(name):
    for cls in MODELS + EXTRA_MODELS:
        if cls.name == name:
            return cls()
    raise ValueError("unknown model {!r}".format(name))
