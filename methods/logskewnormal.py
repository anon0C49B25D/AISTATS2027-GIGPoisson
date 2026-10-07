"""A Poisson rate with a log-skew-normal prior, sampled by MCMC, scored by nested Monte Carlo.

The non-conjugate route the paper argues against, built the way a practitioner outside a
conjugate family would build it:

    s = log lam  ~  SkewNormal(xi, omega, alpha)     three parameters, like the GIG
    y | lam, u   ~  Poisson(f(u) lam)

**Why this prior.** It is as flexible as the GIG in parameter count: a location, a scale and
a shape that moves mass between the two sides of the log-rate, so it can put a heavy tail
on either side of the rate. It is log-concave in ``s``, so the posterior is unimodal and an
MCMC baseline fails for reasons that belong to the estimator rather than to a pathological
target. And it has no conjugate partner: the posterior is carried as a cloud of samples.

**The belief.** Summing Poissons of a common rate, the likelihood of a context's history is
``T s - F exp(s)`` in the accumulated count ``T`` and exposure ``F``, so the target is
exactly known up to a constant. The belief is ``(prior, T, F)`` plus :data:`N_CHAINS`
parallel Markov chains whose current states are the posterior sample.

**The update.** Each observation moves ``(T, F)``, and every chain then takes
:data:`N_STEPS` Metropolis-Hastings steps on the new posterior, starting from where it was.
Each step is, with equal probability, a random walk at ``2.4`` Laplace standard deviations
or an independence proposal from a Student-t at the Laplace mode. Both kernels leave the
posterior invariant, so their mixture does; the independence move is what lets a chain
that sits in the old, wider posterior reach the new, narrower one in a few steps rather than
by diffusion. The Laplace mode is found by Newton's method, which is safe because the log
target is concave.

**The acquisition.** EIG by nested Monte Carlo: half the chains supply outer pairs
``(lam_i, y_i ~ Poisson(f lam_i))``, the other half the inner evidence estimate, so the two
are independent draws of the same posterior. The estimate is biased up at order ``1/M`` and
noisy at order ``1/sqrt(N)``; both are properties of the estimator this paper replaces, and
neither is corrected here. See ``experiments/synthetic/nmc-cost``.

**What the agent is told.** The same as everyone else. Where only two moments are given
(bulk sampling), the skew is set to zero and location and scale matched in log space, which
is the lognormal moment match; a third moment is not available to anyone. Where a
population of rates is given (gamma-ray), all three parameters are fitted to its logarithm
by maximum likelihood, as each other family is fitted in its own parameters.
"""

import numpy as np
from scipy.special import gammaln, log_ndtr, logsumexp
from scipy.stats import skewnorm

__all__ = ["LogSkewNormalPoisson", "LogSkewNormalQuad", "posterior_grid"]

#: Parallel chains. Half feed the outer sum of the EIG estimate, half the inner.
N_CHAINS = 2048
#: Metropolis-Hastings steps per chain per observation.
N_STEPS = 30
#: Degrees of freedom of the independence proposal.
T_DOF = 4.0
#: Counts above which the exact quadrature takes the predictive tail as continuous.
Y_CUT = 20_000
MAX_NEWTON = 100


#: Constants of the skew-normal penalty ``Q(alpha) = c1 log(1 + c2 alpha^2)``, Eq. (15) of
#: Azzalini and Arellano-Valle (2013): ``c1 = 1 / (4 e2)``, ``c2 = e2 / e1``.
PEN_C1, PEN_C2 = 0.875913, 0.856250


def penalised_fit(s, start):
    """Maximum penalised likelihood ``(xi, omega, alpha)`` of a skew-normal sample ``s``.

    Maximises ``l(theta) - Q(alpha)`` with the penalty of Azzalini and Arellano-Valle (2013),
    which leaves the maximum likelihood estimate nearly unchanged when it is interior and keeps
    the shape finite when the likelihood is maximised only as ``|alpha| -> inf``. Started from
    the maximum likelihood fit with its shape clipped, and from a few symmetric starts.
    """
    from scipy.optimize import minimize
    s = np.asarray(s, dtype=float)

    def nll(th):
        xi, om, al = th[0], np.exp(th[1]), th[2]
        v = -float(np.sum(skewnorm.logpdf(s, al, loc=xi, scale=om)))
        v += PEN_C1 * np.log1p(PEN_C2 * al * al)
        return v if np.isfinite(v) else 1e300

    xi0, om0, al0 = start
    starts = [(xi0, om0, float(np.clip(al0, -20.0, 20.0)))]
    starts += [(float(np.mean(s)), float(np.std(s)) + 1e-9, a) for a in (-3.0, 0.0, 3.0)]
    best = None
    for xi, om, al in starts:
        res = minimize(nll, [xi, np.log(max(om, 1e-9)), al], method="Nelder-Mead",
                       options=dict(maxiter=8000, xatol=1e-9, fatol=1e-11))
        if best is None or res.fun < best.fun:
            best = res
    return float(best.x[0]), float(np.exp(best.x[1])), float(best.x[2])


def _log_prior(s, xi, om, al):
    z = (s - xi) / om
    return -0.5 * z * z + log_ndtr(al * z)


def _log_target(s, prior, T, F):
    return _log_prior(s, *prior) + T * s - F * np.exp(s)


def _mills(x):
    """``phi(x) / Phi(x)``, stable in the lower tail."""
    return np.exp(-0.5 * x * x - 0.5 * np.log(2.0 * np.pi) - log_ndtr(x))


def _laplace(prior, T, F, start):
    """Mode and curvature of the log posterior in ``s``. Newton with step halving."""
    xi, om, al = prior
    s = float(start)

    def derivs(s):
        z = (s - xi) / om
        r = _mills(al * z)
        g = -z / om + (al / om) * r + T - F * np.exp(s)
        h = -1.0 / om ** 2 - (al / om) ** 2 * r * (al * z + r) - F * np.exp(s)
        return g, h

    for _ in range(MAX_NEWTON):
        g, h = derivs(s)
        step = -g / h
        f0 = _log_target(s, prior, T, F)
        t = 1.0
        while t > 1e-8 and not (_log_target(s + t * step, prior, T, F) >= f0 - 1e-12):
            t *= 0.5
        s += t * step
        if abs(t * step) < 1e-10:
            break
    return s, float(-derivs(s)[1])


def _width(prior, T, curv):
    """A posterior width in ``s`` that cannot collapse at a kink.

    The Laplace curvature is a good width wherever the log-posterior is smooth at its mode.
    A fitted skew so large that the prior is effectively a half-normal (a flux cut on a small
    class gives ``alpha ~ 1e8``) puts a kink at the edge, and when the data pull the mode onto
    it the curvature there is enormous and the Laplace width almost zero, although the
    posterior still has the likelihood's spread on the open side. The likelihood of ``T``
    counts has width about ``1 / sqrt(T + 1)`` in ``s``, and the posterior is no wider than
    the prior, so the width is never taken below the smaller of those two.
    """
    sd = 1.0 / np.sqrt(max(curv, 1e-300))
    return max(sd, min(prior[1], 1.0 / np.sqrt(T + 1.0)))


def _start(belief):
    """A starting point for Newton's method: the chains' median, or the prior location."""
    return float(np.median(belief["s"])) if "s" in belief else float(belief["prior"][0])


def posterior_grid(prior, T, F, start, f, ymax):
    """``(s, log w)``: quadrature nodes in ``s = log lam`` and normalised log weights.

    Resolves both the posterior and the Poisson factor of a count up to ``ymax`` at exposure
    ``f``. The prior scale bounds the posterior width from above, and the long side of a
    skew-normal is Gaussian at that scale, so fifteen of whichever is wider covers it. A
    fitted skew can be so large that the prior is a half-normal with a hard edge at its mode
    (a flux cut on a small class does this); a count well below the edge then draws its mass
    from just above it, where the Poisson factor falls over a scale of ``1 / (f lam)``, so a
    patch fifty times finer than that scale is added around the mode, and trapezoidal
    weights take the uneven spacing.
    """
    mode, curv = _laplace(prior, T, F, start)
    sd = _width(prior, T, curv)
    half = 15.0 * max(sd, prior[1] if T == 0 else sd)
    h = min(sd / 20.0, 0.25 / np.sqrt(float(ymax) + 1.0))
    n = int(np.clip(np.ceil(2.0 * half / h) + 1, 401, 400_001))
    h_edge = min(h, 0.02 / (f * np.exp(mode) + 1.0))
    s = np.union1d(np.linspace(mode - half, mode + half, n),
                   mode + h_edge * np.arange(-1000, 1001))
    lw = _log_target(s, prior, T, F) + np.log(np.gradient(s))
    lw = lw - logsumexp(lw)
    # Nodes carrying no mass cost time and change nothing.
    keep = lw > lw.max() - 60.0
    s, lw = s[keep], lw[keep]
    return s, lw - logsumexp(lw)


def _predictive(s, lw, f, y, chunk=256):
    """``log p(y)`` on the grid, in chunks of ``y`` so the matrix stays small.

    Above 50 counts the Poisson factor of ``y`` is concentrated within ``12 / sqrt(y)`` of
    ``log(y / f)`` in ``s`` (outside it the factor is below ``e^-70`` of its peak), so a
    chunk only needs the nodes in that window. That makes a support of ``Y`` counts cost
    about ``100 Y`` rather than ``Y`` times the grid.
    """
    y = np.asarray(y, dtype=float)
    out = np.empty(y.size)
    for i in range(0, y.size, chunk):
        yc = y[i:i + chunk]
        lo, hi = float(yc.min()), float(yc.max())
        if lo >= 50.0:
            pad = 12.0 / np.sqrt(lo)
            j0 = np.searchsorted(s, np.log((lo + 0.5) / f) - pad)
            j1 = np.searchsorted(s, np.log((hi + 0.5) / f) + pad)
            ss, ww = s[j0:j1], lw[j0:j1]
            if ss.size == 0:
                out[i:i + chunk] = -np.inf          # filled in on the full grid below
                continue
        else:
            ss, ww = s, lw
        mu = f * np.exp(ss)
        ll = -mu[None, :] + yc[:, None] * np.log(mu)[None, :] - gammaln(yc + 1.0)[:, None]
        out[i:i + chunk] = logsumexp(ll + ww[None, :], axis=1)
    # A count far in the tail of a narrow posterior can have no node in its window, which
    # would score it as impossible; recompute those on the full grid, where they are finite.
    bad = ~np.isfinite(out)
    if np.any(bad):
        mu = f * np.exp(s)
        yb = y[bad]
        ll = -mu[None, :] + yb[:, None] * np.log(mu)[None, :] - gammaln(yb + 1.0)[:, None]
        out[bad] = logsumexp(ll + lw[None, :], axis=1)
    return out


class LogSkewNormalPoisson(object):
    """Poisson likelihood, log-skew-normal prior on the rate, MCMC posterior, NMC EIG."""

    name = "logskewnormal-poisson"
    label = "log-skew-normal Poisson (MCMC + NMC)"
    conjugate = False

    def __init__(self, n_chains=N_CHAINS, n_steps=N_STEPS, seed=0, n_outer=None, name=None,
                 label=None, penalised=False):
        self.n_chains = int(n_chains)
        self.n_steps = int(n_steps)
        #: Fit the prior by maximum penalised likelihood (Azzalini and Arellano-Valle, 2013)
        #: instead of maximum likelihood, which keeps the shape finite.
        self.penalised = bool(penalised)
        #: ``(xi, omega, alpha)`` of every population fit, for reporting.
        self.fits = []
        #: Outer samples of the nested estimate; the remaining chains are its inner sample.
        #: ``None`` splits the chains in half.
        self.n_outer = None if n_outer is None else int(n_outer)
        self.rng = np.random.default_rng(seed)
        if name is not None:
            self.name = name
        if label is not None:
            self.label = label

    def set_rng(self, rng):
        self.rng = rng

    # -- belief -------------------------------------------------------------

    def _belief(self, xi, om, al):
        s = skewnorm.rvs(al, loc=xi, scale=om, size=self.n_chains, random_state=self.rng)
        return {"prior": (float(xi), float(om), float(al)), "T": 0.0, "F": 0.0, "s": s}

    def prior_from_moments(self, mean, var):
        """Skew zero, location and scale from the lognormal moment match."""
        sig2 = float(np.log1p(var / (mean * mean)))
        return self._belief(float(np.log(mean) - 0.5 * sig2), np.sqrt(sig2), 0.0)

    def fit_population(self, rates):
        s = np.log(np.asarray(rates, dtype=float))
        al, xi, om = skewnorm.fit(s)
        if self.penalised:
            xi, om, al = penalised_fit(s, (xi, om, al))
        self.fits.append((float(xi), float(om), float(al)))
        return self._belief(xi, om, al)

    def update(self, belief, y, f):
        prior = belief["prior"]
        T, F = belief["T"] + float(y), belief["F"] + float(f)
        s = belief["s"].copy()
        mode, curv = _laplace(prior, T, F, np.median(s))
        sd = _width(prior, T, curv)
        rng, n = self.rng, s.size
        lt = _log_target(s, prior, T, F)
        tscale = 1.5 * sd

        def log_q(x):
            z = (x - mode) / tscale
            return -0.5 * (T_DOF + 1.0) * np.log1p(z * z / T_DOF)

        for _ in range(self.n_steps):
            indep = rng.random(n) < 0.5
            prop = np.where(indep, mode + tscale * rng.standard_t(T_DOF, size=n),
                            s + 2.4 * sd * rng.standard_normal(n))
            lp = _log_target(prop, prior, T, F)
            log_a = lp - lt + np.where(indep, log_q(s) - log_q(prop), 0.0)
            acc = np.log(rng.random(n)) < log_a
            s = np.where(acc, prop, s)
            lt = np.where(acc, lp, lt)
        return {"prior": prior, "T": T, "F": F, "s": s}

    def rate_mean(self, belief):
        return float(np.mean(np.exp(belief["s"])))

    def rate_moments(self, belief):
        lam = np.exp(belief["s"])
        return float(lam.mean()), float(lam.var())

    def sample_rate(self, belief, n, rng):
        return np.exp(rng.choice(belief["s"], size=n))

    # -- predictive ---------------------------------------------------------

    def logpmf(self, belief, f, y):
        """The model's own posterior predictive, by quadrature of prior times likelihood.

        Used to *score* the model, not by the agent to decide: its designs come from the
        nested estimate in :meth:`eig`. Averaging the Poisson over the chains is unbiased for
        ``p(y)`` but its logarithm is not, and when the held-out count is large the Poisson
        factor is far narrower in ``s`` than the spacing of the samples: on unobserved hard
        X-ray targets at 20 ks the sample average overstated the NLPD by up to 94 nats. The
        target is known in closed form up to a constant, so the predictive is computed from
        it directly, on a grid that resolves both it and the Poisson factor.
        """
        y = np.atleast_1d(np.asarray(y, dtype=float))
        s, lw = posterior_grid(belief["prior"], belief["T"], belief["F"],
                               _start(belief), f, float(y.max()))
        return _predictive(s, lw, f, y)

    # -- acquisition --------------------------------------------------------

    def eig(self, belief, f):
        """Nested Monte Carlo, memoised at the prior.

        Every context of a class starts from the same prior and the same chains, so the
        first round needs one estimate per class and action rather than one per context, as
        for the conjugate models and the exact quadrature. Once a context has data its
        chains are its own and every call draws a fresh estimate.
        """
        if belief["T"] == 0.0 and belief["F"] == 0.0:
            key = (belief["prior"], id(belief["s"]), float(f))
            memo = self.__dict__.setdefault("_prior_memo", {})
            if key not in memo:
                memo[key] = self.nmc_eig(belief, f)
            return memo[key]
        return self.nmc_eig(belief, f)

    def nmc_eig(self, belief, f):
        """Nested Monte Carlo: outer pairs from some chains, the evidence from the others."""
        if f <= 0.0:
            return 0.0
        s = belief["s"]
        h = s.size // 2 if self.n_outer is None else self.n_outer
        mu_out, mu_in = f * np.exp(s[:h]), f * np.exp(s[h:])
        y = self.rng.poisson(mu_out).astype(float)
        ly = gammaln(y + 1.0)
        ll_out = -mu_out + y * np.log(mu_out) - ly
        ll_in = -mu_in[None, :] + y[:, None] * np.log(mu_in)[None, :] - ly[:, None]
        return float(np.mean(ll_out - (logsumexp(ll_in, axis=1) - np.log(mu_in.size))))


class LogSkewNormalQuad(LogSkewNormalPoisson):
    """The same prior and likelihood, with every quantity computed by quadrature.

    The exact non-conjugate baseline. With a Poisson likelihood the posterior of any
    one-dimensional prior is known up to a constant, ``prior(s) exp(T s - F e^s)`` in the
    accumulated count ``T`` and exposure ``F``, so nothing needs sampling: the posterior, the
    predictive and both terms of the expected information gain are quadratures on
    :func:`posterior_grid`. Against :class:`LogSkewNormalPoisson` it isolates what the
    sampler and the nested estimator cost; against the GIG it isolates what conjugacy buys
    beyond exactness. It does not scale past one dimension, which is why the sampled route is
    what a practitioner would reach for.

    The information gain is memoised on ``(prior, T, F, f)``: every context of a class starts
    from the same prior, so the first round needs one evaluation per class and action rather
    than one per context.
    """

    name = "logskewnormal-quad"
    label = "log-skew-normal Poisson (exact quadrature)"

    def __init__(self, tail_tol=1e-6, **kw):
        super(LogSkewNormalQuad, self).__init__(**kw)
        self.tail_tol = float(tail_tol)
        self._memo = {}

    def _belief(self, xi, om, al):
        return {"prior": (float(xi), float(om), float(al)), "T": 0.0, "F": 0.0}

    def update(self, belief, y, f):
        return {"prior": belief["prior"], "T": belief["T"] + float(y),
                "F": belief["F"] + float(f)}

    def _grid(self, belief, f=1.0, ymax=0.0):
        return posterior_grid(belief["prior"], belief["T"], belief["F"], _start(belief),
                              f, ymax)

    def rate_mean(self, belief):
        s, lw = self._grid(belief)
        return float(np.exp(logsumexp(lw + s)))

    def rate_moments(self, belief):
        s, lw = self._grid(belief)
        m = float(np.exp(logsumexp(lw + s)))
        return m, float(np.exp(logsumexp(lw + 2.0 * s)) - m * m)

    def rate_quantile(self, belief, q):
        s, lw = self._grid(belief)
        cdf = np.cumsum(np.exp(lw))
        return np.exp(np.interp(q, cdf, s))

    def sample_rate(self, belief, n, rng):
        return self.rate_quantile(belief, rng.random(n))

    def _predictive_entropy(self, belief, f):
        """``H[p(y | u)]``: an exact sum over the counts, with a controlled far tail.

        The support is grown until the predictive mass beyond it, the posterior average of
        the Poisson survival function, is below ``tail_tol``. Counts up to :data:`Y_CUT` are
        summed exactly. Beyond it the Poisson noise is below one per cent of the rate, and
        the predictive is the density of ``f lam`` to within ``O(1 / y)``, so the entropy of
        that tail is the integral of ``-p log p`` of that density, taken on the posterior
        grid. A heavy-tailed posterior at a long dwell can need millions of counts to reach
        the tolerance, which a plain sum cannot afford.
        """
        from scipy.stats import poisson
        m, v = self.rate_moments(belief)
        ymax = int(f * m + 12.0 * np.sqrt(f * m + f * f * v) + 64.0)
        s, lw = self._grid(belief, f, min(ymax, Y_CUT))
        for _ in range(30):
            tail = float(np.sum(np.exp(lw) * poisson.sf(ymax, f * np.exp(s))))
            if tail < self.tail_tol:
                break
            ymax *= 2
        y_exact = np.arange(min(ymax, Y_CUT) + 1, dtype=float)
        lp = _predictive(s, lw, f, y_exact)
        fin = np.isfinite(lp)       # counts with no mass on the grid contribute nothing
        h = -float(np.sum(np.exp(lp[fin]) * lp[fin]))
        if ymax > Y_CUT:
            # Mass of the counts beyond Y_CUT, and the entropy of the density of f lam there.
            beyond = s > np.log((Y_CUT + 0.5) / f)
            if np.any(beyond):
                lw_b = lw[beyond]
                log_dens = lw_b - np.log(np.gradient(s)[beyond]) - np.log(f) - s[beyond]
                h -= float(np.sum(np.exp(lw_b) * log_dens))
        return h

    def eig(self, belief, f):
        if f <= 0.0:
            return 0.0
        key = (belief["prior"], belief["T"], belief["F"], float(f))
        if key not in self._memo:
            from .countmodels import poisson_entropy
            h_pred = self._predictive_entropy(belief, f)
            s, lw = self._grid(belief, f, 0.0)
            aleatoric = float(np.sum(np.exp(lw) * poisson_entropy(f * np.exp(s))))
            self._memo[key] = h_pred - aleatoric
        return self._memo[key]
