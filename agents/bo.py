"""Bayesian optimisation over the contexts, on a Gaussian process with a discrete output.

The baseline the manuscript does not yet have: a non-conjugate surrogate with a kernel, run
under a standard optimisation acquisition rather than an information one. The surrogate is
:class:`methods.gppoisson.GPPoisson` -- a Gaussian process over log-rates with a Poisson
observation model and a Laplace posterior -- so the output is discrete and the kernel lets a
measurement at one context move the belief at another. Every model in ``countmodels`` treats
the contexts as independent, so this is the only member of the comparison that can pool.

**What the acquisition scores.** Bayesian optimisation ranks *points*, and a point here is a
pair: which context to measure and how much of it. Expected improvement scores the context
and says nothing about the exposure, so the exposure enters as a fidelity, in the way
multi-fidelity Bayesian optimisation handles it. The value of a pair is the improvement the
context is expected to offer, discounted by how much of that context's uncertainty an
observation of this exposure can actually remove,

    acq(k, v) = EI(k) * rho(k, v),
    rho(k, v) = s_k^2 e lam_k / (1 + s_k^2 e lam_k),

with ``e = f(u)`` the exposure, ``s_k^2`` the posterior variance of the latent and ``lam_k``
the posterior mean rate. The second factor is the fraction of the latent variance a Poisson
observation of that exposure is expected to retire, since its Fisher information about the
latent is ``e lam_k``. The study then divides by cost, as it does for every criterion.

**Two acquisitions, because they answer different questions.** Expected improvement is the
default. Upper confidence bound is included because it is the other standard choice and
because it concentrates differently, and the two disagree about how much to explore.

**A note on what this baseline is for.** Expected improvement optimises: it is built to find
the largest rate, not to predict counts well everywhere. It should be read against the
top-``m`` regret column first and the held-out predictive second, where it is expected to
lose to criteria that spread. That is the comparison, not a defect in the implementation.
"""

import numpy as np
from scipy.stats import norm

from methods.gppoisson import GPPoisson

from .base import Agent

__all__ = ["BayesOptEI", "BayesOptUCB"]


class _BayesOpt(Agent):
    """Shared machinery: a joint GP belief behind the per-context agent protocol."""

    closed_form = False

    def __init__(self, model=None, beta=2.0):
        self.gp = GPPoisson() if model is None else model
        self.model = self.gp
        self.beta = float(beta)

    @property
    def name(self):
        return "{}@{}".format(self.criterion, self.gp.name)

    @property
    def label(self):
        return "{} + {}".format(self.criterion_label, self.gp.label)

    # -- belief ------------------------------------------------------------

    def reset(self, env, prior_moments, recovery, rng=None):
        """One joint belief, not one per context.

        An environment that fits its own priors gives the conjugate families more than two
        moments -- the gamma-ray study fits each family to the population of the source's
        class -- so taking ``prior_moments`` there would start this model behind the rest for
        no reason. Where the environment offers a population fit, we take it through the
        lognormal member, whose two parameters are exactly what the latent needs, and convert
        to moments. Where it does not, the shared moments are the same for everyone.
        """
        if hasattr(env, "initial_beliefs"):
            from methods.countmodels import LognormalPoisson
            ln = LognormalPoisson()
            moments = [tuple(float(x) for x in ln.rate_moments(b))
                       for b in env.initial_beliefs(ln)]
        else:
            moments = list(prior_moments)
        feats = env.context_features() if hasattr(env, "context_features") else None
        classes = env.context_classes() if hasattr(env, "context_classes") else None
        self.gp.reset(moments, features=feats, classes=classes)
        self.recovery = recovery

    def observe(self, k, y, f):
        """An observation anywhere moves the belief everywhere, so nothing is cached."""
        self.gp.observe(k, y, f)

    def rate_mean(self, k):
        return self.gp.rate_mean(k)

    def logpmf(self, k, f, y):
        return self.gp.logpmf(k, f, y)

    # -- acting ------------------------------------------------------------

    def _value(self, k, incumbent):
        raise NotImplementedError

    def act(self, env, rng):
        means = np.array([self.gp.rate_mean(k) for k in range(env.n_contexts)])
        incumbent = float(means.max())
        best = None
        for k in range(env.n_contexts):
            mu, s2 = self.gp.latent(k)
            val = self._value(k, incumbent)
            if val <= 0.0:
                val = 1e-300
            for v in env.action_values:
                e = env.exposure((k, float(v)), self.recovery)
                info = s2 * e * means[k]
                rho = info / (1.0 + info)
                s = val * rho / env.cost((k, float(v)))
                if best is None or s > best[0]:
                    best = (s, k, float(v))
        return best[1], best[2]


class BayesOptEI(_BayesOpt):
    """Expected improvement on the rate, in closed form under the lognormal posterior."""

    criterion = "bo-ei"
    criterion_label = "BO (EI)"

    def _value(self, k, incumbent):
        mu, s2 = self.gp.latent(k)
        s = np.sqrt(max(s2, 1e-300))
        a = max(incumbent, 1e-300)
        la = np.log(a)
        d1 = (mu + s2 - la) / s
        d2 = (mu - la) / s
        return float(np.exp(mu + 0.5 * s2) * norm.cdf(d1) - a * norm.cdf(d2))


class BayesOptUCB(_BayesOpt):
    """Upper confidence bound on the rate, at ``beta`` posterior standard deviations."""

    criterion = "bo-ucb"
    criterion_label = "BO (UCB)"

    def _value(self, k, incumbent):
        mu, s2 = self.gp.latent(k)
        return float(np.exp(mu + self.beta * np.sqrt(max(s2, 1e-300))))
