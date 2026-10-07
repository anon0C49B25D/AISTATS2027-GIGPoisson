"""Contexts that differ in the shape of their prior, not only in its first two moments.

Section 3.3 scores a candidate by the mutual information between the rate and the count it
returns. Local D-optimality scores it by ``log(1 + f V[lam]/E[lam])``, which reads the belief
through one scalar, the index of dispersion. Holding the exposure fixed, the two differ by

    EIG(u) - (1/2) D-opt(u)  ->  c(belief)

as the expected count grows, where ``c`` depends on the *shape* of the mixing law and not on
the action. Both allocation studies in the paper give every context a prior of the same
order, so ``c`` is common to all of them, the two criteria are an affine transform of each
other, and they must rank candidates identically. That is a property of those designs rather
than a fact about the criteria.

This study varies the order across contexts and asks when the difference reaches a decision.

**Two shape classes, matched in mean.** Class ``heavy`` carries order ``-2`` and class
``mild`` order ``-0.5``. Both classes share a prior mean, so a criterion cannot tell them
apart by brightness. Their indices of dispersion are set so that D-optimality *strictly
prefers* the heavy class while expected information gain *strictly prefers* the mild one:
the two criteria are given confident and opposite preferences, and neither is ever reduced
to breaking a tie. The gap is inside the window the offset ``c`` can bridge, which is why
the disagreement exists at all.

**Why the disagreement is the interesting direction.** A large index of dispersion under a
heavy-tailed mixing law is mostly *unresolvable*: it comes from a small probability of a very
large rate, and no exposure recovers it. D-optimality reads that variance as something to be
learned. The mutual information knows how much of it a count can actually remove. So the
heavy class is where D-optimality goes and where there is least to gain.

**Two count levels.** The Gaussian proxy behind ``log(1 + f V/E)`` is worst when the expected
count is small, so the classes are run at a low mean rate and at a high one, with everything
else held fixed.

**No family is told the truth.** Each class has a population of rates drawn from its own law,
and an agent's prior is that family fitted to the population of the class, exactly the
protocol of the gamma-ray study. A conjugate agent recovers an order near the true one; a
two-parameter family fits what it can. Truths for an episode are drawn afresh, so a model
knows what a class looks like in general and nothing about the context in front of it.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from methods.gigpoisson import gig_moments, gig_sample                # noqa: E402

__all__ = ["ShapeScarcity", "REGIMES"]

#: ``(mean, index of dispersion)`` per class, per count level. The heavy class always holds
#: the larger index of dispersion, so D-optimality strictly prefers it at every exposure.
#: Every pair here is reachable at its order: a gamma mixing law of order ``alpha`` caps the
#: index of dispersion at ``E[lam]/alpha``, and the solver clamps silently above that, so
#: these were checked against the achieved moments rather than assumed.
REGIMES = {
    "low":  {"heavy": (0.1, 1.30), "mild": (0.1, 1.00)},
    "high": {"heavy": (1.0, 4.00), "mild": (1.0, 2.00)},
}

#: Orders defining the two classes. The heavy order sits well outside the range a gamma
#: mixing law can reach; the mild order is near the boundary.
ORDERS = {"heavy": -2.0, "mild": -0.5}

#: Rates per class used to fit the priors an agent is given. Large enough that the fit is
#: not itself a source of difference between the families.
N_POPULATION = 4000


class ShapeScarcity(object):
    """Contexts in two shape classes, a fixed exposure grid, and a budget in exposure."""

    label = "contexts of differing prior shape"
    action_label = "exposure"
    action_units = ""

    def __init__(self, n_contexts=12, regime="low", budget=8.0, values=None,
                 eval_exposure=2.0, seed=0):
        if regime not in REGIMES:
            raise ValueError("regime must be one of {}".format(tuple(REGIMES)))
        self.n_ctx = int(n_contexts)
        self.regime = regime
        self.budget = float(budget)
        self.values = (np.array([0.5, 1.0, 2.0, 4.0]) if values is None
                       else np.asarray(values, dtype=float))
        self.eval_exposure = float(eval_exposure)

        #: Classes alternate down the context list so that any prefix of it spans both.
        self._classes = ["heavy" if i % 2 == 0 else "mild" for i in range(self.n_ctx)]

        rng = np.random.default_rng(seed)
        self._truth_law, self.population = {}, {}
        for c, (mean, phi) in REGIMES[regime].items():
            law = _gig_with_dispersion(ORDERS[c], mean, phi)
            self._truth_law[c] = law
            self.population[c] = gig_sample(law, N_POPULATION, rng)
        self._belief_cache = {}

    # -- what the two classes actually are ---------------------------------

    def class_summary(self):
        """``(class, order, achieved mean, achieved index of dispersion)`` per class."""
        out = []
        for c in ("heavy", "mild"):
            m, v = gig_moments(self._truth_law[c])
            out.append((c, ORDERS[c], float(m), float(v / m)))
        return out

    # -- the generic environment protocol ----------------------------------

    n_contexts = property(lambda self: self.n_ctx)
    action_values = property(lambda self: self.values)
    target_exposure = property(lambda self: self.eval_exposure)

    def actions(self):
        return [(k, float(v)) for k in range(self.n_ctx) for v in self.values]

    def exposure(self, u, recovery=None):
        """The action delivers its own exposure; there is no per-context factor here."""
        return float(u[1])

    def cost(self, u):
        """Budget is spent in exposure, so a criterion is scored per unit of it."""
        return float(u[1])

    # -- what every agent is told ------------------------------------------

    def initial_beliefs(self, model):
        """One prior per context: the law its class was drawn from.

        Every criterion in this study runs under the same model and therefore carries the
        same belief, so handing all of them the generating law is the cleanest isolation of
        the acquisition: nothing separates two agents here except how they rank candidates.
        Correct specification is arranged rather than assumed, as in Section 4.1.

        Fitting the class population instead was tried and abandoned. ``geninvgauss.fit`` is
        unreliable on a heavy-tailed sample: on 4000 draws from the ``order -2`` class it
        returned an index of dispersion of ``0.25`` against a true ``1.30``, which collapses
        the designed contrast between the classes and silently turns the study into a
        comparison of two criteria that agree. The design gate in ``run.py`` catches exactly
        that, and it is why the gate is there.

        A family other than the generating one is matched to the class's first two moments,
        which is the protocol the other studies use.
        """
        out = []
        for c in self._classes:
            key = (model.name, c)
            if key not in self._belief_cache:
                if model.name == "gig-poisson":
                    self._belief_cache[key] = dict(self._truth_law[c])
                else:
                    m, v = gig_moments(self._truth_law[c])
                    self._belief_cache[key] = model.prior_from_moments(float(m), float(v))
            out.append(self._belief_cache[key])
        return out

    # -- ground truth ------------------------------------------------------

    def blocks(self, rng):
        """Draw one episode: a true rate per context, from that context's own class law."""
        truths = np.array([float(gig_sample(self._truth_law[c], 1, rng)[0])
                           for c in self._classes])
        moments = [tuple(float(x) for x in gig_moments(self._truth_law[c]))
                   for c in self._classes]
        moments = [(m, v) for m, v in moments]
        return moments, truths, np.ones(self.n_ctx)

    def streams(self, truths, rng):
        """One realised arrival process per context, in exposure.

        Counts from a steady rate form a Poisson process in exposure, so an observation of
        exposure ``f`` consumes the next ``f`` of the stream. Two agents that spend the same
        exposure on the same context therefore see the same counts, which takes the sampling
        noise out of the comparison between them.
        """
        extent = self.budget * 1.05 + max(self.values) + 10.0
        out = []
        for lam in truths:
            n = rng.poisson(lam * extent)
            out.append(np.sort(rng.uniform(0.0, extent, size=n)))
        return out

    def held_out(self, truths, rng, n=24):
        """Counts at a fixed evaluation exposure, for the predictive score."""
        return [rng.poisson(lam * self.eval_exposure, size=n) for lam in truths]


def _gig_with_dispersion(order, mean, phi, tol=1e-12):
    """The mixing law of this order whose mean is ``mean`` and ``V/E`` is ``phi``.

    Bisects the concentration, which the index of dispersion is monotone in over this range.
    Raises rather than clamping if the target is out of reach at this order, because a silent
    clamp would quietly turn a designed contrast into no contrast at all.
    """
    from methods.gigpoisson import gig_from_mean
    lo, hi = 1e-4, 1e4

    def dispersion(omega):
        m, v = gig_moments(gig_from_mean(order, mean, omega))
        return v / m

    if dispersion(lo) < phi:
        raise ValueError(
            "index of dispersion {} unreachable at order {} and mean {} "
            "(the largest available is {:.4f})".format(phi, order, mean, dispersion(lo)))
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if dispersion(mid) > phi:
            lo = mid
        else:
            hi = mid
    law = gig_from_mean(order, mean, np.sqrt(lo * hi))
    m, v = gig_moments(law)
    if abs(v / m - phi) > 1e-4 * phi:
        raise ValueError("bisection did not reach V/E={}; got {:.6f}".format(phi, v / m))
    return law
