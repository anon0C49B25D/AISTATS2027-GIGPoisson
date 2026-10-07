"""The bulk-sampling environment: the ground truth, the actions, and the budget.

The setting Sichel invented the GIGP for. Diamonds occur in clusters, so stone counts from a
sample of gravel are far more variable than a Poisson process allows, and no standard
discrete law reproduces observed stone-count frequencies. That model is used to *value* a
deposit once samples are in hand. It is not used to decide which samples to take.

A property is divided into ``K`` blocks. Block ``k`` carries an unknown stone density
``lam_k`` in stones per cubic metre::

    lam_k          ~  GIG(alpha, a_k, b_k)          drawn once per block
    u = (k, v)                                      block, and volume of gravel to process
    f(u) = v * r_k                                  effective exposure
    y | lam_k, u   ~  Poisson( f(u) lam_k )         stones recovered
    cost(u) = c0 + v                                budget spent, in m^3; c0 = 0 unless set

``r_k`` is the plant's known recovery factor for that material. It multiplies the exposure,
so it is exactly the attenuation of Assumption 1.

**This object holds no model.** It knows what is true and what an action costs; it does not
know what any agent believes. Agents carry their own beliefs, because a conjugate agent
carries a few numbers and a non-conjugate one carries a grid, and the study compares them.

**What every agent is told.** The environment reports, per block, the mean and variance of
the law its rate was drawn from. That is the same prior information for everyone. An agent
whose family can reproduce those two moments and the shape besides is better placed than one
that can only match the moments, and measuring that difference is the point.

**The budget is in cubic metres, not in samples.** A policy that always requests the largest
sample learns more per round while spending the budget many times faster, so counting the
budget in rounds would measure appetite rather than judgement.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from methods.gigpoisson import gig_from_mean, gig_moments, gig_sample   # noqa: E402

__all__ = ["BulkSampling", "FaciesBulkSampling"]


class BulkSampling(object):
    """Ground truth, action set and budget for one sampling campaign."""

    label = "alluvial diamond bulk sampling"
    action_label = "sample volume"
    action_units = "m$^3$"

    def __init__(self, n_blocks=12, volumes=None, order=-0.5,
                 grade_range=(0.05, 4.0), omega_range=(0.3, 2.0),
                 recovery_range=(0.55, 0.98), budget=240.0, eval_volume=5.0, overhead=0.0):
        self.n_blocks = int(n_blocks)
        #: Pit schedule: the sample volumes the plant is set up to process, in m^3.
        self.volumes = (np.array([1.0, 2.5, 5.0, 10.0, 20.0, 40.0]) if volumes is None
                        else np.asarray(volumes, dtype=float))
        self.order = float(order)
        self.grade_range = grade_range
        self.omega_range = omega_range
        self.recovery_range = recovery_range
        self.budget = float(budget)
        #: Volume of a held-out sample. Prediction is scored here.
        self.eval_volume = float(eval_volume)
        #: Budget charged per sample on top of the gravel processed (moving the plant to the
        #: block, setting up), in m^3 of processing. Zero in the paper's main settings.
        self.overhead = float(overhead)

    # -- the generic environment protocol ----------------------------------

    #: Agents address contexts and choose an action value; the domain names sit beside.
    n_contexts = property(lambda self: self.n_blocks)
    action_values = property(lambda self: self.volumes)
    #: Exposure of the reference observation a prediction-oriented criterion aims at.
    target_exposure = property(lambda self: self.eval_volume)

    # -- the decision problem ----------------------------------------------

    def actions(self):
        return [(k, float(v)) for k in range(self.n_blocks) for v in self.volumes]

    def exposure(self, u, recovery):
        """``f(u) = v r_k``, the effective volume of gravel the plant sees."""
        k, v = u
        return float(v) * float(recovery[k])

    def cost(self, u):
        """Budget spent by an action, in cubic metres of gravel processed."""
        return self.overhead + float(u[1])

    # -- ground truth ------------------------------------------------------

    def blocks(self, rng):
        """Draw a property.

        Returns ``(prior_moments, truths, recovery)``, where ``prior_moments`` is the mean
        and variance of the law each rate was drawn from, which is what every agent is told,
        and ``truths`` are the rates themselves, which no agent is told.

        Blocks differ in expected grade, in how clustered the stones are at that grade, and
        in plant recovery, so an agent that tracks only one of the three is separable from
        one that tracks all of them.
        """
        prior_moments, truths, recovery = [], [], []
        for _ in range(self.n_blocks):
            grade = float(np.exp(rng.uniform(*np.log(self.grade_range))))
            omega = float(np.exp(rng.uniform(*np.log(self.omega_range))))
            law = gig_from_mean(self.order, grade, omega)
            prior_moments.append(gig_moments(law))
            truths.append(float(gig_sample(law, 1, rng)[0]))
            recovery.append(float(rng.uniform(*self.recovery_range)))
        return prior_moments, np.asarray(truths), np.asarray(recovery)

    def stone_fields(self, truths, recovery, rng):
        """One realised stone field per block, as a Poisson process in processed volume.

        Sampling ``v`` cubic metres of block ``k`` consumes ``v r_k`` metres of its field, so
        two agents that process the same gravel recover the same stones and any difference
        between them is a difference in decisions rather than in luck.
        """
        fields = []
        for k in range(self.n_blocks):
            extent = self.budget * float(recovery[k]) * 1.05 + 10.0
            n = int(rng.poisson(truths[k] * extent))
            fields.append(np.sort(rng.uniform(0.0, extent, size=n)))
        return fields

    def held_out(self, truths, recovery, rng, n=24):
        """Fresh counts at the evaluation volume, shared by every agent."""
        return [rng.poisson(self.eval_volume * recovery[k] * truths[k], size=n)
                for k in range(self.n_blocks)]


class FaciesBulkSampling(BulkSampling):
    """Bulk sampling of a deposit whose blocks belong to facies, with priors from a small
    calibration catalogue.

    The same decision problem as :class:`BulkSampling`, posed the way the two telescope
    studies pose theirs. Every block belongs to one of a few facies, the sedimentary units of
    the deposit, and the grades of a facies follow its own GIG law,

        lam_k  ~  GIG(order_c, a_c, b_c)          c the facies of block k,

    so the population of grades is heavy-tailed: a few blocks carry most of the stones.
    No agent is told those laws. What a mine has instead is the grade of a few blocks of each
    facies that were mined before, and every model fits its own family to that catalogue by
    maximum likelihood, exactly as the telescope studies fit theirs to the catalogued
    members of a source class. The catalogue holds ``pop_size`` grades per facies, one fixed
    sample drawn from the facies law, independent of the blocks an episode draws.

    The property is large against the budget: ``n_blocks`` blocks share ``budget`` cubic
    metres, so most blocks are sampled once or not at all and what is predicted for them
    rests on the prior. Prediction is scored on held-out samples of ``eval_volume`` cubic
    metres, the volume of a production sample.
    """

    label = "alluvial diamond bulk sampling, facies priors"

    #: ``(name, mean grade in stones per m^3, order, concentration omega)`` per facies.
    #: Mean grades span an order of magnitude, and the orders the range the measured rate
    #: fields of the two telescope studies show (between -0.5 and -2.6).
    FACIES = (("f1", 0.25, -0.5, 0.3),
              ("f2", 0.5, -1.0, 0.3),
              ("f3", 1.0, -1.5, 0.3),
              ("f4", 2.0, -2.0, 0.1))

    def __init__(self, n_blocks=96, volumes=None, budget=240.0, eval_volume=20.0,
                 recovery_range=(0.55, 0.98), pop_size=20, pop_seed=7, facies=None,
                 overhead=0.0):
        super(FaciesBulkSampling, self).__init__(
            n_blocks=n_blocks, volumes=volumes, recovery_range=recovery_range,
            budget=budget, eval_volume=eval_volume, overhead=overhead)
        spec = self.FACIES if facies is None else tuple(facies)
        self.CLASSES = tuple(s[0] for s in spec)
        #: The law each facies draws its grades from. Ground truth; no agent sees it.
        self.laws = {name: gig_from_mean(order, mean, omega)
                     for name, mean, order, omega in spec}
        self._classes = [self.CLASSES[i % len(self.CLASSES)] for i in range(self.n_blocks)]
        rng = np.random.default_rng(pop_seed)
        #: The calibration catalogue: grades of previously mined blocks, per facies.
        self.population = {c: gig_sample(self.laws[c], int(pop_size), rng)
                           for c in self.CLASSES}

    # -- what every agent is told ------------------------------------------

    def context_classes(self):
        """The facies of each block, for a kernel over discrete context labels."""
        return list(self._classes)

    def initial_beliefs(self, model):
        """One prior per block: this family, fitted to the catalogue of the block's facies."""
        cache, out = {}, []
        for c in self._classes:
            key = (model.name, c)
            if key not in cache:
                cache[key] = model.fit_population(self.population[c])
            out.append(cache[key])
        return out

    # -- ground truth ------------------------------------------------------

    def blocks(self, rng):
        """Draw a property: one grade per block from its facies law, and a plant recovery.

        ``prior_moments`` are the catalogue mean and variance of the block's facies, the
        two-moment summary of what every agent is told.
        """
        truths = np.array([float(gig_sample(self.laws[c], 1, rng)[0]) for c in self._classes])
        recovery = rng.uniform(*self.recovery_range, size=self.n_blocks)
        moments = [(float(self.population[c].mean()), float(self.population[c].var()))
                   for c in self._classes]
        return moments, truths, recovery
