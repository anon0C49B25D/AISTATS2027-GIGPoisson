"""Allocating telescope time across gamma-ray sources.

A pointed observation looks at one source at a time. The instrument is a photon counter, so
the number of photons recorded in an integration is Poisson about the source's flux times
the telescope's effective area times the time on target::

    lam_k                             true photon flux of source k, from the catalogue
    u = (k, t)                        which source, and how long to integrate
    f(u) = A_eff * t                  exposure, in cm^2 s
    y | lam_k, u  ~  Poisson(f(u) lam_k)
    cost(u) = c0 + t                  budget spent, in megaseconds; c0 = 0 unless set

That is the whole of the observing model, and it is Assumption 1 exactly: the action scales
the rate through a known constant and cannot change which source is being measured.

**What is real here.** The rates are not simulated. They are the integral photon fluxes above
1 GeV of 4557 sources in the Fermi-LAT fourth source catalogue, and the counting process is
the one that measured them. The previous study generates its rates from the GIG-Poisson law
itself, so a model that assumes that law is correct by construction; here no model in the
comparison produced the field it is asked to learn.

**Priors are conditioned on source class, and that is where the problem lives.** A gamma-ray
observer knows what kind of object a target is before pointing at it, and the classes are not
alike: in this catalogue pulsars have a median flux twelve times that of blazar candidates
and a variance-to-mean ratio twelve hundred times larger. Giving every source the same
population prior would make the targets exchangeable, and a schedule that simply visits each
in turn would then be near-optimal by construction, leaving nothing for any acquisition to
find. Conditioning on class restores the heterogeneity that is actually there: some targets
are already well constrained by what they are, and some are not.

**No leakage.** The catalogue is split in two, within each class. One half supplies the
population an agent fits its class priors to; the other supplies the sources an episode
observes. A model therefore knows what pulsars look like in general and nothing about the
particular pulsar in front of it, which is the position a real observer is in.

**The budget is scarce relative to the target list, and that is deliberate.** Forty-eight
targets share 120 Ms, so most of them cannot be observed at all and the campaign must decide
which to leave alone. An abundant budget makes this a different problem: if every target can
be measured well whatever the schedule, the prior differences wash out and visiting each in
turn is near-optimal, so no acquisition rule can distinguish itself. Scarcity is the regime
adaptive allocation exists for, and it is the regime a real observing proposal is written in.

**Two simplifications, both stated rather than hidden.** The brightest and faintest five per
cent of the catalogue are excluded: the faint end sits at the detection threshold where a
flux-limited catalogue is incomplete, and a handful of exceptionally bright sources would
otherwise consume any budget by themselves. And there is no diffuse background. A real
observation counts source photons plus a known background, which is an *additive* term and
so falls outside Assumption 1; carrying one would need a different conjugacy argument and is
left to the discussion.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from fetch import fetch                                               # noqa: E402

__all__ = ["PhotonCounting"]

#: Fluxes are held in units of 1e-10 photons per square centimetre per second, and the
#: exposure constant below absorbs the same factor, so counts come out in photons.
#: 8000 cm^2 is the order of the Fermi-LAT effective area above 1 GeV.
EXPOSURE_PER_MS = 8000.0 * 1e6 * 1e-10


class PhotonCounting(object):
    """Ground truth, action set and budget for one observing campaign."""

    label = "gamma-ray source photometry"
    action_label = "integration time"
    action_units = "Ms"

    #: Classes with enough members to fit a prior to each half of the catalogue. Rarer
    #: classes are pooled into the unassociated group rather than fitted on a handful of
    #: sources, which would put noise in the prior and call it heterogeneity.
    CLASSES = ("psr", "fsrq", "bll", "bcu", "unassoc")

    def __init__(self, n_sources=48, times=None, budget=120.0, eval_time=5.0,
                 band=(5.0, 95.0), split_seed=0, pop_size=None, pop_seed=7, overhead=0.0):
        self.n_sources = int(n_sources)
        #: Integration times the scheduler may grant, in megaseconds.
        self.times = (np.array([0.5, 1.0, 2.5, 5.0, 10.0, 25.0]) if times is None
                      else np.asarray(times, dtype=float))
        self.budget = float(budget)
        self.eval_time = float(eval_time)
        #: Time charged per observation on top of the integration (slewing, settling), in
        #: megaseconds. Zero in the paper's main settings, where cost is proportional to
        #: exposure and so the shortest integration always maximises information per cost.
        self.overhead = float(overhead)

        flux, klass = fetch(with_class=True)
        flux = flux * 1e10
        lo, hi = np.percentile(flux, band[0]), np.percentile(flux, band[1])
        keep = (flux >= lo) & (flux <= hi)
        flux, klass = flux[keep], klass[keep]
        klass = np.array([k if k in self.CLASSES else "unassoc" for k in klass],
                         dtype=object)

        #: Which class each slot in the target list holds. Fixed at construction rather
        #: than assigned when a programme is drawn, so that the priors an agent is given do
        #: not depend on whether a programme has been drawn yet.
        self._classes = [self.CLASSES[i % len(self.CLASSES)]
                         for i in range(self.n_sources)]

        rng = np.random.default_rng(split_seed)
        #: Per class: the half priors are fitted to, and the half episodes draw from.
        self.population, self.targets = {}, {}
        for c in self.CLASSES:
            sub = flux[klass == c]
            order = rng.permutation(sub.size)
            half = sub.size // 2
            self.population[c] = sub[order[:half]]
            self.targets[c] = sub[order[half:]]
        # A small calibration catalogue: priors fitted to only ``pop_size`` known members of
        # each class, as for a rare class, a new instrument or a pilot survey. One fixed
        # sample per class, drawn from the prior half; the targets are unchanged.
        if pop_size is not None:
            rng_pop = np.random.default_rng(pop_seed)
            self.population = {c: rng_pop.choice(v, min(int(pop_size), v.size), replace=False)
                               for c, v in self.population.items()}

    # -- the generic environment protocol ----------------------------------

    n_contexts = property(lambda self: self.n_sources)
    action_values = property(lambda self: self.times)
    target_exposure = property(lambda self: self.eval_time * EXPOSURE_PER_MS)

    # -- the decision problem ----------------------------------------------

    def actions(self):
        return [(k, float(t)) for k in range(self.n_sources) for t in self.times]

    def exposure(self, u, recovery=None):
        """``f(u) = A_eff t``. No per-source factor: a pointed observation is a pointing."""
        return float(u[1]) * EXPOSURE_PER_MS

    def cost(self, u):
        """Budget spent by an observation, in megaseconds of telescope time."""
        return self.overhead + float(u[1])

    # -- what every agent is told ------------------------------------------

    def context_classes(self):
        """The source class of each target, for a kernel over discrete context labels."""
        return list(self._classes)

    def initial_beliefs(self, model):
        """One prior per source: this family, fitted to this source's class.

        Cached per (model, class), since fitting is the same work for every source of a
        class and the campaign asks for it once per programme.
        """
        cache = {}
        out = []
        for c in self._classes:
            key = (model.name, c)
            if key not in cache:
                cache[key] = model.fit_population(self.population[c])
            out.append(cache[key])
        return out

    # -- ground truth ------------------------------------------------------

    def blocks(self, rng):
        """Draw an observing programme: a mixed target list across source classes.

        Classes are drawn in round-robin order so that a programme always spans the range of
        prior knowledge, from a blazar candidate whose flux is already tightly constrained by
        its class to a pulsar whose is not. That mixture is the decision problem: a fixed
        budget, and targets that differ in how much there is left to learn about them.
        """
        truths = np.array([self.targets[c][rng.integers(self.targets[c].size)]
                           for c in self._classes])
        moments = [(float(self.population[c].mean()), float(self.population[c].var()))
                   for c in self._classes]
        return moments, truths, np.ones(self.n_sources)

    def stone_fields(self, truths, recovery, rng):
        """One realised photon arrival stream per source, in exposure.

        Photons from a steady source form a Poisson process in exposure, so an observation
        of exposure ``f`` consumes the next ``f`` of that stream. Two agents that spend the
        same time on the same source therefore record the same photons.
        """
        streams = []
        extent = self.budget * EXPOSURE_PER_MS * 1.05 + 10.0
        for k in range(self.n_sources):
            n = int(rng.poisson(truths[k] * extent))
            streams.append(np.sort(rng.uniform(0.0, extent, size=n)))
        return streams

    def held_out(self, truths, recovery, rng, n=24):
        """Fresh photon counts at the reference integration time, shared by every agent."""
        f_eval = self.eval_time * EXPOSURE_PER_MS
        return [rng.poisson(f_eval * truths[k], size=n) for k in range(self.n_sources)]
