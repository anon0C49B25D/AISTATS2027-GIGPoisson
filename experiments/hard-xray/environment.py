"""Scheduling hard X-ray follow-up of Swift-BAT sources with a pointed telescope.

The same decision problem as the gamma-ray study, in a different sky: a focusing hard X-ray
telescope has a fixed allocation of time and a list of Swift-BAT-selected targets far longer
than it can observe well, and must decide which to point at and for how long::

    lam_k                             hard X-ray flux of target k, from the BAT catalogue
    u = (k, t)                        which target, and how many kiloseconds to integrate
    f(u) = C t                        exposure: C converts flux to counts per kilosecond
    y | lam_k, u  ~  Poisson(f(u) lam_k)
    cost(u) = c0 + t                  budget spent, in kiloseconds; c0 = 0 unless set

**What is real.** The rates are the 14-195 keV fluxes of the 1632 sources of the Swift-BAT
105-month survey. No model in the comparison produced them.

**Why the tail is not a gamma's.** The survey is flux-limited over sources spread through
space, so its flux distribution follows the source counts: for a Euclidean population
``N(>S) ~ S^{-3/2}``, a power-law tail. A gamma mixing law has an exponential tail and cannot
follow it; a GIG of negative order has a power-law tail and can. Fitted per class on the
1-99 per cent band, every class has a negative order, between -0.3 and -3.4.

**The count scale.** ``counts_per_ks`` converts catalogue flux (1e-12 erg cm^-2 s^-1 in
14-195 keV) to detected counts per kilosecond. Over the full 3-79 keV band a NuSTAR-like
telescope records about 31: a photon index of 1.8 puts 37 per cent of the 14-195 keV flux in
2-10 keV, and a 1 mCrab source (2.4e-11 erg cm^-2 s^-1 in 2-10 keV) gives about one count per
second in each of two modules. The default, 3.1, is the hard band above 20 keV alone: for
the same spectrum 16 per cent of the 3-79 keV photons arrive above 20 keV, where the
effective area is roughly half its peak, so about a tenth of the full-band rate. That band
overlaps the one BAT measures, and it is what hard X-ray follow-up of BAT-selected active
galaxies is designed around. At this scale a 0.5 ks snapshot of a median target records
about 22 counts.

**Priors are conditioned on source class**, fitted to one half of each class, with targets
drawn from the other half, exactly as in the gamma-ray study. Classes with fewer than 75
catalogue members are pooled into ``other``, together with the unidentified sources.

**Simplifications, as in the gamma-ray study**: no background (an additive term outside
Assumption 1), time-averaged rather than variable fluxes, and a flux-limited parent sample.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch_bat import fetch                                             # noqa: E402

__all__ = ["HardXRayFollowup", "source_class"]


def source_class(kind):
    """Group the catalogue's source types into classes large enough to fit a prior to."""
    t = str(kind).strip()
    if t in ("Sy1", "Sy1.2", "Sy1.5", "Sy1;broad-line AGN"):
        return "sy1"
    if t in ("Sy1.8", "Sy1.9", "Sy2", "Sy2 candidate"):
        return "sy2"
    if t == "Beamed AGN":
        return "beamed"
    if t in ("LMXB", "HMXB", "XRB"):
        return "xrb"
    if t == "CV":
        return "cv"
    return "other"


class HardXRayFollowup(object):
    """Ground truth, action set and budget for one follow-up programme."""

    label = "hard X-ray follow-up"
    action_label = "integration time"
    action_units = "ks"

    CLASSES = ("sy1", "sy2", "beamed", "xrb", "cv", "other")

    def __init__(self, n_sources=192, times=None, budget=120.0, eval_time=20.0,
                 band=(1.0, 99.0), counts_per_ks=3.1, split_seed=0, pop_size=None,
                 pop_seed=7, overhead=0.0):
        self.n_sources = int(n_sources)
        #: Integration times the scheduler may grant, in kiloseconds.
        self.times = (np.array([0.5, 1.0, 2.5, 5.0, 10.0, 25.0]) if times is None
                      else np.asarray(times, dtype=float))
        self.budget = float(budget)
        self.eval_time = float(eval_time)
        self.counts_per_ks = float(counts_per_ks)
        #: Time charged per observation on top of the integration (slewing, settling), in
        #: kiloseconds. Zero in the paper's main settings.
        self.overhead = float(overhead)

        flux, kind = fetch()
        lo, hi = np.percentile(flux, band[0]), np.percentile(flux, band[1])
        keep = (flux >= lo) & (flux <= hi)
        flux = flux[keep]
        klass = np.array([source_class(k) for k in kind[keep]], dtype=object)

        self._classes = [self.CLASSES[i % len(self.CLASSES)] for i in range(self.n_sources)]
        rng = np.random.default_rng(split_seed)
        self.population, self.targets = {}, {}
        for c in self.CLASSES:
            sub = flux[klass == c]
            order = rng.permutation(sub.size)
            half = sub.size // 2
            self.population[c] = sub[order[:half]]
            self.targets[c] = sub[order[half:]]
        # A small calibration catalogue, as in the gamma-ray study: priors fitted to only
        # ``pop_size`` known members of each class, one fixed sample per class drawn from the
        # prior half; the targets are unchanged.
        if pop_size is not None:
            rng_pop = np.random.default_rng(pop_seed)
            self.population = {c: rng_pop.choice(v, min(int(pop_size), v.size), replace=False)
                               for c, v in self.population.items()}

    # -- the generic environment protocol ----------------------------------

    n_contexts = property(lambda self: self.n_sources)
    action_values = property(lambda self: self.times)
    target_exposure = property(lambda self: self.eval_time * self.counts_per_ks)

    def actions(self):
        return [(k, float(t)) for k in range(self.n_sources) for t in self.times]

    def exposure(self, u, recovery=None):
        """``f(u) = C t``: a pointing is a pointing, with no per-target factor."""
        return float(u[1]) * self.counts_per_ks

    def cost(self, u):
        return self.overhead + float(u[1])

    # -- what every agent is told ------------------------------------------

    def context_classes(self):
        return list(self._classes)

    def initial_beliefs(self, model):
        cache, out = {}, []
        for c in self._classes:
            key = (model.name, c)
            if key not in cache:
                cache[key] = model.fit_population(self.population[c])
            out.append(cache[key])
        return out

    # -- ground truth ------------------------------------------------------

    def blocks(self, rng):
        truths = np.array([self.targets[c][rng.integers(self.targets[c].size)]
                           for c in self._classes])
        moments = [(float(self.population[c].mean()), float(self.population[c].var()))
                   for c in self._classes]
        return moments, truths, np.ones(self.n_sources)

    def stone_fields(self, truths, recovery, rng):
        """One realised photon stream per target, in exposure; shared by every agent."""
        extent = self.budget * self.counts_per_ks * 1.05 + 10.0
        return [np.sort(rng.uniform(0.0, extent, size=int(rng.poisson(t * extent))))
                for t in truths]

    def held_out(self, truths, recovery, rng, n=24):
        f = self.eval_time * self.counts_per_ks
        return [rng.poisson(f * truths[k], size=n) for k in range(self.n_sources)]
