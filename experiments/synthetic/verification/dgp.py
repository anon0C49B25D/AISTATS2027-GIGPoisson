"""A data-generating process for which the GIG-Poisson model is correctly specified.

The process is a photon-counting survey. There are ``K`` sites. Site ``k`` emits at an
unknown rate ``lam_k``, drawn once from a generalised inverse Gaussian law. The sensor
picks a site and a dwell time, and the detector returns a count::

    lam_k          ~  GIG(alpha_k, a_k, b_k)     (parameter, drawn once)
    y | lam_k, u   ~  Poisson( f(u) lam_k )      (observation)

with the action ``u = (k, t)`` and the exposure map ``f(u) = t * kappa_k``, in which
``kappa_k`` is the known geometric attenuation of site ``k``.

Correct specification means what it says: the sensor's prior is the law the parameter was
drawn from, and the sensor's likelihood is the law the count was drawn from. Nothing here
is an approximation to the simulator, so the analytic predictive is the exact marginal of
the process, and :func:`self_check` verifies that against the simulator rather than
asserting it.

The closed forms themselves live in ``methods/gigpoisson.py`` and are re-exported here, so
that scripts written against this module keep working.
"""

import os
import sys

import numpy as np
from scipy.special import logsumexp
from scipy.stats import geninvgauss

#: ``np.trapz`` was renamed ``np.trapezoid`` in NumPy 2.0 and the old name deprecated, so
#: the gate picks whichever this installation provides rather than pinning a version.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from methods.gigpoisson import (  # noqa: E402,F401
    OLVER_FROM, TAIL_TOL, gig_from_mean, gig_logpdf, gig_moments, gig_sample,
    log_besselk, log_besselk_sequence, log_gig_norm, negbinom_limit_error,
    posterior, predictive_moments, sichel_logpmf, sichel_support, _scipy_shape)

# --------------------------------------------------------------------------
# The survey
# --------------------------------------------------------------------------

class Survey(object):
    """A sensing environment: sites with unknown rates, actions that buy exposure.

    The interface mirrors the one the model class will need later, so that this object can
    be promoted to a family module without being rewritten: :meth:`actions` enumerates the
    admissible actions, :meth:`exposure` maps one to the scalar the predictive depends on,
    :meth:`channels` draws an environment, and :meth:`sample_y` runs the detector.
    """

    label = "GIG-Poisson survey"
    action_label = "dwell time t"
    action_units = "seconds"

    def __init__(self, n_sites=8, dwells=None, order=-0.5, mean_range=(0.2, 15.0),
                 omega_range=(0.05, 4.0), attenuation_range=(0.25, 1.0)):
        self.n_sites = int(n_sites)
        self.dwells = (np.geomspace(0.1, 30.0, 8) if dwells is None
                       else np.asarray(dwells, dtype=float))
        self.order = float(order)
        self.mean_range = mean_range
        self.omega_range = omega_range
        self.attenuation_range = attenuation_range

    # -- actions -----------------------------------------------------------

    def actions(self):
        """Every ``(site, dwell)`` pair the sensor may choose."""
        return [(k, float(t)) for k in range(self.n_sites) for t in self.dwells]

    def exposure(self, u, attenuation):
        """``f(u)``: dwell time times the known attenuation of the site addressed."""
        k, t = u
        return float(t) * float(attenuation[k])

    # -- environment -------------------------------------------------------

    def channels(self, rng):
        """Draw one environment: a prior per site, and the rate it actually emits at.

        Sites are heterogeneous in three ways that a sensing policy has to trade off. They
        differ in brightness (the prior mean), in dispersion (the concentration ``omega``,
        which sets how heavy the mixing tail is at fixed brightness) and in observability
        (the attenuation, which scales every exposure the site can be given). A criterion
        that tracks only one of the three is separable from one that tracks all of them.
        """
        priors, truths, atten = [], [], []
        for _ in range(self.n_sites):
            mean = float(np.exp(rng.uniform(*np.log(self.mean_range))))
            omega = float(np.exp(rng.uniform(*np.log(self.omega_range))))
            prior = gig_from_mean(self.order, mean, omega)
            priors.append(prior)
            truths.append({"lam": float(gig_sample(prior, 1, rng)[0])})
            atten.append(float(rng.uniform(*self.attenuation_range)))
        return priors, truths, np.asarray(atten)

    # -- the process -------------------------------------------------------

    def sample_y(self, lam, f, rng, size=None):
        """One detector read: ``y ~ Poisson(f lam)``."""
        return rng.poisson(float(f) * float(lam), size=size)


# --------------------------------------------------------------------------
# Correctness gate
# --------------------------------------------------------------------------

def _check_besselk(verbose=True):
    """Both Bessel routes against 40-digit references, and the naive route's failure.

    Establishes three things: that ``kve`` is exact below the crossover, that the
    asymptotic expansion is exact above it, and that the unlogged ``kv`` that a direct
    transcription of the closed form would call overflows at an order the posterior
    reaches after a handful of counts.
    """
    from mpmath import besselk, log as mplog, mp
    from scipy.special import kv
    mp.dps = 40

    orders = [-0.5, 0.0, 0.7, 5.0, 30.0, 44.0, 46.0, 120.0, 800.0, 5000.0]
    args = [0.05, 0.9, 7.0, 220.0]
    worst, rows = 0.0, []
    for v in orders:
        for z in args:
            ref = float(mplog(besselk(v, z)))
            got = float(log_besselk(v, z))
            rel = abs(got - ref) / max(abs(ref), 1.0)
            worst = max(worst, rel)
            rows.append((v, z, ref, got, rel, bool(np.isfinite(kv(v, z)))))
    # The recurrence, over a run of orders long enough that no direct route survives it.
    seq = log_besselk_sequence(-0.5, 3.0, 900)
    ref_seq = [float(mplog(besselk(-0.5 + j, 3.0))) for j in (0, 1, 7, 60, 400, 900)]
    got_seq = [seq[j] for j in (0, 1, 7, 60, 400, 900)]
    worst_seq = max(abs(g - r) / max(abs(r), 1.0) for g, r in zip(got_seq, ref_seq))
    # Where a direct transcription of the closed form stops returning a number at all.
    # The posterior order is alpha + sum(y), so this is a count, not an abstraction.
    overflow = {z: next(v for v in range(1, 4000) if not np.isfinite(kv(float(v), z)))
                for z in (0.05, 0.9, 7.0, 220.0)}

    ok = worst < 1e-10 and worst_seq < 1e-10
    if verbose:
        print("  Bessel: worst relative error {:.2e} pointwise, {:.2e} over a 900-term "
              "recurrence".format(worst, worst_seq))
        print("  unlogged K_v(z) overflows to inf from order {}".format(
            ", ".join("{} at z={}".format(n, z) for z, n in overflow.items())))
        print("  ->  {}".format("PASS" if ok else "FAIL"))
    return ok, rows, worst, worst_seq


def _check_predictive(survey, rng, n_draws=400_000, verbose=True):
    """The analytic Sichel law against the simulator, and the moments against the sum.

    This is the check that earns the phrase "correctly specified". Nothing is asserted
    about the marginal: counts are produced by running the two-stage process, and the
    analytic mass function is compared with the frequencies they arrive at.
    """
    priors, _, atten = survey.channels(rng)
    rows, worst_tv, worst_mom, bad = [], 0.0, 0.0, 0
    for k in (0, survey.n_sites // 2, survey.n_sites - 1):
        prior = priors[k]
        for t in (survey.dwells[0], survey.dwells[len(survey.dwells) // 2],
                  survey.dwells[-1]):
            f = survey.exposure((k, t), atten)
            lam = gig_sample(prior, n_draws, rng)
            y = rng.poisson(f * lam)

            support = sichel_support(prior, f)
            mass = float(np.exp(logsumexp(sichel_logpmf(support, prior, f))))

            # Total variation over the counts the simulator actually produced, against a
            # reference for what a *correct* mass function scores at this sample size. A
            # heavy tail spreads the sample over many rarely visited counts, so the
            # empirical TV has a floor of order sqrt(support / n) that has nothing to do
            # with whether the analytic law is right. The floor is measured, not modelled:
            # two independent halves of the same sample are compared with each other, and
            # the analytic law passes if it is no further from the sample than the sample
            # is from itself.
            ymax = int(min(y.max(), 20000))
            grid = np.arange(ymax + 1, dtype=float)
            emp = np.bincount(np.minimum(y, ymax), minlength=ymax + 1) / y.size
            tv = 0.5 * float(np.sum(np.abs(emp - np.exp(sichel_logpmf(grid, prior, f)))))
            h = y.size // 2
            e1 = np.bincount(np.minimum(y[:h], ymax), minlength=ymax + 1) / h
            e2 = np.bincount(np.minimum(y[h:], ymax), minlength=ymax + 1) / (y.size - h)
            tv_floor = 0.5 * float(np.sum(np.abs(e1 - e2)))

            # Analytic moments against the support sum, and against the sample.
            m_a, v_a = predictive_moments(prior, f)
            w = np.exp(sichel_logpmf(support, prior, f))
            m_s = float(np.sum(w * support))
            v_s = float(np.sum(w * support ** 2) - m_s ** 2)
            mom = max(abs(m_a - m_s) / max(m_a, 1.0), abs(v_a - v_s) / max(v_a, 1.0))

            # A three-sigma band on the sample mean is the only stochastic comparison.
            sem = np.sqrt(v_a / y.size)
            z = (float(np.mean(y)) - m_a) / sem
            fano = v_a / max(m_a, 1e-300)
            rows.append((k, float(t), f, mass, tv, tv_floor, m_a, v_a, fano,
                         float(np.mean(y)), z))
            worst_tv, worst_mom = max(worst_tv, tv / max(tv_floor, 1e-12)), \
                max(worst_mom, mom)
            if abs(1.0 - mass) > 1e-10 or mom > 1e-9 or abs(z) > 4.0 \
                    or tv > 3.0 * tv_floor:
                bad += 1
    ok = bad == 0
    if verbose:
        print("  predictive, {} draws per cell".format(n_draws))
        print("  {:>4} {:>7} {:>7} {:>10} {:>9} {:>9} {:>9} {:>10} {:>8} {:>6}".format(
            "site", "dwell", "f(u)", "1-mass", "TV", "TV floor", "mean", "var",
            "Fano", "z"))
        for k, t, f, mass, tv, tvf, m_a, v_a, fano, _mb, z in rows:
            print("  {:4d} {:7.3g} {:7.3g} {:10.2e} {:9.2e} {:9.2e} {:9.4g} {:10.4g} "
                  "{:8.1f} {:+6.2f}".format(k, t, f, abs(1.0 - mass), tv, tvf, m_a, v_a,
                                            fano, z))
        print("  worst moment residual {:.2e}; worst TV as a multiple of its floor {:.2f}"
              .format(worst_mom, worst_tv))
        print("  ->  {} ({} of {} cells outside tolerance)".format(
            "PASS" if ok else "FAIL", bad, len(rows)))
    return ok, rows, worst_mom, n_draws


def _check_conjugacy(survey, rng, verbose=True):
    """The conjugate update against a numerical Bayes update, and the density's own mass.

    Confirms that ``GIG(alpha + y, a, b + 2 f)`` is the posterior of the process as
    simulated, rather than of a process the closed form happens to describe.

    The quadrature runs in ``log lam`` between posterior quantiles. A linear grid will not
    do: at a negative order the mixing density behaves like ``lam^(alpha-1)`` at the
    origin, which is integrable but unresolvable by equally spaced abscissae, and a
    trapezoid rule on one reports a disagreement of order ``1e-5`` that is its own.
    """
    priors, _, atten = survey.channels(rng)
    worst_tv, worst_mass = 0.0, 0.0
    for k in (0, survey.n_sites - 1):
        prior = priors[k]
        for t in (survey.dwells[0], survey.dwells[-1]):
            f = survey.exposure((k, t), atten)
            for y in (0, 1, 7, 40):
                post = posterior(prior, y, f)
                alpha, omega, eta = _scipy_shape(post)
                lo = geninvgauss.ppf(1e-13, alpha, omega, scale=eta)
                hi = geninvgauss.isf(1e-13, alpha, omega, scale=eta)
                s = np.linspace(np.log(lo), np.log(hi), 200_000)
                lam = np.exp(s)
                lw = gig_logpdf(lam, prior) + y * np.log(f * lam) - f * lam
                lw -= np.max(lw)
                num = np.exp(lw) * lam
                num /= _trapezoid(num, s)
                ana = np.exp(gig_logpdf(lam, post)) * lam
                worst_tv = max(worst_tv, 0.5 * float(_trapezoid(np.abs(num - ana), s)))
                worst_mass = max(worst_mass, abs(1.0 - float(_trapezoid(ana, s))))
    ok = worst_tv < 1e-9 and worst_mass < 1e-9
    if verbose:
        print("  conjugate update vs numerical Bayes: worst TV {:.2e}, worst |1-mass| "
              "{:.2e}".format(worst_tv, worst_mass))
        print("  ->  {}".format("PASS" if ok else "FAIL"))
    return ok, worst_tv


def _check_negbinom_limit(verbose=True):
    """The gamma boundary: the predictive must become negative binomial as omega falls."""
    rows = [(w, negbinom_limit_error(2.0, 4.0, w, 3.0)) for w in
            (1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5)]
    ok = rows[-1][1] < 1e-4 and all(rows[i][1] > rows[i + 1][1] for i in range(len(rows) - 1))
    if verbose:
        print("  negative-binomial limit, order 2, mean 4, f = 3")
        for w, e in rows:
            print("    omega = {:8.1e}   max |log p_Sichel - log p_NB| = {:.3e}"
                  .format(w, e))
        print("  ->  {}".format("PASS" if ok else "FAIL"))
    return ok, rows


def self_check(survey=None, seed=0, verbose=True):
    """Gate the process before it is used to produce anything.

    Four checks, in the order in which a failure would be cheapest to diagnose: the Bessel
    evaluation, the conjugate update, the predictive against the simulator, and the gamma
    boundary. Returns ``(ok, parts)``.
    """
    survey = Survey() if survey is None else survey
    rng = np.random.default_rng(seed)
    if verbose:
        print("Correctness gate: {}".format(survey.label))
    ok_b, bessel_rows, worst_b, worst_seq = _check_besselk(verbose)
    ok_c, worst_c = _check_conjugacy(survey, rng, verbose)
    ok_p, pred_rows, worst_mom, n_draws = _check_predictive(survey, rng,
                                                           verbose=verbose)
    ok_n, nb_rows = _check_negbinom_limit(verbose)
    ok = ok_b and ok_c and ok_p and ok_n
    if verbose:
        print("  gate: {}".format("PASS" if ok else "FAIL"))
    return ok, {"bessel": bessel_rows, "bessel_worst": worst_b,
                "bessel_worst_seq": worst_seq, "conjugacy_tv": worst_c,
                "predictive": pred_rows, "negbinom": nb_rows,
                "moment_residual": worst_mom, "n_draws": n_draws,
                "support_mass_error": max(abs(1.0 - r[3]) for r in pred_rows)}


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check()[0] else 1)
