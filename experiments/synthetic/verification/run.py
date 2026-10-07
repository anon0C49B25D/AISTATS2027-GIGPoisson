"""Generate and verify the GIG-Poisson data-generating process.

Runs the correctness gate first and aborts if it fails, then simulates one environment
over the whole action grid and writes what later experiments and the paper will read.

Writes to ``data/`` and ``results/``. Figures are produced separately by ``visualize.py``.

Run: python experiments/synthetic/verification/run.py
"""

import csv
import os
import sys
import time

import numpy as np
from scipy.special import kv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dgp import (Survey, gig_moments, gig_sample, log_besselk,      # noqa: E402
                 posterior, predictive_moments, self_check, sichel_logpmf,
                 sichel_support)

SEED = 0
N_REPS = 512          # detector reads per (site, dwell) at the realised rate
N_MARGINAL = 200_000  # draws from the two-stage process, for the marginal columns
N_ROUNDS = 40         # rounds of the order-growth demonstration
HERE = os.path.dirname(os.path.abspath(__file__))


def save_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("  wrote {} ({} rows)".format(os.path.relpath(path, HERE), len(rows)))


def main():
    dd, rd = os.path.join(HERE, "data"), os.path.join(HERE, "results")
    survey = Survey()

    ok, parts = self_check(survey, seed=SEED)
    save_csv(os.path.join(rd, "gate_besselk.csv"),
             ["order", "argument", "log_k_reference", "log_k_ours", "rel_error",
              "unlogged_kv_finite"], parts["bessel"])
    save_csv(os.path.join(rd, "gate_predictive.csv"),
             ["site", "dwell", "exposure", "support_mass", "tv_empirical", "tv_floor",
              "mean_analytic", "var_analytic", "fano", "mean_empirical", "z"],
             parts["predictive"])
    save_csv(os.path.join(rd, "gate_negbinom_limit.csv"),
             ["omega", "max_abs_log_ratio"], parts["negbinom"])
    # The scalar figures the manuscript quotes in its verification section. Every
    # other gate number lands in a CSV, and these used to be printed only, so they
    # could not be checked without a rerun.
    save_csv(os.path.join(rd, "gate_summary.csv"),
             ["quantity", "value"],
             [("draws_per_two_stage_check", parts["n_draws"]),
              ("support_mass_error", parts["support_mass_error"]),
              ("predictive_moment_residual", parts["moment_residual"]),
              ("conjugate_update_total_variation", parts["conjugacy_tv"]),
              ("log_besselk_relative_error", parts["bessel_worst"]),
              ("log_besselk_recurrence_relative_error", parts["bessel_worst_seq"])])
    if not ok:
        print("\nABORT: the process failed its correctness gate; anything downstream "
              "would be noise.")
        return 1

    # -- the environment ---------------------------------------------------
    rng = np.random.default_rng(SEED)
    priors, truths, atten = survey.channels(rng)

    env_rows = []
    for k, (prior, truth) in enumerate(zip(priors, truths)):
        m, v = gig_moments(prior)
        env_rows.append((k, prior["alpha"], prior["a"], prior["b"],
                         float(np.sqrt(prior["a"] * prior["b"])),
                         float(np.sqrt(prior["a"] / prior["b"])), m, v, v / m,
                         float(atten[k]), truth["lam"]))
    save_csv(os.path.join(dd, "environments.csv"),
             ["site", "alpha", "a", "b", "omega", "eta", "prior_mean", "prior_var",
              "prior_var_over_mean", "attenuation", "lam_true"], env_rows)

    # -- observations over the whole action grid ---------------------------
    # Two different things get simulated here and they must not be confused.
    #
    #   the survey trace: the counts a sensor actually collects, drawn at the rate the
    #   site was realised at. Conditional on that rate the counts are Poisson, so their
    #   Fano factor is one however overdispersed the marginal is.
    #
    #   the marginal draws: the two-stage process run end to end, a fresh rate from the
    #   mixing law for every read. This is the law the predictive is a law for, and the
    #   only one the analytic mass function should be compared against.
    print("\nSimulating {} reads at each of {} actions, plus {} marginal draws"
          .format(N_REPS, len(survey.actions()), N_MARGINAL))
    obs_rows, disp_rows = [], []
    for k, t in survey.actions():
        f = survey.exposure((k, t), atten)
        y = survey.sample_y(truths[k]["lam"], f, rng, size=N_REPS)
        for r, yi in enumerate(y):
            obs_rows.append((k, float(t), f, r, int(yi)))
        ym = rng.poisson(f * gig_sample(priors[k], N_MARGINAL, rng))
        m_a, v_a = predictive_moments(priors[k], f)
        disp_rows.append((k, float(t), f, m_a, v_a, v_a / m_a,
                          float(np.mean(ym)), float(np.var(ym, ddof=1)),
                          float(np.var(ym, ddof=1) / max(np.mean(ym), 1e-12)),
                          float(np.var(y, ddof=1) / max(np.mean(y), 1e-12)),
                          int(ym.max())))
    save_csv(os.path.join(dd, "observations.csv"),
             ["site", "dwell", "exposure", "replicate", "y"], obs_rows)
    save_csv(os.path.join(rd, "overdispersion.csv"),
             ["site", "dwell", "exposure", "mean_analytic", "var_analytic",
              "fano_analytic", "mean_marginal", "var_marginal", "fano_marginal",
              "fano_conditional", "y_max_marginal"], disp_rows)

    fanos = [r[5] for r in disp_rows]
    cond = [r[9] for r in disp_rows]
    print("    marginal Fano factor over the action grid: {:.2f} to {:.1f}"
          .format(min(fanos), max(fanos)))
    print("    conditional on the realised rate: {:.2f} to {:.2f}, as Poisson requires"
          .format(min(cond), max(cond)))

    # -- the predictive law against the process, for the figures -----------
    pmf_rows, hist_rows = [], []
    for k in (0, survey.n_sites // 2, survey.n_sites - 1):
        for t in (survey.dwells[0], survey.dwells[len(survey.dwells) // 2],
                  survey.dwells[-1]):
            f = survey.exposure((k, t), atten)
            support = sichel_support(priors[k], f)
            lp = sichel_logpmf(support, priors[k], f)
            keep = lp > np.log(1e-9)
            for yi, lpi in zip(support[keep], lp[keep]):
                pmf_rows.append((k, float(t), f, int(yi), float(lpi)))
            ym = rng.poisson(f * gig_sample(priors[k], N_MARGINAL, rng))
            counts = np.bincount(ym)
            for yi in np.flatnonzero(counts):
                hist_rows.append((k, float(t), f, int(yi), int(counts[yi]), ym.size))
    save_csv(os.path.join(rd, "predictive_pmf.csv"),
             ["site", "dwell", "exposure", "y", "log_pmf"], pmf_rows)
    save_csv(os.path.join(dd, "marginal_histogram.csv"),
             ["site", "dwell", "exposure", "y", "count", "n_draws"], hist_rows)

    # -- why log space is not optional -------------------------------------
    # The posterior order is alpha + sum(y). Following one site through the survey shows
    # how many rounds the unlogged closed form survives, which is the practical content
    # of the numerical hazard.
    print("\nOrder growth under repeated observation at the brightest site")
    k = int(np.argmax([r[6] for r in env_rows]))
    prior, lam = priors[k], truths[k]["lam"]
    f = survey.exposure((k, survey.dwells[-1]), atten)
    order_rows, post = [], dict(prior)
    for rnd in range(1, N_ROUNDS + 1):
        y = int(survey.sample_y(lam, f, rng))
        post = posterior(post, y, f)
        z = float(np.sqrt(post["a"] * post["b"]))
        t0 = time.perf_counter()
        lk = float(log_besselk(post["alpha"], z))
        secs = time.perf_counter() - t0
        # What a direct transcription of the closed form would have returned.
        naive = float(kv(abs(post["alpha"]), z))
        order_rows.append((rnd, y, post["alpha"], z, lk, naive, bool(np.isfinite(naive)),
                           secs))
    save_csv(os.path.join(rd, "order_growth.csv"),
             ["round", "y", "order", "argument", "log_besselk", "unlogged_kv",
              "unlogged_finite", "seconds"], order_rows)
    first_bad = next((r[0] for r in order_rows if not r[6]), None)
    print("    site {}, exposure {:.3g}, mean count {:.1f}".format(
        k, f, predictive_moments(prior, f)[0]))
    print("    order after {} rounds: {:.0f}".format(N_ROUNDS, order_rows[-1][2]))
    if first_bad is None:
        print("    the unlogged Bessel function survived all {} rounds".format(N_ROUNDS))
    else:
        print("    the unlogged Bessel function overflows to inf at round {}"
              .format(first_bad))
    print("    log-space evaluation stays finite throughout, at {:.2g} s per call"
          .format(float(np.median([r[7] for r in order_rows]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
