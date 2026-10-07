"""Prediction error against sample size, over regimes of the generating parameters.

The model is correctly specified throughout (``dgp.py`` gates that separately), so what
this measures is not bias but the rate at which the price of not knowing the rate is paid
off. Each replicate draws its own true rate from the mixing law, observes counts at unit
exposure, and is scored on how far its posterior predictive sits from the predictive an
oracle would use who knew the rate exactly.

Two error measures, both exact rather than estimated from a held-out sample:

    KL divergence   ``KL( P(y | f lam*) || p(y | u, D_n) )``, summed over the support of
                    the oracle predictive. This is the prediction error proper: it is zero
                    only when the two laws agree, and it carries no Monte-Carlo noise.

    rate error      ``| E[lam | D_n] - lam* |``, which is what Proposition 2 bounds, and is
                    reported alongside so the two rates can be compared.

Sample size is reported as the accumulated exposure ``F_n = n f`` as well as the count ``n``,
because the theory is a statement about the former. Here ``f = 1`` so they coincide, which
is the point of choosing unit exposure.

Writes to ``results/``. Figures come from ``visualize_prediction.py``.

Run: python experiments/synthetic/verification/run_prediction.py
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dgp import (gig_from_mean, gig_moments, gig_sample,          # noqa: E402
                 posterior, predictive_moments, sichel_logpmf)

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0
N_ENV = 400                                  # replicate rates per regime
SAMPLE_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
F_TRAIN = 1.0                                # unit exposure, so F_n = n
F_EVAL = 1.0

#: Regimes of the generating parameters. The first three vary the tail of the mixing law
#: at a fixed mean, the fourth drops the mean into the regime where most counts are zero.
REGIMES = [
    ("light",  2.0,  4.0, 4.0),
    ("moderate", -0.5, 1.0, 4.0),
    ("heavy", -1.5,  0.1, 4.0),
    ("sparse", -0.5, 1.0, 0.5),
]


def save_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("  wrote {} ({} rows)".format(os.path.relpath(path, HERE), len(rows)))


def poisson_support(mu, tol_pad=12.0):
    """Counts carrying essentially all the mass of ``Poisson(mu)``."""
    hi = int(mu + tol_pad * np.sqrt(max(mu, 1e-12)) + 60.0)
    return np.arange(hi + 1, dtype=float)


def kl_oracle_to_model(lam_true, belief, f):
    """``KL( Poisson(f lam*) || Sichel(. | belief, f) )`` in nats, by a support sum."""
    from scipy.special import gammaln
    mu = f * lam_true
    y = poisson_support(mu)
    log_p = -mu + y * np.log(mu) - gammaln(y + 1.0)
    p = np.exp(log_p)
    log_q = sichel_logpmf(y, belief, f)
    return float(np.sum(p * (log_p - log_q)))


def kl_oracle_to_negbinom(lam_true, shape, rate, f):
    """The same divergence for a misspecified gamma-Poisson agent.

    Its rate carries ``Gamma(shape, rate)`` rather than the law the rates were drawn from,
    so it is wrong about the shape of the prior and right about nothing else. The update is
    still conjugate and the predictive is still negative binomial, which is the point: this
    is the model a practitioner reaches for by default, run on the same counts.
    """
    from scipy.stats import nbinom
    mu = f * lam_true
    y = poisson_support(mu)
    from scipy.special import gammaln
    log_p = -mu + y * np.log(mu) - gammaln(y + 1.0)
    p = np.exp(log_p)
    log_q = nbinom.logpmf(y, shape, rate / (rate + f))
    return float(np.sum(p * (log_p - log_q)))


def main():
    rd = os.path.join(HERE, "results")
    rng = np.random.default_rng(SEED)
    rows, summary = [], []

    for name, alpha, omega, mean in REGIMES:
        prior = gig_from_mean(alpha, mean, omega)
        pm, pv = gig_moments(prior)
        print("\nregime {}: alpha={:+.1f} omega={:.2g} prior mean={:.3g} "
              "(V/E={:.1f})".format(name, alpha, omega, pm, pv / pm))

        # One true rate per replicate, drawn from the law the model assumes.
        lam_true = gig_sample(prior, N_ENV, rng)
        # One training stream per replicate, long enough for the largest sample size.
        counts = rng.poisson(F_TRAIN * lam_true[:, None],
                             size=(N_ENV, max(SAMPLE_SIZES)))
        cum = np.cumsum(counts, axis=1)

        # The misspecified arm: a gamma mixing law matched to the same two prior moments,
        # so it is told exactly what the correctly specified agent is told and differs only
        # in the family it expresses that in.
        nb_shape0, nb_rate0 = pm * pm / pv, pm / pv

        for n in SAMPLE_SIZES:
            kls, rate_errs, post_vars, nb_kls = [], [], [], []
            for e in range(N_ENV):
                belief = {"alpha": prior["alpha"] + float(cum[e, n - 1]),
                          "a": prior["a"],
                          "b": prior["b"] + 2.0 * n * F_TRAIN}
                kls.append(kl_oracle_to_model(lam_true[e], belief, F_EVAL))
                m, v = gig_moments(belief)
                rate_errs.append(abs(m - lam_true[e]))
                post_vars.append(v)
                nb_kls.append(kl_oracle_to_negbinom(
                    lam_true[e], nb_shape0 + float(cum[e, n - 1]),
                    nb_rate0 + n * F_TRAIN, F_EVAL))
            kls = np.asarray(kls)
            rate_errs = np.asarray(rate_errs)
            post_vars = np.asarray(post_vars)
            nb_kls = np.asarray(nb_kls)
            rows.append((name, alpha, omega, mean, n, n * F_TRAIN,
                         float(np.mean(kls)),
                         float(np.std(kls, ddof=1) / np.sqrt(N_ENV)),
                         float(np.median(kls)),
                         float(np.mean(rate_errs)),
                         float(np.std(rate_errs, ddof=1) / np.sqrt(N_ENV)),
                         float(np.mean(post_vars)),
                         float(np.mean(nb_kls)),
                         float(np.std(nb_kls, ddof=1) / np.sqrt(N_ENV))))
            print("    n={:4d}  KL={:.4e} +/- {:.1e}   NB KL={:.4e}   "
                  "|E[lam]-lam*|={:.4e}".format(n, rows[-1][6], rows[-1][7],
                                                rows[-1][12], rows[-1][9]))

        # Log-log slope over the upper half of the range, where any transient has passed.
        sel = [r for r in rows if r[0] == name and r[4] >= SAMPLE_SIZES[len(SAMPLE_SIZES) // 2]]
        ln = np.log([r[4] for r in sel])
        for label, col in (("kl", 6), ("rate_error", 9), ("post_var", 11),
                           ("kl_negbinom", 12)):
            slope = float(np.polyfit(ln, np.log([r[col] for r in sel]), 1)[0])
            summary.append((name, alpha, omega, mean, label, slope))
            print("    log-log slope, {:<10s} {:+.3f}".format(label, slope))

    save_csv(os.path.join(rd, "prediction_error.csv"),
             ["regime", "alpha", "omega", "prior_mean", "n", "exposure", "kl_mean",
              "kl_sem", "kl_median", "rate_error_mean", "rate_error_sem",
              "posterior_var_mean", "kl_negbinom_mean", "kl_negbinom_sem"], rows)
    save_csv(os.path.join(rd, "prediction_slopes.csv"),
             ["regime", "alpha", "omega", "prior_mean", "quantity", "loglog_slope"],
             summary)

    print("\nExpected slopes if the theory holds: KL -1, rate error -1/2, "
          "posterior variance -1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
