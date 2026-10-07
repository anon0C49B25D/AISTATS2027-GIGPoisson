"""Which mixing law does a real field of photon rates need?

The bulk-sampling study of the previous section generates its rates from the GIG-Poisson law
itself, so the model is correctly specified by construction and the comparison there measures
what misspecification costs rather than what the family is worth. This study removes that
circularity: the rates are 5065 gamma-ray source fluxes measured by a photon-counting
instrument, and no model in the comparison generated them.

Each candidate mixing law is fitted by maximum likelihood to the same fluxes and scored by
AIC, so the third parameter has to earn its place against the penalty.

Writes to ``results/``. Figures come from ``visualize.py``.

Run: python experiments/gamma-ray/fit.py
"""

import csv
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch import fetch                                               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

#: Fluxes are of order 1e-10, which is numerically awkward for a fitter and irrelevant to
#: the comparison, so everything is expressed in units of 1e-10 photons cm^-2 s^-1.
SCALE = 1e10


def fits(flux):
    """Maximum-likelihood fit of each mixing law, with the location held at zero.

    A rate is positive, so no model is given a free location parameter to spend on shifting
    the support. Every family therefore has exactly the parameters its name implies.
    """
    out = []

    shape, _, sc = stats.gamma.fit(flux, floc=0.0)
    out.append(("gamma", 2, float(stats.gamma.logpdf(flux, shape, 0.0, sc).sum()),
                "shape={:.4g}, scale={:.4g}".format(shape, sc)))

    s, _, sc = stats.lognorm.fit(flux, floc=0.0)
    out.append(("lognormal", 2, float(stats.lognorm.logpdf(flux, s, 0.0, sc).sum()),
                "sigma={:.4g}, scale={:.4g}".format(s, sc)))

    p, b, _, sc = stats.geninvgauss.fit(flux, floc=0.0)
    out.append(("generalised inverse Gaussian", 3,
                float(stats.geninvgauss.logpdf(flux, p, b, 0.0, sc).sum()),
                "p={:.4g}, omega={:.4g}, scale={:.4g}".format(p, b, sc)))
    return out


def main():
    flux = fetch() * SCALE
    print("Fermi-LAT 4FGL: {} source fluxes, in 1e-10 photons cm^-2 s^-1".format(flux.size))
    print("  mean {:.3f}   variance-to-mean {:.1f}   skewness {:.1f}".format(
        flux.mean(), flux.var() / flux.mean(),
        float(((flux - flux.mean()) ** 3).mean() / flux.std() ** 3)))

    rows = fits(flux)
    best = min(2 * k - 2 * ll for _, k, ll, _ in rows)
    print("\n  {:<30} {:>2} {:>12} {:>11} {:>9}".format(
        "mixing law", "k", "log-lik", "AIC", "dAIC"))
    table = []
    for name, k, ll, params in rows:
        aic = 2 * k - 2 * ll
        print("  {:<30} {:>2} {:>12.1f} {:>11.1f} {:>9.1f}".format(
            name, k, ll, aic, aic - best))
        print("       {}".format(params))
        table.append((name, k, ll, aic, aic - best, params))

    with open(os.path.join(HERE, "results", "mixing_law_fits.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mixing_law", "n_params", "loglik", "aic", "delta_aic", "parameters"])
        w.writerows(table)
    print("\n  wrote results/mixing_law_fits.csv")

    # A single number for the paper: how far the fitted GIG sits from the gamma boundary.
    p, omega, _, sc = stats.geninvgauss.fit(flux, floc=0.0)
    print("\n  fitted concentration omega = {:.4f}".format(omega))
    print("  the gamma is the omega -> 0 edge of this family, and the negative order")
    print("  p = {:.3f} is outside the gamma's range entirely".format(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
