# How accurate the deterministic EIG is

Written 2026-10-06, in answer to a review comment that "exact EIG" oversells a computation
that truncates a sum and applies a fixed quadrature. Everything below is produced by `run.py`
and is deterministic.

## What is computed

The EIG of the GIG-Poisson model (`GIGPoisson.eig` in `methods/countmodels.py`) is the
entropy of the Sichel predictive minus the expected Poisson entropy under the belief.

| term | numerics | settings |
|---|---|---|
| predictive entropy | sum over the support, in logs | widened until the mass outside is below `1e-9`, capped at `2e6` counts |
| expected Poisson entropy | Gauss-Legendre in `log lam` | 64 nodes, window where the log density is within 75 nats of its maximum |

The value is deterministic but not exact, and neither error is bounded analytically.

## The reference

The predictive is summed in chunks of `1e6` until the mass outside is below `1e-12`, or to
`6e7` counts, and the second term is integrated by SciPy's adaptive `quad` over a 200-nat
window. The reference shares only the Bessel routine with the shipped computation, which the
verification study checks separately (`synthetic/verification/`, relative error `1.3e-13`
against 40-digit references).

The beliefs are those where the support is longest: the fitted class priors of the three
settings, from the calibration catalogues of `comparison/run.py`, and the belief after one
observation at the shortest action, at each of the six actions. These are the states that
`comparison/timing.py` times. 180 evaluations in all.

## What it found

| quantity | value |
|---|---|
| largest error | `6.6e-7` nats (relative `2.9e-7`) |
| largest error of the quadrature alone | `3.6e-13` nats |
| evaluations where the cap binds | 4 of 180, all at a fitted prior and one of the two longest actions |
| mass outside where the cap binds | up to `2.1e-8` |
| error where the cap binds | up to `6.6e-7` nats |
| mass outside elsewhere | up to `9.9e-10` |
| error elsewhere | up to `2.7e-8` nats |
| longest support after one observation | 19,724 counts |

The capped evaluations are the gamma-ray class of unassociated sources at 25 Ms, the hard
X-ray binaries at 25 ks, and the hard X-ray class of unidentified and rarer sources at 10 and
25 ks. The first is the value of 2.33 nats quoted in Section 6.2 of the paper, which is
within `7e-7` nats of the reference.

The error grows in proportion to the mass left outside, at about 30 nats per unit mass
(`figures/eig_accuracy.pdf`): the omitted tail carries an entropy of about `-log p` per unit
mass, and `p` is near `e^-30` where the sum stops.

## Running

```bash
python experiments/eig-accuracy/run.py         # ~3 min
python experiments/eig-accuracy/visualize.py
```

`data/priors.csv` holds the fitted class priors, `results/eig_accuracy.csv` one row per
evaluation, with the error split into its support and quadrature parts.
