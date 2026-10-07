# Where does the third parameter pay in the tail?

An illustration of the Sichel tail lemma (Section 2 of the paper, proof in the
appendix). Not a benchmark: the Sichel arm is given the true hyperparameters.

## What the lemma predicts

At matched predictive mean and variance, the Sichel tail decays geometrically at
`q = 2f/(b + 2f)` and the negative binomial at `q_Gamma = f/(beta + f)`, with `q > q_Gamma`.
After both are updated on the same data, the gap `log(q_F / q_Gamma,F)` depends on the
exposure `F` alone, not the counts, and shrinks as `O(1/F)`.

## Design

One context. The truth is a GIG at the moments of Table 1 (predictive mean 6, variance 48
at unit exposure), at 11 orders from `-3` to `0.9` times the ceiling `r = mean^2/var = 0.857`.
The negative binomial arm is the moment-matched gamma-Poisson, which is the same at every
order. After an accumulated count `T` at exposure `F`, both are updated and the next count
at unit exposure is scored by the **censored likelihood score** (Diks, Panchenko & van Dijk
2011) on `y >= t`, at `t` = the 95th, 99th and 99.9th percentile of the matched NB prior
predictive (20, 32, 48).

Because the Sichel arm is the exact Bayesian predictive, the expected score gap given `T` is
exactly the KL divergence between the two censored forecasts, summed over the support. The
only Monte Carlo is over `T` (1500 draws, none at `F = 0`). The first version scored against
a drawn rate instead. That has the same mean, but its conditional variance swamped a
millinat-sized signal (|t| < 2 almost everywhere).

The gain is reported per forecast and **per exceedance** (divided by `P(Y >= t)`).

## Result (`figures/tail_scores.pdf`)

Per-exceedance gain at the 99th percentile, in nats:

| order / ceiling | F = 0 | F = 1 | F = 4 | F = 16 | F = 64 |
|---|---|---|---|---|---|
| −3 | 0.554 | 0.044 | 0.0045 | 3.4e-4 | 2.2e-5 |
| −1 | 0.126 | 0.018 | 0.002 | 1.6e-4 | 1.1e-5 |
| 0 | 0.038 | 0.008 | 0.001 | 7.3e-5 | 4.9e-6 |
| 0.9 | 6.7e-4 | 1.3e-4 | 1.5e-5 | 1.1e-6 | 7.5e-8 |

- **Order.** The gain is monotone in the distance below the ceiling, and it vanishes at the
  ceiling, where the GIG becomes the gamma.
- **Threshold.** At `F = 0` and order −3 it is 0.33, 0.55 and 1.54 nats per exceedance at the
  95th, 99th and 99.9th percentiles. The further out the decision, the more the third
  parameter is worth.
- **Exposure.** The gain falls by about 10× from `F = 0` to `F = 1`, then as about `1/F^2`.
  This is the square of the lemma's `O(1/F)` rate gap, as expected for a divergence, which
  is quadratic in the parameter gap. The contours of the rate gap in panel (a) follow the
  iso-gain bands.

Per forecast (undivided), the gain at `F = 0` is a few millinats. A centre-weighted score
will not register an advantage of this size.

## Running

```bash
python experiments/synthetic/tail-scores/run.py        # ~30 s, writes results/summary.csv
python experiments/synthetic/tail-scores/visualize.py  # writes figures/tail_scores.pdf
```

Reproducible from seed 0.
