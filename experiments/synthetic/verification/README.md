# A data-generating process for which the GIG-Poisson model is correctly specified

Written 2026-08-08. Everything below is produced by `run.py` and reproducible from seed 0.

## The process

A photon-counting survey over `K = 8` sites. Site `k` emits at an unknown rate, drawn once
from a generalised inverse Gaussian law. The sensor chooses a site and a dwell time, and
the detector returns a count:

```
lam_k        ~  GIG(alpha, a_k, b_k)     parameter, drawn once per site
y | lam_k, u ~  Poisson( f(u) lam_k )    observation
u = (k, t)                               action: site and dwell time
f(u) = t * kappa_k                       exposure: dwell times known attenuation
```

The exposure map is any known nonnegative function of the action. Dwell time times a
geometric attenuation is one instance; every closed form here is inherited unchanged by
any other, which is the practical content of the classification.

Correct specification is literal: the sensor's prior is the law the parameter was drawn
from, the sensor's likelihood is the law the count was drawn from, and no closed form is an
approximation to the simulator. The gate verifies that rather than asserting it.

## Why this family

Writing the mixing density as `lam^(alpha-1) exp{-(a/lam + b lam)/2}`, one observation sends

```
GIG(alpha, a, b)  ->  GIG(alpha + y, a, b + 2 f(u))
```

so the exposure enters the second scale parameter additively and the family is closed under
the action: a nu-action. The marginal is the Sichel law,

```
log p(y|u) = y log f(u) - lgamma(y+1) + logZ(alpha+y, a, b+2f(u)) - logZ(alpha, a, b)
logZ(alpha,a,b) = log 2 + (alpha/2)(log a - log b) + log K_alpha(sqrt(ab))
```

Not for free overdispersion: a gamma mixing law already gives a negative binomial
predictive, whose mean and variance are both free. What the GIG adds is a third parameter,
so that with the predictive mean and variance both held fixed there is still freedom to
move mass between the centre of the distribution and its tail. Measured at matched mean
and variance, the tail mass beyond `y = 50` differs from the moment-matched negative
binomial by factors from 216 down to 0.7 as `omega` falls, and the mass at zero by up to
1.6. That is the standard argument for the Sichel over the negative binomial in the count
literature.

## Parameterisation

`(alpha, a, b)` is reparameterised as an order `alpha`, a scale `eta = sqrt(a/b)` and a
concentration `omega = sqrt(ab)`. Fixing the mixing mean then determines the scale in
closed form, `eta = mean * K_alpha(omega) / K_{alpha+1}(omega)`, so brightness and dispersion are
set independently and no solver is involved (`gig_from_mean`).

`omega` is the departure from the gamma. It is the knob the later experiments should sweep:
at `omega -> 0` the mixing law becomes a gamma and the predictive becomes the negative
binomial of the conjugate Poisson pair, which is the member the companion paper already
worked. The measured distance to that boundary falls as `omega^2`:

| omega | max abs log-ratio to the negative binomial |
|---|---|
| 1e0 | 7.90e-1 |
| 1e-1 | 1.45e-2 |
| 1e-2 | 1.50e-4 |
| 1e-3 | 1.50e-6 |
| 1e-5 | 1.50e-10 |

Sites are heterogeneous in three ways a policy has to trade off: brightness (prior mean),
dispersion (`omega`) and observability (attenuation, which scales every exposure the site
can be given).

## What the gate checks, and what it found

`self_check()` runs four checks and aborts the study on any failure.

1. **Bessel evaluation** against 40-digit `mpmath` references: worst relative error
   `1.3e-13` pointwise over orders `-0.5` to `5000` and arguments `0.05` to `220`, and
   `1.4e-15` over a 900-term order recurrence.
2. **Conjugate update** against numerical Bayes on a log-spaced grid between posterior
   quantiles: worst total variation `7.6e-13`, worst `|1 - mass|` `1.5e-12`.
3. **Predictive against the simulator**, 400k draws of the two-stage process per cell:
   support mass to `4e-13`, analytic moments against the support sum to `9e-11`, sample
   mean inside 2 sigma everywhere, and empirical total variation at `0.61` times its own
   split-half floor.
4. **The gamma boundary**, the table above.

## Overdispersion

Over the 64-action grid the marginal Fano factor runs from **1.01 to 3202**. A Poisson
sensor models every one of them as 1. The Fano factor is `1 + f(u) Var[lam]/E[lam]`, so
*how overdispersed the data look is itself a function of the action*, which is why the
action matters here at all.

One distinction the figures keep separate. Conditional on the rate a site was realised at,
the counts are Poisson and their Fano factor is 1 (measured: 0.85 to 1.14). The
overdispersion lives in the marginal, and it is parameter uncertainty. That is the right
reading for an active sensing problem, where the predictive is what the acquisition scores,
but it means the process is not overdispersed in the sense of a random effect that persists
after the parameter is known. See *Open questions* below.

## Numerics

`K_v(z)` overflows double precision at moderate order: the unlogged function returns `inf`
from order **97** at `z = 0.05` and **147** at `z = 0.9`. The posterior order is
`alpha + sum(y)`, so this is not a remote corner. Following the brightest site at full dwell,
where the predictive mean is 224 counts, a direct transcription of the closed form
**overflows at round 4** and the order reaches 4946 after 40 rounds.

Two arrangements keep it finite. `log_besselk` uses the exponentially scaled `kve` below
order 45 and Olver's uniform asymptotic expansion above it. `log_besselk_sequence` gets the
run of orders the support sum needs from the recurrence `K_{v+1} = K_{v-1} + (2v/z) K_v`
accumulated through `logaddexp`; every combined term is positive, so no cancellation is
possible at any order. Median cost `7e-5` s per evaluation.

This is the second instance of the companion paper's `rem:conditioning`: analytic
tractability and numerical reliability are different properties.

## Prediction error against sample size

`run_prediction.py` measures how fast the posterior predictive approaches the predictive an
oracle would use who knew the rate. 400 replicate rates per regime, counts at unit exposure
so the accumulated exposure `F_n` of the consistency proposition equals the sample size `n`.
Error is the KL divergence from `Poisson(f lam*)` to the Sichel predictive, summed exactly
over the support rather than estimated from a held-out sample, so the curves carry no
Monte-Carlo noise beyond the spread over replicates.

Four regimes: three vary the tail of the mixing law at a fixed mean of 4 (prior `V/E` of
0.8, 4.0, 40), the fourth drops the mean to 0.5 where most counts are zero.

| regime | V/E | KL slope | rate-error slope | posterior-variance slope |
|---|---|---|---|---|
| light | 0.8 | -0.971 | -0.492 | -0.987 |
| moderate | 4.0 | -0.952 | -0.472 | -0.995 |
| heavy | 40 | -0.959 | -0.498 | -0.997 |
| sparse | 0.5 | -0.935 | -0.480 | -0.973 |
| **theory** | | **-1** | **-1/2** | **-1** |

The generating parameters set the constant, not the exponent. At `n = 1` the heaviest regime
starts at 0.244 nats against 0.178 for the lightest; by `n = 512` the spread has closed to a
factor of 1.14, so fifty times more prior dispersion costs about one extra observation.

Rescaling the posterior variance by `F_n / E[lam*]` collapses all four regimes onto 1 to
within 2 per cent by `n = 512`, which is the sharpest available check on the constant in the
consistency result, since it tests a constant rather than an exponent.

**This is a verification, not a comparison.** No competing model is run. A misspecified
negative-binomial arm on the same data is the obvious next study and is not here.

## Files

| file | what it is |
|---|---|
| `dgp.py` | the process, the closed forms, and the correctness gate |
| `run.py` | runs the gate, simulates, writes `data/` and `results/` |
| `run_prediction.py` | prediction error against sample size, over four parameter regimes |
| `visualize_prediction.py` | three panels into `figures/prediction.pdf`, included by the paper |
| `visualize.py` | four panels into `figures/dgp.png` |
| `viz.py` | figure style, carried over from the companion project |
| `data/environments.csv` | the eight sites: `(alpha, a, b)`, `omega`, `eta`, moments, attenuation, realised rate |
| `data/observations.csv` | survey trace, 512 reads per action at the realised rates |
| `data/marginal_histogram.csv` | 200k draws of the full two-stage process per diagnostic cell |
| `results/overdispersion.csv` | per action: analytic and empirical mean, variance, marginal and conditional Fano |
| `results/predictive_pmf.csv` | the analytic Sichel law over its support |
| `results/order_growth.csv` | posterior order per round, and where the unlogged route dies |
| `results/prediction_error.csv` | per regime and sample size: KL, rate error, posterior variance |
| `results/prediction_slopes.csv` | fitted log-log slopes |
| `results/gate_*.csv` | the three gate tables |

Run `python run.py` then `python visualize.py`. Roughly two minutes end to end; the gate
dominates.

## Open questions

**Two readings of "overdispersed count data".** Here the rate is drawn once per site and
held fixed, so the data are conditionally Poisson and the marginal overdispersion is
parameter uncertainty that shrinks as the sensor learns. The alternative is a random effect
redrawn per observation, under which the counts stay overdispersed however much is known
and there is no fixed rate to learn. This module implements the first, which is what the
conjugate update in the note describes and what an acquisition function scores. Which one
the paper's claim needs is not yet settled, and the answer changes what the baselines have
to be.

**Promotion.** `dgp.py` is shaped like the companion project's `methods/cpv/base.py`
interface (`actions`, `exposure`, `channels`, `sample_y`) so it can move to a `methods/`
package without rewriting once a second experiment needs it. It is here rather than there
because there is no second consumer yet.

**Not yet built.** No acquisition function, no policies, no baselines. The information
quantities need the same truncated support sum as the negative binomial member, and
`sichel_support` and `sichel_logpmf` are already in the arrangement they will want.
