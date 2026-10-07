# Alluvial diamond bulk sampling

## The setting the paper now uses (2026-10-03)

The manuscript's experiment is run by `experiments/comparison/run.py bulk`, on
`FaciesBulkSampling` in `domain.py`, posed the way the two telescope studies pose theirs:

- **Facies.** 96 blocks in four facies, cycled down the list. Each facies has its own GIG
  law of grade, heavy-tailed so that a few blocks carry most of the stones:

  | facies | mean grade (stones/m^3) | order | concentration |
  |---|---|---|---|
  | f1 | 0.25 | -0.5 | 0.3 |
  | f2 | 0.5 | -1.0 | 0.3 |
  | f3 | 1.0 | -1.5 | 0.3 |
  | f4 | 2.0 | -2.0 | 0.1 |

- **Priors from a calibration catalogue.** No agent is told those laws. Each model fits its
  own family, by maximum likelihood, to the grades of 20 previously mined blocks per facies
  (one fixed sample per facies, seed 7), as the telescope studies fit theirs to 20 catalogued
  members per source class.
- **A scarce budget.** 240 m^3 over 96 blocks, 2.5 m^3 a block on average, on the same pit
  schedule `{1, 2.5, 5, 10, 20, 40}` m^3 and the same plant recovery `U(0.55, 0.98)`.
- **Prediction at production scale.** NLPD on 24 held-out samples of 20 m^3 per block.

Why this regime: a 20-member catalogue is too small for a flexible conjugate rival (a gamma
mixture has `3J - 1` parameters) and too heavy-tailed for a gamma, and with most blocks
sampled once or not at all the prior is what the predictions rest on (Lemma `sichel_tail`
(iii): the gap between mixing laws is used up by exposure). In a 12-episode pilot the
gamma-Poisson and the gamma mixture were 0.104 and 0.072 nats behind the GIG at the full
budget (paired t = 11.5 and 5.7); with 48 blocks the gaps halved, and with NLPD scored on
5 m^3 samples they were 0.050 and 0.039.

The design below (12 blocks with an order of -0.5 throughout, every agent told the true prior
moments) is the earlier one. It made the gamma mixture a single gamma and the GIG correct by
construction, and gave a gap of 0.011 nats. Its outputs are not part of this release.

## The earlier single-study design

Written 2026-08-08. Reproducible from seed 0: `python run.py` then `python visualize.py`,
about four minutes end to end.

## Why this problem

The setting Sichel invented the GIGP for. Diamonds occur in clusters, so stone counts from
a sample of gravel are far more variable than a Poisson process allows, and no standard
discrete law reproduces observed stone-count frequencies. That model is used to **value** a
deposit once samples are in hand. It is not used to decide **which** samples to take, which
is the gap this study occupies.

Three things line up here that do not line up in word frequencies or photon counting:

1. The model class belongs to the domain, not to us, so "why this distribution" has an
   answer that predates the paper.
2. The exposure action is physical and unambiguous: cubic metres of gravel through the
   plant, times a known per-block plant recovery factor.
3. **The budget is naturally in cubic metres, not in number of samples.** That forces every
   criterion to be scored per unit of gravel. A budget counted in rounds would be won by
   whichever policy asks for the biggest sample, which is a measurement artefact.

**This is a simulator, not a field trial.** Its structure is taken from the domain; no
public dataset is used. Do not let the paper imply otherwise.

## The decision problem

```
lam_k          ~  GIG(alpha, a_k, b_k)     stones per m^3, drawn once per block
u = (k, v)                                 block, and volume of gravel to process
f(u) = v * r_k                             effective exposure, r_k the plant recovery
y | lam_k, u   ~  Poisson( f(u) lam_k )    stones recovered
cost(u) = v                                budget spent, in m^3
```

12 blocks, volumes `{1, 2.5, 5, 10, 20, 40}` m^3, budget 240 m^3, 48 replicate properties.
Blocks differ in expected grade (0.05 to 4 stones/m^3), in how clustered the stones are at
that grade (`omega` in 0.3 to 2.0, prior variance-to-mean 2 to 13), and in plant recovery
(0.55 to 0.98).

Common random numbers are exact rather than approximate: each block's stone field is
realised once as a Poisson process in effective processed volume, and a sample consumes the
next `v r_k` metres of it. Two policies that process the same gravel recover the same
stones, so any difference between them is a difference in decisions.

## What is scored

The three metrics of the problem statement, plus one diagnostic:

- **NLPD**: average surprisal of held-out stone counts under the agent's own predictive.
  Needs no true grade.
- **regret**: cumulative rate gap weighted by the gravel each sample processed,
  `sum_t v_t (lam_{k*} - lam_{k_t})`, where `k_t` is the block with the highest posterior
  mode after sample `t`. In stones. Ties at a zero mode (a gamma posterior with shape
  below one) go to the higher posterior mean.
- **inference cost**: wall-clock seconds in `act` plus `observe`, i.e. choosing a design and
  updating on its outcome. `inference_seconds` is cumulative; divide by `rounds`.
- **grade RMSE** over blocks, kept as a diagnostic and not reported in the paper's tables.

## Agents: two axes, crossed

An agent is a **model** plus an **acquisition criterion**. `methods/countmodels.py` holds
the models, `agents/` holds the criteria, and `run.py` crosses them, so the two can be varied
independently rather than one at a time.

| mixing law | params | conjugate | predictive | ms/design |
|---|---|---|---|---|
| GIG-Poisson (ours) | 3 | yes | Sichel | 3.1 |
| gamma-Poisson | 2 | yes | negative binomial | 0.23 |
| lognormal-Poisson | 2 | **no** | grid quadrature | 6.7 |

| criterion | what it scores |
|---|---|
| `eig` | expected information gain about the grade (ours) |
| `epig` | expected predictive information gain; no closed form here |
| `maxent` | maximum entropy sampling, `H[y|u]` |
| `d-optimality` | log Fano factor, the variance-stabilised Fisher form |
| `neyman` | Neyman optimal allocation, share proportional to `sqrt(lam_k / r_k)` |
| `predictive-variance`, `epistemic` | uncertainty sampling, two forms |
| `thompson` | posterior sampling |
| `uniform`, `random` | belief-free designs, run once |

Every model is initialised from the **same prior mean and variance per block** and none is
told the true parameters. Every model returns a pmf on the non-negative integers, so NLPD is
comparable across models. Each model's acquisition is gated against nested Monte Carlo *in
its own family*.

EPIG is not crossed with the lognormal: its outer expectation needs a posterior predictive
entropy per node, and under a grid posterior that is a quadrature inside a quadrature. The
exclusion is a result about non-conjugacy and is reported, not hidden.

### Results, 48 properties

**The criterion effect is invariant to the model.** NLPD relative to EIG under the same
model, in nats:

| criterion | GIG | gamma | lognormal |
|---|---|---|---|
| EPIG | +0.003 | -0.002 | not run |
| D-optimality | +0.001 | +0.001 | -0.003 |
| Neyman | +0.015 | +0.018 | +0.009 |
| epistemic var. | +0.077 | +0.066 | +0.083 |
| total var. | +0.089 | +0.079 | +0.084 |
| Thompson | +0.214 | +0.219 | +0.191 |
| max-entropy | +0.302 | +0.308 | +0.296 |

**The model effect is invariant to the criterion**, and much smaller: the gamma is behind the
GIG by +0.012 to +0.018 nats (t = +3.3 to +3.8) under the criteria that spread; the lognormal
by +0.006 (t = +2.0) under EIG and less elsewhere.

**Choosing the wrong acquisition costs ~25x what choosing the wrong mixing law costs.**

Caveat: the generating process *is* the GIG-Poisson law, so our model is correct by
construction and the model-axis number measures what misspecification costs rather than what
the family is worth in general. The complementary sweep over generating dispersion is not
run.

## Gate

`run.py` checks the closed-form EIG against bias-corrected nested Monte Carlo before
running anything, and aborts on failure.

## Files

| file | what it is |
|---|---|
| `domain.py` | the environment: ground truth, actions, budget, gravel. Holds no model |
| `../../methods/countmodels.py` | the three mixing laws behind one interface |
| `../../agents/` | one module per acquisition criterion |
| `run.py` | gate, sequential study over the criterion-by-model grid |
| `visualize.py` | three panels into `figures/bulk.pdf`, included by the paper |
| `results/sequential.csv` | per policy, property and checkpoint: inference seconds, NLPD, grade RMSE, regret |
| `results/allocation.csv` | volume each policy gave each block, against its true grade |
| `results/gate_eig.csv` | each model's EIG against nested Monte Carlo |

## Notes on the numerics

Two changes were forced by this study and both are in `methods/gigpoisson.py`.

**The Bessel routing.** The order recurrence is cancellation-free but its rounding compounds
along the run: against 30-digit references it is exact for the first few dozen orders, out
by 3e-12 at 10^3, 6e-9 at 5e4 and 3e-7 at 2e5, where the uniform asymptotic expansion is
still exact. It is also a Python loop and an order of magnitude slower there. `sichel_logpmf`
now takes the recurrence below 512 orders and the direct route above.

**The support tolerance.** A heavy-tailed predictive needs a support of order 10^5 to reach
a tail mass of 1e-13, which is far more than an entropy needs. The applied study passes
`tail_tol=1e-9`; the verification study keeps the strict default.
