# When does an exact information criterion pay?

The bulk-sampling and gamma-ray studies both find expected information gain, EPIG and local
D-optimality indistinguishable on accuracy. This study explains why, and finds the regime
where they separate.

## The reason for the tie

D-optimality here is

```
D-opt(u) = log(1 + f(u) V[lam] / E[lam])
```

which reads the belief through a single scalar, the index of dispersion. The exact criterion
reads the whole mixing law. Measuring the two against each other as the expected count grows,

```
EIG(u) - (1/2) D-opt(u)  ->  c(belief)
```

where `c` depends on the **shape** of the mixing law and not on the action. So the two are an
affine transform of each other along the exposure axis, and:

> If every context carries a prior of the same order, `c` is common to all of them and the
> two criteria rank candidates identically. No experiment run that way can separate them.

Both allocation studies are run that way. `bulk-sampling` calls
`prior_from_moments`, which holds the order at the `GIGPoisson` default and solves only for
the concentration. `gamma-ray` does fit an order per source class, but the
classes are spread over an index of dispersion from 2.9 to 15.0, and the offset `c` spans
about 0.31 nats, enough to flip a ranking only when two Fano factors lie within a factor of
roughly 1.9 of each other. Most class pairs there are outside that window.

## The design

Twelve contexts in two shape classes, alternating down the list.

| class | order | E[lam], low | V/E, low | E[lam], high | V/E, high |
|---|---|---|---|---|---|
| heavy | −2.0 | 0.1 | 1.30 | 1.0 | 4.00 |
| mild | −0.5 | 0.1 | 1.00 | 1.0 | 2.00 |

The classes share a mean, so no criterion can tell them apart by brightness. The heavy class
always carries the **larger** index of dispersion, so D-optimality strictly prefers it, while
the exact criterion strictly prefers the mild class. Neither is ever reduced to breaking a
tie: both have confident and opposite preferences, which `run.py`'s design gate verifies and
aborts on.

Why that direction. A large index of dispersion under a heavy-tailed mixing law is mostly
*unresolvable* — it comes from a small probability of a very large rate, and no exposure
recovers it. D-optimality reads that variance as something to be learned. The mutual
information knows how much of it a count can actually remove.

Exposures are `{0.5, 1, 2, 4}`, cost is the exposure itself, the budget is swept from 2 to 64
(0.17 to 5.3 per context), and both count levels are run. Every criterion carries the same
belief, because all of them run under the GIG-Poisson model: this is the upper block of
Tables 2 and 4, with the acquisition the only thing varying. Rates are drawn from each
class's own law, so the model is correctly specified and nothing but the ranking separates
two agents.

Each context's counting process is realised once as a Poisson process in exposure, so two
agents that spend the same exposure on the same context see the same counts.

## A note on fitting the priors

The first version fitted each class prior to a population of its own rates, as the gamma-ray
study does. That was abandoned: `scipy.stats.geninvgauss.fit` is unreliable on a heavy-tailed
sample, returning an index of dispersion of `0.25` on 4000 draws from the `order -2` class
against a true `1.30`. That collapses the contrast between the classes and quietly turns the
study into a comparison of two criteria that agree. The design gate exists because of it, and
`initial_beliefs` now hands every family the generating law of the class.

## Results

`run.py` writes `results/episodes.csv` (one row per episode, agent and budget) and
`results/summary.csv` (means, paired `t` against EIG, and standard errors).
`visualize.py` writes `figures/shape_scarcity.pdf`.

Reproducible from seed 0:

```bash
cd experiments/synthetic/shape-scarcity && python run.py && python visualize.py
```
