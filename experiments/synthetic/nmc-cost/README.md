# What nested Monte Carlo costs

Section 3.3 computes the expected information gain by two deterministic quadratures. Outside
a conjugate family the same quantity is estimated by nested Monte Carlo. This study measures
what that substitution costs, in three ways.

## The estimator

Draw `N` outer pairs from the joint and estimate the evidence at each outcome by an inner
average over `M` fresh rate draws:

```
EIG ~= (1/N) sum_i [ log p(y_i | lam_i) - log( (1/M) sum_j p(y_i | lam'_j) ) ]
```

The inner average sits inside a logarithm, so by Jensen the estimate is biased **up**, at
order `1/M`. The inner sample is shared across outer draws, as in the correctness gates of
the two allocation studies. `nmc.py` chunks the outer loop so memory stays at `chunk * M`
rather than `N * M`; unchunked, `N = M = 10^4` is eight hundred megabytes of matrix.

## What the three studies found

**A. Accuracy against wall-clock** (`results/accuracy.csv`). The exact quadrature costs
between 1.2 and 34 ms depending on the belief and the exposure. To reach an error of about
0.01 nats the estimator needs `N = M` in the thousands, which is one to two orders of
magnitude more wall-clock.

**B. Where the error lives** (`results/bias.csv`). The point the manuscript's phrase "a bias
no sample size removes" is reaching for, stated precisely: the bias is a function of the
**inner** sample size. At `M = 64` it sits near `+0.11` nats and stays there as `N` goes from
256 to 16384 — sixty-four times the work for no movement at all. Raising `M` is what retires
it, at roughly `1/M`.

**C. Whether it reaches the decision** (`results/decisions.csv`). Over 120 problems, each a
grid of contexts by exposures ranked by EIG per unit cost, the estimator's preferred action
differs from the exact one on 56% of problems at `n = 64` and still on 13% at `n = 1024`. An
estimator biased by a constant would rank correctly; this one does not, because the bias
depends on the candidate through its own predictive.

## The heavy-tailed regime is where it breaks down

`results/heavytail.csv`. At order `-2` and an exposure of 40 the error distribution is itself
heavy-tailed, so the median looks acceptable while the tail does not:

| n | median | p90 | worst of 120 |
|---|---|---|---|
| 256 | 0.099 | 2.27 | 27.7 |
| 1024 | 0.053 | 0.64 | 2.59 |
| 4096 | 0.039 | 0.52 | 5.58 |

against an exact value of 1.435 nats. Rare outer draws land where the shared inner sample has
almost no mass, and the log of a near-zero evidence estimate is unbounded. Averaging over
repeats does not help, because the estimator is used **once** per decision. Read `p90` as the
tail statistic; `p99` and `max` are single order statistics out of 120 and are quoted as
observed worst cases, not as estimates of a quantile.

That this is the heavy-tailed regime matters: it is the regime the third parameter of the GIG
exists for, so the estimator is least reliable exactly where the family is most needed.

## An honest floor for the quadrature

`results/floor.csv`. Plotting the quadrature against the estimator on a log error axis means
claiming a number for its own accuracy, so the two settings that control it — the node count
of the Gauss-Legendre rule over the mixing law, and the tail tolerance of the support sum —
are tightened and the value is watched. The shipped settings agree with much tighter ones to
between `1e-16` and `2.6e-8`, the worst case being the heavy tail at exposure 40. That number,
not zero, is where the quadrature is drawn in the figure.

## Running

```bash
cd experiments/synthetic/nmc-cost
python run.py        # ~11 min
python refine.py     # ~25 min: error floor, heavy-tail quantiles, 120-problem decisions
python visualize.py
```

Reproducible from seed 0. `refine.py` overwrites `results/decisions.csv` with the larger run.
