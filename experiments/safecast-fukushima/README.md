# Measured Geiger counts around Fukushima Daiichi: where the GIG prior pays

Written 2026-10-05. `python fetch.py`, then `python run.py`, then `python visualize.py`.
Companion: `../safecast-osaka/`, the same study on natural background, where it does not.

## Why this data set

Every other study in `experiments/` simulates its counts: the rates are measured (Fermi,
Swift-BAT) or drawn from GIG laws (diamonds), and the counts are drawn from a Poisson law
about them. Here **the counts themselves are measured**. Safecast (Brown et al. 2016,
`brown2016safecast` in `references.bib`) is a citizen-science radiation survey released into
the public domain (CC0). Its bGeigie
Nano is a Geiger-Mueller tube (LND 7317) with a GPS receiver, driven around on a car, and
every five seconds it logs how many pulses the tube gave in those five seconds. A tube's
pulses are Poisson about the dose rate at the tube, and the five seconds are the exposure,
so the observation model of Section 2, `y ~ Poisson(f(u) lambda)`, is the measurement
process here, with nothing simulated. The rate includes the natural background and the
cosmic-ray component along with the fallout: it is the total count rate the tube sees, so
no additive background term is needed.

Around Fukushima Daiichi the field of rates is heavy-tailed, which is the regime the GIG
prior is built for: most of the cells in which the cars drove are at background, a few read
hundreds of times more.

## Setting

| | |
|---|---|
| region | 60 km around Fukushima Daiichi (37.4211 N, 141.0328 E) |
| logs | the 120 drive logs found with measurements from 2024-01-01 on (`data/imports.csv`) |
| why from 2024 | caesium-134 (half-life 2.1 y) is down to about 1 per cent of its 2011 activity, so a cell's rate drifts by a few per cent a year: one rate per cell over the window |
| context | a 100 m cell; its readings in time order |
| class | distance band from the plant, 0-10, 10-20, 20-40, 40-60 km: known before going out |
| exposure | 5 s per reading |
| quality control | a log whose median count inside the region is over 3x the median over logs is dropped (none here; two in Osaka) |
| contexts used | cells with at least 6 readings, so that something is left to predict |

## Protocol

`../common/natural.py`, identical for both Safecast studies.

- An **episode** splits each class's cells at random into a calibration pool and a test half.
- Each law's class prior is fitted by **marginal likelihood to the counts** of the first
  `n` cells of the pool (`../common/countfit.py`), for catalogues of `n = 10, 20, 40, 100,
  400` cells per class, nested. A class with fewer than `2n` cells uses its whole pool.
- Every test cell (up to 500 per class) has its readings 6 to 15 held out, and they are
  predicted after its first `j = 0, 1, 2, 5` readings. `j = 0` is a cell never measured.
- Two scores per test cell: the mean surprisal of the held-out readings one at a time, and
  the surprisal of their sum, one long integration of up to 50 s.
- 50 episodes; laws compared paired by episode (t-test and Wilcoxon, Holm-corrected over the
  rivals; "significant" means both at p < 0.01).

The laws: the GIG (ours); the gamma (its `omega -> 0` edge, the conjugate incumbent); a
gamma mixture with up to four components by BIC (shapes capped at 1e3, components needing
two cells, as in `methods/gammamixture.py`); the lognormal; and the two in-regime rivals of
the revision, the compound gamma (beta prime, a power-law tail) and the skew normal on
`log lambda` with the Azzalini-Arellano-Valle penalty. A Poisson with one rate per class
(no mixing) is the floor. `countfit.gate()` runs first and aborts on any failure: the GIG
and gamma predictives match `methods/` (Sichel and negative binomial) to 1e-8, the
quadratures match a brute-force integral to 1e-6, and every predictive sums to one.

## Results

From `results/` after `python run.py 50 6` (24 min on 6 workers). Figures:
`figures/data.pdf` (the field, its tail, the Poisson check) and `figures/scores.pdf`.

### The data

120 logs, 324 235 readings, 15 522 cells, of which 4 939 have six or more readings.

| band | cells | readings | median cpm | 99th pct cpm | max cpm | 99th pct / median | within-cell Fano (median, IQR) | Fano of a cell's 1st reading | Fano of its 10-reading sum |
|---|---|---|---|---|---|---|---|---|---|
| 0-10 km | 1109 | 150741 | 47 | 489 | 16739 | 10.5 | 1.11 (0.95-1.35) | 295 | 4584 |
| 10-20 km | 713 | 23855 | 41 | 88 | 104 | 2.1 | 1.08 (0.85-1.31) | 1.48 | 5.0 |
| 20-40 km | 837 | 13822 | 38 | 166 | 263 | 4.4 | 1.04 (0.82-1.35) | 2.19 | 15.3 |
| 40-60 km | 2280 | 111939 | 39 | 76 | 142 | 1.9 | 1.04 (0.83-1.30) | 1.33 | 2.9 |

Within a cell the counts are close to Poisson (Fano factor about 1.04-1.11); across cells
they are overdispersed by up to three orders of magnitude, and more so the longer the
exposure, as `1 + f V / E` says they must be.

### Does the third parameter earn its place? Every law fitted to every cell of a band

`results/population.csv`; dAIC from the best law of the band.

| band | GIG order | GIG omega | Poisson | gamma | lognormal | gamma mixture | compound gamma | log-skew-normal | GIG |
|---|---|---|---|---|---|---|---|---|---|
| 0-10 km | -3.67 | 2e-6 | 2943721 | 2105 | 770 | **0** (J=4) | 391 | 177 | 391 |
| 10-20 km | -12.23 | 5e-5 | 9578 | 77 | 55 | **0** (J=2) | 45 | 24 | 45 |
| 20-40 km | -7.31 | 4e-6 | 11506 | 506 | 336 | **0** (J=3) | 230 | 132 | 230 |
| 40-60 km | -24.63 | 9e-5 | 7822 | 192 | 127 | **0** (J=3) | 79 | 4 | 78 |

- In every band the GIG fits at its **inverse-gamma edge** (`omega -> 0`, `b -> 0`), with a
  negative order: a power-law tail `P(lambda > x) ~ x^order`, as heavy as `x^-3.7` in the
  innermost band. No gamma has such a tail, and the GIG is 33-1714 AIC ahead of it.
- The compound gamma has the same inverse-gamma limit (shape `r -> infinity`) and lands on
  the same law, within 0.1 nats of log-likelihood over hundreds of cells. On this field the
  two three-parameter families with a power-law tail are one model.
- With **hundreds to a thousand cells per band** the gamma mixture (two to four components)
  fits best everywhere, and the skew normal on `log lambda` beats the GIG too. The tail of the
  field is not a single power law: `figures/data.pdf` (b) shows a shoulder between 100 and
  1000 cpm, and a handful of cells at 5000-17000 cpm, that only the mixture follows. A large
  catalogue can afford the components to describe that; a small one cannot, which is the
  next table.

### Predicting new cells from small catalogues

`results/summary.csv`. Each entry is the rival's mean held-out surprisal minus the GIG's, in
nats, over 50 episodes (positive: the GIG predicted better), with the number of episodes in
which the GIG was ahead. **Bold**: significant (paired t and Wilcoxon, both p < 0.01 after
Holm over the six rivals). The first row is the GIG's own surprisal.

**A cell never measured (`j = 0`), one 5 s reading at a time:**

| | 10 cells/band | 20 | 40 | 100 | 400 |
|---|---|---|---|---|---|
| GIG (nats) | 2.541 | 2.382 | 2.358 | 2.349 | 2.344 |
| gamma | **+0.578** (50/50) | **+0.436** (50/50) | **+0.351** (50/50) | **+0.307** (50/50) | **+0.212** (50/50) |
| gamma mixture | **+0.664** (49/50) | **+0.384** (49/50) | **+0.194** (47/50) | **+0.153** (39/50) | +0.011 (11/50) |
| lognormal | +0.003 (44/50) | **+0.050** (48/50) | **+0.040** (50/50) | **+0.035** (50/50) | **+0.029** (50/50) |
| compound gamma | -0.023 (22/50) | -0.001 (29/50) | -0.000 (30/50) | -0.000 (29/50) | -0.000 (29/50) |
| log-skew-normal | -0.003 (42/50) | **+0.028** (48/50) | **+0.009** (47/50) | **+0.006** (42/50) | **+0.004** (46/50) |
| Poisson, no mixing | **+6.54** (50/50) | **+5.44** (50/50) | **+5.37** (50/50) | **+4.83** (50/50) | **+4.59** (50/50) |

**A cell never measured, the 50 s sum of its held-out readings (one long integration):**

| | 10 cells/band | 20 | 40 | 100 | 400 |
|---|---|---|---|---|---|
| GIG (nats) | 3.954 | 3.578 | 3.527 | 3.504 | 3.495 |
| gamma | **+1.156** (50/50) | **+0.650** (50/50) | **+0.502** (50/50) | **+0.438** (50/50) | **+0.320** (50/50) |
| gamma mixture | **+1.791** (49/50) | **+0.631** (48/50) | **+0.242** (43/50) | **+0.176** (32/50) | -0.016 (6/50) |
| lognormal | -0.039 (45/50) | **+0.082** (49/50) | **+0.070** (50/50) | **+0.063** (50/50) | **+0.058** (50/50) |
| compound gamma | -0.051 (15/50) | -0.001 (30/50) | -0.000 (37/50) | +0.000 (44/50) | +0.000 (43/50) |
| log-skew-normal | -0.052 (42/50) | **+0.035** (43/50) | -0.003 (11/50) | **-0.012** (0/50) | **-0.015** (0/50) |
| Poisson, no mixing | **+61.5** (50/50) | **+51.7** (50/50) | **+51.0** (50/50) | **+46.6** (50/50) | **+44.7** (50/50) |

**After five readings of the cell (`j = 5`), one reading at a time:**

| | 10 cells/band | 20 | 40 | 100 | 400 |
|---|---|---|---|---|---|
| GIG (nats) | 2.312 | 2.281 | 2.277 | 2.275 | 2.274 |
| gamma | +0.062 (47/50) | **+0.025** (48/50) | **+0.019** (47/50) | **+0.018** (49/50) | **+0.015** (50/50) |
| gamma mixture | +0.125 (47/50) | **+0.028** (44/50) | +0.005 (31/50) | +0.001 (21/50) | **-0.006** (3/50) |
| lognormal | -0.004 (37/50) | **+0.004** (47/50) | **+0.004** (50/50) | **+0.004** (50/50) | **+0.004** (50/50) |
| compound gamma | -0.003 (14/50) | -0.000 (34/50) | -0.000 (33/50) | +0.000 (41/50) | +0.000 (45/50) |
| log-skew-normal | -0.005 (36/50) | +0.001 (31/50) | **-0.002** (3/50) | **-0.003** (0/50) | **-0.003** (0/50) |

### What this says

- **Against the conjugate incumbent, the gamma-Poisson, the GIG is ahead at every catalogue
  size, in every episode**: 0.21-0.58 nats per reading on a cell never measured, 0.32-1.16
  nats on a 50 s integration, and still 0.015-0.06 per reading after five readings. That is
  the power-law tail the gamma cannot have.
- **Against the gamma mixture it is ahead with 10-100 cells per band** (0.15-0.66 nats per
  reading on a new cell), and **the mixture catches up with 400**: level on new cells, ahead
  by 0.006 nats per reading once five readings are in. The same pattern as Table 1 of the
  paper, on measured counts: what the third parameter buys over a flexible conjugate prior
  is efficiency in the catalogue.
- **Against the lognormal**, a small and consistent lead of 0.03-0.05 nats per reading from
  20 cells per band on.
- **The two in-regime rivals are no worse.** The compound gamma is the same fitted law. The
  penalised skew normal on `log lambda` is slightly behind on single readings of a new cell
  (0.004-0.028 nats), slightly ahead once five readings are in (0.002-0.003), and ahead by
  0.01-0.02 nats on long integrations once 40 or more cells per band are known. It is not
  conjugate: its predictive here is a quadrature, and in a design loop its posterior would
  need one at every step.
- **With 10 cells per band nothing with a flexible tail is better than anything else**: the
  GIG, the compound gamma, the skew normal and the lognormal are within 0.05 nats and none of
  the differences is significant, while the gamma and the mixture are 1.2-1.8 nats behind on
  a long integration. A catalogue of ten does not identify a tail: the fitted GIG order runs
  from -26 to +31 (10th to 90th percentile over episodes and bands), against -25 to -3.6
  with 400 cells, where every fit has a power-law tail.
- **Ignoring the heterogeneity is not an option here**: one rate per band (Poisson, no
  mixing) is 4.6-6.8 nats per reading behind.

## Caveats

- **A moving detector.** A five-second reading integrates over the 50-80 m the car moves,
  and a 100 m cell holds readings from different passes, lanes and days. Within-cell
  dispersion is close to Poisson (median Fano factor 1.04-1.11 by band), but above about
  100 counts per reading the spatial gradient inside a cell shows as extra variance
  (`figures/data.pdf`, panel c). Every law shares this likelihood, so it does not favour one.
- **Where cars drive.** Cells are on roads, and cells with six or more readings on the roads
  driven most. The population is that of driven cells, not of the land.
- **Dead time.** A Geiger tube misses pulses that arrive within its dead time (of order
  50-100 microseconds) of the last one, which makes counts slightly underdispersed at high
  rates. Over 99 per cent of the cells read below 500 cpm, about 8 pulses a second, where the
  tube is dead for a fraction of order 1e-3 of the time; it approaches one per cent only in
  the handful of cells near 10^4 cpm.
- **Bounded fits.** On a catalogue that happens to look homogeneous, the maximum-likelihood
  GIG runs towards its point mass (concentrations of 1e34 were seen) and its normalising
  constants lose every digit. The fit is bounded at a concentration of 1e5 and an order of
  +/-1000, and the gamma at a shape of 1e6, where each is already indistinguishable from its
  point-mass limit for these data. 15 of the 1000 GIG fits of the episodes stopped at the
  concentration bound, none at the order bound. `countfit.gate()` tests a homogeneous
  catalogue, and the scoring stops on any negative or non-finite surprisal.
- **Episodes resample one data set.** Splits overlap, so the paired tests measure
  consistency across splits, not sampling from a population of regions.
- **The window.** 2024-2026 only; earlier logs are excluded on purpose (decay), so the
  number of logs is what Safecast volunteers drove in those years.

## Files

| file | what it is |
|---|---|
| `config.py` | the region: centre, radius, window, distance bands, cell size |
| `fetch.py` | downloads the logs listed in `data/imports.csv` (or `--discover`s them) and parses them into `data/readings.npz` |
| `run.py` | the gate, the description, the population fits and the 50 episodes |
| `visualize.py` | `figures/data.pdf` and `figures/scores.pdf` |
| `data/imports.csv` | the Safecast log ids used, with their first measurement date |
| `data/readings.npz` | every valid reading in the region: log id, time, count, position |
| `results/describe.csv` | per class: cells, readings, spread of the rates, dispersion |
| `results/population.csv` | every law fitted to every cell of a class: log-likelihood, AIC, BIC, parameters |
| `results/scores.csv`, `results/fits.csv` | per episode: scores, and every fitted prior |
| `results/summary.csv` | paired comparison with the GIG per catalogue size, condition and score |
| `results/cells.csv`, `results/dispersion.csv` | per cell, for the figures |
| `results/partial/` | one file pair per finished episode; a rerun reads them back and resumes (episodes written by older fitting code are recomputed) |

Raw logs are cached outside the repository (`SAFECAST_CACHE`, default
`~/.cache/safecast-bgeigie`, about 0.5 MB a log).
