# Measured Geiger counts on natural background (Osaka): where the GIG prior buys nothing

Written 2026-10-05. `python fetch.py`, then `python run.py`, then `python visualize.py`.
Companion: `../safecast-fukushima/`, the same study on a heavy-tailed field, where it pays.

## Why this data set

A limitation should be shown on data, not argued. The GIG prior's third parameter exists to
move mass between the centre and the tail of the population of rates (Lemma 3). On a
population whose rates hardly vary, and whose counts are Poisson about them, there is no tail
to describe: the counts are **not overdispersed**, the Fano factor `1 + f V / E` of Eq. (6) is
close to one at every exposure, and the GIG can do no more than the simplest law.

Natural background radiation is that population. Osaka lies 583 km from Fukushima Daiichi,
too far for the 2011 fallout to raise its dose rate measurably, so the count rate of a cell is
the natural radioactivity of the ground and the buildings near it plus the cosmic-ray
component: it varies by tens of per cent between streets, not by orders of magnitude.
Radioactive decay is also the textbook Poisson process, so the within-cell counts are as close
to the model's likelihood as measured counts get. The data are the same kind as in the
Fukushima study: raw five-second pulse counts from Safecast bGeigie Nano drive logs (CC0;
Brown et al. 2016, `brown2016safecast` in `references.bib`).

## Setting

| | |
|---|---|
| region | 30 km around central Osaka (34.69 N, 135.50 E) |
| logs | the 256 drive logs found by the measurement API within 30 km (`data/imports.csv`), 2011-2025; background does not change over the years, so every year is used |
| context | a 100 m cell; its readings in time order |
| class | distance band from the centre, 0-7.5, 7.5-15, 15-22.5, 22.5-30 km (kept so that the protocol is the Fukushima one; the bands are not expected to differ) |
| exposure | 5 s per reading |
| quality control | a log whose median count inside the region is over 3x the median over logs is dropped: two logs (15999 and 19833, medians of about 50 counts per 5 s against 3), each a few minutes long, which were measuring a source rather than the ambient field |
| contexts used | cells with at least 6 readings |

## Protocol

Identical to the Fukushima study, `../common/natural.py`: 50 episodes, class priors fitted by
marginal likelihood to the counts of 10, 20, 40, 100 and 400 cells per class, held-out
readings 6 to 15 of up to 500 test cells per class predicted after 0, 1, 2 and 5 readings,
scored one reading at a time and as their 50 s sum. Same laws, same gate.

## Results

From `results/` after `python run.py 50 6` (29 min on 6 workers). Figures:
`figures/data.pdf` and `figures/scores.pdf`.

### The data

256 logs (two dropped by quality control), 790 731 readings, 59 148 cells, of which 27 800
have six or more readings.

| band | cells | readings | median cpm | 99th pct cpm | max cpm | 99th pct / median | within-cell Fano (median, IQR) | Fano of a cell's 1st reading | Fano of its 10-reading sum |
|---|---|---|---|---|---|---|---|---|---|
| 0-7.5 km | 4571 | 148618 | 38 | 60 | 85 | 1.57 | 1.05 (0.86-1.27) | 1.09 | 1.6 |
| 7.5-15 km | 8224 | 219480 | 38 | 58 | 90 | 1.53 | 1.03 (0.83-1.25) | 1.08 | 1.3 |
| 15-22.5 km | 7469 | 148501 | 39 | 62 | 93 | 1.59 | 1.02 (0.80-1.26) | 1.10 | 1.3 |
| 22.5-30 km | 7536 | 192777 | 38 | 61 | 95 | 1.60 | 1.03 (0.82-1.27) | 1.13 | 1.5 |

Against the Fukushima study's innermost band (99th percentile 10.5 times the median, Fano
factor 295 for a single reading) this population barely varies: a cell's first reading is
overdispersed by 10 per cent across cells, a 50 s sum by 30-60 per cent.

### Every law fitted to every cell of a band

`results/population.csv`; dAIC from the best law of the band.

| band | GIG order | GIG omega | Poisson | gamma | lognormal | gamma mixture | compound gamma | log-skew-normal | GIG |
|---|---|---|---|---|---|---|---|---|---|
| 0-7.5 km | -65.1 | 9e-4 | 4743 | 124 | 111 | **0** (J=2) | 104 | 99 | 104 |
| 7.5-15 km | -124.9 | 1e-3 | 4402 | 180 | 175 | **0** (J=2) | 174 | 167 | 174 |
| 15-22.5 km | -98.5 | 9e-4 | 2843 | 100 | 86 | **0** (J=2) | 74 | 42 | 74 |
| 22.5-30 km | -67.2 | 1e-3 | 6060 | 116 | 90 | **0** (J=2) | 71 | 54 | 71 |

- With thousands of cells per band the rates are measurably heterogeneous (every mixing law
  is thousands of AIC ahead of one rate per band), by about 10 per cent between cells.
- The GIG again fits at its inverse-gamma edge, but with an order of -65 to -125: a tail
  index so large that the law is light-tailed for every practical purpose. It is ahead of
  the gamma by 6-45 AIC over 4 571-8 224 cells, **at most 0.003 nats per cell**, against up
  to 0.77 nats per cell in the Fukushima study.
- A two-component gamma mixture fits best in every band, by 70-180 AIC: a small group of
  cells reads higher than the rest (`figures/data.pdf` (b), above 70 cpm). Granite and a
  detector that reads a few per cent high would both produce one; which it is has not been
  checked. No single unimodal law, the GIG included, describes it.

### Predicting new cells

`results/summary.csv`. Rival minus GIG, **in units of 1e-4 nats**, over 50 episodes, with the
number of episodes the GIG was ahead; the first row is the GIG's own surprisal in nats.
**Bold**: significant (paired t and Wilcoxon, both p < 0.01 after Holm). Episodes resample
one data set, so with differences this small "significant" means consistent across splits,
not important.

**A cell never measured, one 5 s reading at a time:**

| | 10 cells/band | 20 | 40 | 100 | 400 |
|---|---|---|---|---|---|
| GIG (nats) | 2.020 | 2.018 | 2.016 | 2.016 | 2.015 |
| gamma | +0.55 (36/50) | **+0.45** (42/50) | **+0.39** (46/50) | **+0.43** (44/50) | **+0.33** (46/50) |
| gamma mixture | +0.55 (36/50) | **+0.50** (43/50) | **+0.44** (47/50) | **+0.55** (44/50) | +0.17 (35/50) |
| lognormal | -0.07 (30/50) | +0.15 (37/50) | **+0.16** (37/50) | +0.18 (34/50) | +0.06 (43/50) |
| compound gamma | -0.12 (9/50) | **-0.14** (12/50) | **-0.04** (14/50) | -0.01 (21/50) | -0.05 (21/50) |
| log-skew-normal | -0.01 (31/50) | +0.13 (36/50) | **+0.19** (36/50) | +0.18 (32/50) | +0.05 (39/50) |
| Poisson, no mixing | **+30** (45/50) | **+36** (47/50) | **+37** (48/50) | **+34** (50/50) | **+28** (50/50) |

**A cell never measured, the 50 s sum of its held-out readings:**

| | 10 cells/band | 20 | 40 | 100 | 400 |
|---|---|---|---|---|---|
| GIG (nats) | 3.001 | 2.987 | 2.976 | 2.969 | 2.965 |
| gamma | **+8.40** (41/50) | **+5.99** (44/50) | **+5.12** (44/50) | **+5.16** (45/50) | **+6.29** (48/50) |
| gamma mixture | **+8.40** (41/50) | **+8.73** (44/50) | **+7.40** (45/50) | **+6.80** (45/50) | **+5.62** (39/50) |
| lognormal | +3.76 (33/50) | **+2.11** (35/50) | **+1.35** (33/50) | +1.21 (30/50) | **+1.86** (40/50) |
| compound gamma | **-3.13** (8/50) | **-2.17** (9/50) | **-0.87** (15/50) | -0.17 (23/50) | -0.07 (30/50) |
| log-skew-normal | +2.89 (34/50) | **+1.76** (34/50) | +0.96 (30/50) | +0.88 (29/50) | +0.90 (32/50) |
| Poisson, no mixing | **+305** (50/50) | **+351** (50/50) | **+375** (50/50) | **+377** (50/50) | **+350** (50/50) |

**After five readings of the cell, the 50 s sum:**

| | 10 cells/band | 20 | 40 | 100 | 400 |
|---|---|---|---|---|---|
| GIG (nats) | 2.995 | 2.980 | 2.969 | 2.962 | 2.957 |
| gamma | **+10** (41/50) | **+6.85** (44/50) | **+6.03** (44/50) | **+5.55** (40/50) | **+7.27** (44/50) |
| gamma mixture | **+10** (41/50) | +10 (44/50) | **+7.79** (45/50) | **+6.90** (40/50) | +3.66 (35/50) |
| lognormal | **+5.90** (38/50) | +2.72 (35/50) | +1.54 (34/50) | +1.16 (27/50) | **+2.34** (38/50) |
| compound gamma | **-5.58** (6/50) | **-3.46** (10/50) | **-1.50** (17/50) | -0.21 (25/50) | +0.08 (32/50) |
| log-skew-normal | **+3.89** (36/50) | +2.12 (35/50) | +0.85 (32/50) | +0.78 (27/50) | +1.35 (31/50) |
| Poisson, no mixing | **+365** (50/50) | **+417** (50/50) | **+446** (50/50) | **+455** (50/50) | **+428** (50/50) |

### What this says

- **The third parameter buys nothing here.** Every mixing law is within 0.0001 nats per
  reading and 0.0011 nats per 50 s sum of the GIG, at every catalogue size and with or
  without readings of the cell. The GIG's lead over the gamma, 0.00003-0.00006 nats per
  reading on a new cell, is about ten thousand times smaller than in the Fukushima study
  (0.21-0.58), and the compound gamma is marginally ahead of the GIG. The GIG costs Bessel
  functions and buys nothing the gamma-Poisson does not give in closed form.
- **Even the mixing barely matters.** One rate per band, no mixing at all, is 0.003-0.004
  nats per reading behind, and 0.03-0.05 nats on a 50 s sum: the heterogeneity is real but
  small, and the longer the exposure the more it shows, as `1 + f V / E` says.
- This is the regime the abstract's "overdispersed count rates" excludes. When the counts of
  new contexts are close to Poisson about a common rate, the choice of mixing law, the GIG's
  advantage included, is immaterial; the model is not wrong here, it is unnecessary.

## Caveats

- Natural background has a little real structure (granite, building materials, bridges,
  tunnels that shield the cosmic-ray component), which is the heterogeneity measured here. A
  region with more varied geology would show more; it would still be light-tailed.
- As in the Fukushima study: a moving detector, cells on driven roads, and episodes that
  resample one data set.

## Files

As in `../safecast-fukushima/README.md`: `config.py`, `fetch.py`, `run.py`, `visualize.py`,
`data/imports.csv`, `data/readings.npz`, `results/*.csv`, `results/partial/` (one file pair
per finished episode, so an interrupted run resumes), `figures/data.pdf`,
`figures/scores.pdf`.
