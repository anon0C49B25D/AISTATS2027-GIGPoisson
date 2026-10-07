# Experiments

Grouped by where the rates come from.

| path | data | what it answers |
|---|---|---|
| `synthetic/verification/` | generated from (2) | correctness gate and convergence rates; the model is right by construction |
| `synthetic/shape-scarcity/` | generated from (2) | when the exact criterion separates from a two-moment surrogate |
| `synthetic/nmc-cost/` | generated from (2) | what nested Monte Carlo costs against the quadratures |
| `bulk-sampling/` | alluvial diamond simulator | allocating a processing budget over blocks in facies |
| `gamma-ray/` | Fermi-LAT 4FGL | scheduling telescope time over source classes |
| `hard-xray/` | Swift-BAT | also schedules telescope time over source classes |
| `comparison/` | the three above | every agent on every setting, one harness: the paper's Section 6 |
| `eig-accuracy/` | fitted priors of the three above | how far the deterministic EIG is from a tighter reference, also where its support cap binds |
| `safecast-fukushima/` | measured Geiger counts (Safecast) | the priors fitted to measured counts of a heavy-tailed field: where the GIG pays |
| `safecast-osaka/` | measured Geiger counts (Safecast) | the same on natural background, a non-overdispersed field: where it buys nothing |
| `common/` | — | `viz.py`, the figure style every study shares; the harness and comparison figures; `countfit.py`, `natural.py` and `safecast*.py` for the two measured-count studies |

The three settings of the paper are run by `comparison/run.py`, all with class priors fitted
to calibration catalogues of 20 known members per class (see `comparison/README.md`).

The two Safecast studies (added 2026-10-05) are the only ones whose **counts are measured
rather than simulated**: raw five-second pulse counts of a Geiger tube, at a known exposure.
They have no design or budget; they fit each mixing law to the counts of a calibration
catalogue of cells by marginal likelihood and score held-out counts of new cells. Same
instrument and protocol in both, so the only difference between them is the population of
rates. They are reported in the measured-counts section of the paper, with the data,
protocol and full results in its appendix, whose figure
(`figures/natural_counts.pdf`) and table (`tables/natural_counts.tex`) are written by
`python experiments/common/natural_paper.py` from the two studies' `results/`.

## Conventions

Each study directory is self-contained: `run.py` (or `compare.py`) writes `results/`,
`visualize.py` writes `figures/`, and a `README.md` says what the study found. Everything is
reproducible from seed 0. A study never writes into another study's directory.

`common/viz.py` used to live inside the verification study, which meant every other
visualizer reached into it by relative path. It is now a peer of the studies that use it.