# Head-to-head on the three settings of the paper

Rewritten 2026-10-03, when every setting moved to the small-catalogue regime and every agent
was re-run. The previous settings are not part of this release. With full
catalogues the gamma mixture came within 0.01 nats of the GIG and top-m regret did not
separate the models.

```bash
python experiments/comparison/train_dad.py bulk       # likewise gamma, hardxray (2-4 min each)
python experiments/comparison/run.py all --reps 50    # the study
python experiments/comparison/significance.py         # paired tests, Holm-corrected
python experiments/comparison/prior_fit.py            # the priors alone, by catalogue size
python experiments/comparison/tables.py               # the manuscript's tables, into experiments/tables/
python experiments/comparison/paper_figures.py        # the manuscript's figures, into figures/
python experiments/<study>/visualize_comparison.py    # per-study figures, per group of methods
```

**Additional runs of 2026-10-04.** New runs go under the
`fact_` prefix so the runs above stay as they were; `run_revision.sh` holds every command:

```bash
experiments/comparison/run_revision.sh factorial    # 3 priors x {EIG, D-opt, systematic}, LUCB, nested EIG under the GIG
experiments/comparison/run_revision.sh rivals       # compound gamma and penalised skew-normal, by EIG
experiments/comparison/run_revision.sh overhead     # cost c0 + tau, two levels of c0 per setting
experiments/comparison/run_revision.sh diagnostics  # nmc_agreement.py, prior_only.py, significance, tables
experiments/comparison/run_revision.sh timing       # single-process cost per EIG evaluation (idle machine)
```

The harness now also records `nlpd_obs` and `nlpd_unobs` (NLPD over observed and unobserved
contexts), `mean_dwell` and `frac_shortest`; runs written before load these as NaN. The
environments take an `overhead` (per-observation cost, default 0). The GIG with nested EIG
(`methods/countmodels.py:GIGPoissonNMC`, plain or PCE) runs at N = M chosen per setting to match
the exact EIG's cost per round (`agents.NMC_SIZES`). The compound gamma is
`methods/compoundgamma.py`; the penalised skew-normal fit is `logskewnormal.penalised_fit`.

Reproducible from seed 0. Episodes run in parallel (`--jobs`, default 12), one
single-threaded process per episode. Run with the `agents` conda environment (Python 3.14,
numpy 2.4, scipy 1.17, matplotlib 3.10), which reproduces the archived Linux runs exactly
(checked on the gamma-ray survey at 20 and 40 sources per class); the figures need
matplotlib 3.7 or later.

## The settings

Every setting is posed in the regime the model is built for. Contexts fall into classes;
every model fits its prior for a class, by maximum likelihood in its own parameters, to a
**calibration catalogue of 20 known members of that class**; the list of contexts is long
against the budget; and prediction is scored on long integrations.

| setting | contexts | classes | budget | actions | NLPD scored at |
|---|---|---|---|---|---|
| `bulk` | 96 blocks | 4 facies with GIG laws of grade (simulated) | 240 m^3 | 1-40 m^3 | 20 m^3 |
| `gamma` | 192 Fermi-LAT 4FGL sources, 1-99 % flux band | 5 source classes | 120 Ms | 0.5-25 Ms | 25 Ms |
| `hardxray` | 192 Swift-BAT 105-month sources, 1-99 % band | 6 source classes | 120 ks | 0.5-25 ks | 20 ks |

They are the `SETTINGS` of `run.py`; the flags `--sources`, `--eval-time`, `--pop-size` and
`--band` override them, and an override prefixes the output files (`pop40_`, `popall_`,
`pop200_`, ...). The facies laws of the bulk setting are in `bulk-sampling/README.md`.

Why this regime. A mixture of `J` gammas has `3J - 1` parameters against the GIG's three, and
a gamma cannot follow the power-law tail of a flux-limited population or of a heavy-tailed
facies law. With most contexts observed once or not at all, the prior is what the predictions
rest on, and the gap between mixing laws is largest where little exposure has been spent
(Lemma `sichel_tail` (iii)). The priors alone show it (`prior_fit.py`, both telescope
catalogues, 20 draws per size): the fitted GIG is ahead of the fitted gamma mixture by
2.87, 0.51, 0.19, 0.07 and 0.006 nats per held-out source with 10, 20, 40, 80 and all
catalogued members per class, and ahead of the gamma by 0.28 to 0.49 at every size.

## Agents

`agents.build_comparison()`, fifteen in all.

| name | model | allocation |
|---|---|---|
| `eig` | GIG-Poisson (ours), conjugate | expected information gain, two quadratures |
| `eig@gamma-poisson` | gamma-Poisson, conjugate | EIG, exact sum |
| `eig@gammamix-poisson` | finite gamma mixture, J by BIC (1-4), conjugate | EIG, exact |
| `eig@logskewnormal-quad` | log-skew-normal prior, posterior by quadrature | EIG by quadrature |
| `eig@logskewnormal-poisson` | log-skew-normal prior, MCMC (2048 chains x 30 MH steps) | EIG by nested MC, N = M = 1024 |
| `eig@logskewnormal-poisson-m8k` | the same | nested MC, N = 1024, M = 8192 |
| `bo-ei@gp-poisson` | GP over log-rates, Poisson output, RBF kernel on prior features | expected improvement x fidelity |
| `bo-ei@gp-poisson-cat` | the same, categorical kernel on class | the same |
| `systematic` | GIG-Poisson | round-robin at the largest action that visits every context once |
| `random` | GIG-Poisson | uniform over contexts and actions |
| `d-optimality` | GIG-Poisson | local D-optimality per unit cost |
| `thompson` | GIG-Poisson | posterior sampling at the largest action |
| `lucb`, `lucb@gamma-poisson` | GIG-Poisson, gamma-Poisson | Bayes-LUCB for the top-5 set, shortest action |
| `dad` | GIG-Poisson | amortised linear-softmax policy (REINFORCE on sPCE), weights in `dad/` |

**DAD.** Trained per setting in 2-4 minutes. Validation kept the starting point, D-optimality
per unit cost, for bulk sampling and the gamma-ray survey; for hard X-ray it selected a
trained policy whose held-out sPCE (277.4) is below the start's (285.0), and which predicts
0.023 nats worse than EIG.

**Memoisation (2026-10-03).** Every context of a class starts from the same prior, so the
first round of an information criterion needs one evaluation per class and action rather than
one per context. The exact log-skew-normal quadrature already memoised on that; the GIG and
the gamma mixture now memoise their EIG on the belief and the exposure, and the MCMC
log-skew-normal memoises its nested estimate at the prior (once a context has data its chains
are its own and every call draws afresh). Values are unchanged; first rounds are cheaper. The
gate calls the raw nested estimator, `nmc_eig`, so its replicates stay independent.

## Metrics (`experiments/common/harness.py`)

- **NLPD**: average surprisal of 24 held-out counts per context at the evaluation exposure.
- **top-m regret**, m = 1..5: `1 - sum_{k in S} lam_k / sum_{k in S*} lam_k`, S the agent's m
  contexts of highest posterior mean, S* the true top m; simple regret at each checkpoint.
- **time**: wall-clock of `act` plus `observe`, per round, with 12 episodes running at once on
  one machine (8 cores, 16 threads, Windows).
- diagnostics: RMSE and log10 error of the posterior mean rate, fraction of contexts touched.

`significance.py` tests every pair of methods per setting on NLPD at the full budget and on
top-1 and top-5 regret, at the full budget and as the area under the budget curve: paired t
and Wilcoxon, each Holm-corrected over all pairs in the setting.

## Results, 50 episodes

NLPD at the full budget, nats; the difference is the rival minus ours, paired by episode
(positive: ours better), with the episodes in which ours is better.

| setting | GIG-Poisson | gamma-Poisson | gamma mixture | BO-EI (RBF / categorical) |
|---|---|---|---|---|
| bulk | 2.917 | +0.106 (49/50, t 15.4) | +0.069 (48/50, t 10.4) | +0.227 / +0.195 |
| gamma-ray | 5.412 | +0.155 (50/50, t 20.0) | +0.105 (48/50, t 13.4) | +0.348 / +0.355 |
| hard X-ray | 6.671 | +0.064 (49/50, t 14.3) | +0.046 (47/50, t 11.8) | +1.251 / +1.259 |

Every one of these is significant at p < 1e-11 after Holm. The gaps are wider at the first
checkpoint, when most contexts are unobserved: +0.138/+0.093 (bulk, 30 m^3), +0.260/+0.203
(gamma-ray, 15 Ms), +0.435/+0.335 (hard X-ray, 15 ks).

Top-m regret, area under the budget curve (mean over the five checkpoints):

| setting | m | GIG | gamma | mixture |
|---|---|---|---|---|
| bulk | 1 | 0.086 | 0.139 (t 4.5, Holm p 0.002) | 0.099 (t 1.2, n.s.) |
| bulk | 5 | 0.123 | 0.166 (t 5.6, Holm p 2e-5) | 0.130 (t 1.3, n.s.) |
| gamma-ray | 1 | 0.031 | 0.033 (n.s.) | 0.033 (n.s.) |
| gamma-ray | 5 | 0.054 | 0.058 (t 2.6, n.s. after Holm) | 0.058 (t 2.8, n.s. after Holm) |
| hard X-ray | 1 | 0.008 | 0.008 | 0.007 (n.s.) |
| hard X-ray | 5 | 0.018 | 0.018 | 0.016 (t -1.9, n.s.) |

So: the GIG is ahead on NLPD in all three settings, ahead on regret against the gamma in bulk
sampling, nominally ahead on gamma-ray top-5 regret (0.009 against 0.018 at the full budget,
not significant after Holm), and tied on regret in hard X-ray, where one 0.5 ks snapshot
carries tens of counts and overrides any of the priors.

**Cost.** Per round, GIG / gamma / mixture: 897 / 2 / 22 ms (bulk), 137 / 4 / 8 ms
(gamma-ray), 344 / 41 / 20 ms (hard X-ray). The GIG's exact EIG sums the Sichel predictive
over a support sized to leave 1e-9 of the mass outside; a prior fitted to 20 members can be
very heavy-tailed (in bulk, facies f3 fits order -1.54 at concentration 0.02, whose predictive
at 40 m^3 needs about two million terms), and that is where the time goes.

**Allocation rules** (under the GIG): D-optimality and DAD predict within 0.007 nats of EIG
in all three settings but find the top contexts later (gamma-ray top-5 regret area 0.18
against 0.054); the systematic programme predicts within 0.03 nats but its regret area is 3
to 14 times ours; Bayes-LUCB has the smallest top-5 regret area in bulk sampling (0.105 and
0.103 against 0.123) but not in the telescope settings, and predicts worse everywhere (by
0.07 to 1.6 nats); Thompson sampling and random designs are worst on both.

**The log-skew-normal** (appendix; the hard X-ray runs were still in progress when this was
written, 2026-10-03 22:30). With 20-member catalogues its maximum-likelihood shape diverges
for 9 of the 15 classes (2 of 4 facies, 4 of 5 gamma-ray classes, 3 of 6 hard X-ray classes),
leaving 2 to 14 per cent of each such class below a hard edge; across 220 random 20-member
draws from the two telescope catalogues it diverged in 47 per cent (`prior_fit.py`). Even so:

| setting | GIG | quadrature | MCMC, M = 1024 | MCMC, M = 8192 |
|---|---|---|---|---|
| bulk NLPD | 2.917 | +0.000 (n.s.) | +0.007 (n.s.) | +0.004 (n.s.) |
| gamma-ray NLPD | 5.412 | -0.010 (t -3.4; Holm p_t 0.03, p_W 0.05) | -0.009 (n.s.) | -0.009 (n.s.) |
| bulk ms/round | 897 | 1507 | 1417 | 11965 |
| gamma-ray ms/round | 136 | 858 | 1013 | 8415 |

Regret does not separate any of them from the GIG. The nested estimator is biased up at the
prior (gate): 2.6-3.1 nats against an exact 1.1-1.2 in bulk at 40 m^3, 13.4 against 2.2 for a
gamma-ray class at 25 Ms, 6.7 against 2.9 in hard X-ray at 25 ks.

## Gate

Before a setting runs: GIG, gamma and gamma-mixture EIGs against bias-corrected nested MC in
their own families (fail only if |z| > 4 and the gap exceeds 0.02 nats); the exact
log-skew-normal EIG against a fine grid (fail above 0.01 nats); the MCMC posterior mean after
one observation against the grid (|z| < 6). The MCMC baseline's nested EIG is compared with
the grid too and recorded, never failed: its bias is the baseline's. The grid reference is
summed in blocks of counts (2026-10-03), since the heavy-tailed priors of a 20-member
catalogue give it a support of 4e4 counts on hard X-ray, too long for one table. All three
settings pass (`results/gate_*.csv`).

## Files

| file | contents |
|---|---|
| `results/{bulk,gamma,hardxray}.csv` | per agent, episode, checkpoint |
| `results/summary_*.csv` | final budget: mean, sem, paired t vs `eig` (positive = better than `eig`), ms per round/act/observe |
| `results/pop40_*`, `popall_*`, `pop200_*` | the catalogue-size runs, three models |
| `results/gate_*.csv` | the gate |
| `results/significance.{csv,txt}` | every pair, every metric, Holm-corrected |
| `results/prior_fit.{csv,txt}` | the priors alone, by catalogue size |
| `results/partial/` | one file per (episode, agent) run; rerunning skips what is there |
| `results/*.log` | the run logs; `pipeline.log` has the stages and times |
