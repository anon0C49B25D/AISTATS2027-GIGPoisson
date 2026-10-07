# Gamma-ray source photometry: allocating telescope time

Written 2026-08-09. `python fetch.py` then `python fit.py` then `python run.py`.

## Why this setting

The bulk-sampling study generates its rates from the GIG-Poisson law, so a model that assumes
that law is correct by construction and its advantage there is discountable. This study
removes the circularity: **the rates are measured, not simulated.**

The Fermi-LAT fourth source catalogue lists, for each detected gamma-ray source, an integral
photon flux above 1 GeV in photons per square centimetre per second. That is a rate, and the
instrument that measured it is a photon counter, so the counting model is not an analogy
here. It is the measurement process.

Reference: Abdollahi et al., *Fermi Large Area Telescope Fourth Source Catalog*, ApJS 247, 33
(2020). Served by VizieR as `J/ApJS/247/33/4FGL`, fetched over the public TSV interface and
cached in `data/`, so the study reruns offline.

## The observing model

```
lam_k                          true photon flux of source k, from the catalogue
u = (k, t)                     which source, and how long to integrate
f(u) = A_eff * t               exposure, in cm^2 s
y | lam_k, u  ~  Poisson(f(u) lam_k)
cost(u) = t                    budget spent, in megaseconds of telescope time
```

A pointed observation looks at one source at a time. `A_eff` is of order 8000 cm^2, the
Fermi-LAT effective area above 1 GeV. Integration times are `{0.5, 1, 2.5, 5, 10, 25}` Ms
against a budget of 120 Ms. There is one instrument constant and one action variable, which
is Assumption 1 exactly.

**48 targets share that budget, so most cannot be observed at all.** This is deliberate and
it took three attempts to get right. With 12 targets on the same budget every source ends up
well measured whatever the schedule does, the prior differences wash out, and round-robin is
near-optimal by construction: no acquisition rule can distinguish itself, and none did.
Scarcity is the regime adaptive allocation exists for and the one an observing proposal is
written in.

**Priors are conditioned on source class.** A gamma-ray observer knows what kind of object a
target is before pointing, and the classes are not alike: pulsars have a median flux twelve
times that of blazar candidates and a variance-to-mean ratio twelve hundred times larger.
Each source gets its class's prior, fitted to a held-out half within that class. Giving every
source one population prior makes the targets exchangeable and removes the problem.

## Does the third parameter earn its place on real rates?

`fit.py` fits each mixing law to the 5065 catalogue fluxes by maximum likelihood and scores
it by AIC, so the extra parameter has to pay its penalty.

| mixing law | k | log-likelihood | AIC | dAIC |
|---|---|---|---|---|
| gamma | 2 | -18485.3 | 36974.7 | **+5959** |
| lognormal | 2 | -15998.4 | 32000.9 | **+986** |
| generalised inverse Gaussian | 3 | -15504.6 | 31015.3 | 0 |

Fitted concentration `omega = 0.026`, deep in the heavy-tailed regime and far from the gamma
boundary, and order `p = -1.115`, which is outside the gamma's range entirely.

The result survives restricting to the 5th-95th percentile band the study actually uses:
dAIC +1773 against the gamma and +555 against the lognormal, so it is not an artefact of a
handful of extreme sources.

## Fairness

**No leakage.** The catalogue is split in half *within each class*. One half is the
population an agent fits its class priors to; the other supplies the sources an episode
observes. A model knows what pulsars look like in general and nothing about the pulsar in
front of it.

**Each family gets its own maximum-likelihood fit** to that population rather than two matched
moments. Where a population of comparable rates has been seen, that is both fairer and more
realistic: each model is allowed the best description its own parameters can express.

**Photons are shared exactly.** Each source's photons form a realised Poisson process in
exposure, and an observation consumes the next `f` of that stream, so two agents that spend
the same time on the same source record the same photons.

## What is scored

- **NLPD**: average surprisal of held-out photon counts at a reference integration time,
  under the agent's own predictive.
- **regret**: cumulative rate gap weighted by integration time,
  `sum_t t_t (lam_{k*} - lam_{k_t})`, where `k_t` is the source with the highest posterior
  mode after observation `t`. In `1e-10 cm^-2 s^-1 Ms`. Ties at a zero mode go to the
  higher posterior mean.
- **inference cost**: wall-clock seconds in `act` plus `observe`. `inference_seconds` is
  cumulative; divide by `rounds`.
- **flux error in dex**, the root-mean-square error of `log10` of the posterior mean flux,
  kept as a diagnostic and not reported in the paper's tables.

## Simplifications, stated rather than hidden

**The extreme 5 per cent tails are excluded.** The faint end sits at the detection threshold
where a flux-limited catalogue is incomplete, and a handful of exceptionally bright sources
would consume any budget by themselves.

**There is no diffuse background.** A real observation counts source photons plus a known
background. That is an *additive* term, so it falls outside Assumption 1 and would need a
different conjugacy argument. This is a genuine limitation of the framework and belongs in
the discussion, not in a footnote.

**Catalogue fluxes are estimates, not exact rates**, and the catalogue is the detected
population rather than the intrinsic one. Neither affects a comparison between mixing laws on
the same numbers, but both would matter to any claim about the true luminosity function.

## Files

| file | what it is |
|---|---|
| `fetch.py` | downloads and caches the 4FGL fluxes |
| `fit.py` | maximum-likelihood fit and AIC of each mixing law |
| `environment.py` | the observing problem: truth, actions, budget, photons |
| `run.py` | gate and sequential study; writes `data/` and `results/` |
| `results/mixing_law_fits.csv` | the AIC comparison above |
| `results/gate_eig.csv` | each model's EIG against nested Monte Carlo |
| `results/sequential.csv` | per agent, programme and checkpoint |
| `results/allocation.csv` | time each agent gave each source, against its true flux |

Agents come from `agents/` and models from `methods/countmodels.py`, unchanged from the
bulk-sampling study: the two environments share a protocol (`n_contexts`, `action_values`,
`exposure`, `cost`, `target_exposure`), so the same agents run on both without modification.

`run.py` writes `results/sequential.csv` after every programme, so an interrupted run still
leaves usable results.
