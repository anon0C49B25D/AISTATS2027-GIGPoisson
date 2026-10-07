# Actively sensing overdispersed count data with Sichel distributed predictions

Code, data and results for the paper of the same title, submitted for anonymous review.

A sensor chooses an action `u`. The action sets a known exposure `f(u)`, and the count it
returns is `y ~ Poisson(f(u) λ)` with an unknown rate `λ ~ GIG(α, a, b)`. The pair is
conjugate. An observation sends

```
GIG(α, a, b)  ->  GIG(α + y, a, b + 2 f(u))
```

so the exposure enters the second scale parameter additively and the posterior predictive is
the Sichel law. Every information quantity an acquisition function needs is available in
closed form, up to a truncated sum over the support.

## Layout

| path | contents |
|---|---|
| `methods/gigpoisson.py` | the core: GIG mixing law, Sichel predictive, conjugate update and Bessel numerics |
| `methods/countmodels.py` | GIG-, gamma- and other Poisson mixtures behind one interface |
| `methods/gammamixture.py`, `compoundgamma.py`, `logskewnormal.py`, `gppoisson.py` | the rival models of the comparison |
| `agents/` | one module per acquisition criterion, written against a small environment protocol |
| `experiments/` | the studies of the paper, each with its own `README.md`, `results/` and `figures/` |
| `experiments/tables/` | the results tables of the paper, written by `experiments/comparison/tables.py` |
| `figures/` | the figures of the main text, written by `experiments/comparison/paper_figures.py` and `experiments/common/natural_paper.py` |

An agent is a count model plus an acquisition criterion. `agents/__init__.py` crosses the two
axes. Holding the model fixed and varying the criterion asks whether the acquisition matters.
Holding the criterion fixed and varying the mixing law asks whether the model matters.

`experiments/README.md` lists the studies and what each one answers.

## Running

Tested with Python 3.14, numpy 2.4, scipy 1.17, matplotlib 3.10, pandas 3.0 and joblib 1.5.
The verification gate additionally uses mpmath.

```bash
pip install -r requirements.txt
```

The head-to-head comparison on the three sensing settings is run from the repository root:

```bash
python experiments/comparison/train_dad.py bulk       # likewise gamma, hardxray
python experiments/comparison/run.py all --reps 50    # the study
python experiments/comparison/significance.py         # paired tests, Holm-corrected
python experiments/comparison/tables.py               # tables, into experiments/tables/
python experiments/comparison/paper_figures.py        # figures, into figures/
```

The other studies are run from their own directory. Their READMEs give the commands. Every
study is reproducible from seed 0. The recorded results are included, so the tables and
figures can be regenerated without rerunning the studies. Per-episode checkpoints
(`results/partial/`) are left out, since their contents are merged into the results CSVs.

## Numerics

`K_v(z)` overflows double precision at moderate order, and the posterior order `α + Σy` grows
without bound as data arrive. Everything is therefore computed in log space. `log_besselk`
uses the exponentially scaled `kve` below order 45 and Olver's uniform asymptotic expansion
above it. Support sums use an upward recurrence on the order below 512 terms and the direct
route above it.

## License

MIT. See [LICENSE](LICENSE).
