"""Sensing agents: pairs of a count model and an acquisition criterion.

The registry below spans two axes deliberately, because they answer different questions.

**Does the acquisition matter?** Hold the model at GIG-Poisson and vary the criterion. That
is the first group, and it is where maximum entropy sampling, D-optimality, Thompson
sampling and the rest appear.

**Does the model matter?** Hold the criterion at expected information gain and vary the
mixing law. That is the second group. Every model is initialised from the same prior mean
and variance per block, so a difference between them is a difference in what the family can
express and not in what it was told.

Considered and not included: adaptive cluster sampling (Thompson, 1990), the standard design
for rare and clustered populations, which expands a neighbourhood around any unit exceeding
a threshold. It needs a spatial adjacency between sampling units to expand into, and the
blocks here are independent under the prior, so the rule has nothing to act on. It becomes
relevant once the context set is given spatial structure, which is the extension named in
the discussion.
"""

from methods.countmodels import (GammaPoisson, GIGPoisson, LogNormalLikelihood,
                                 LognormalPoisson)

from .base import Agent, ScoredAgent          # noqa: F401
from .bo import BayesOptEI, BayesOptUCB
from .dopt import DOptimality
from .dad import AmortisedDesign
from .eig import EIG
from .lucb import BayesLUCB
from .epig import EPIG
from .maxent import MaxEntropy
from .neyman import Neyman
from .systematic import BudgetedSystematic, Systematic
from .thompson import Thompson
from .uniform_random import RandomAgent
from .variance import EpistemicVariance, TotalVariance

#: Criteria compared under the proposed model.
CRITERIA = (EIG, EPIG, MaxEntropy, DOptimality, Neyman, TotalVariance,
            EpistemicVariance, Thompson, Systematic, RandomAgent)

#: Mixing laws compared under the proposed criterion.
ALTERNATIVE_MODELS = (GammaPoisson, LognormalPoisson)

#: A lognormal *likelihood* on the counts rather than a mixing law over the rate: the other
#: standard remedy for overdispersion, and the one practitioners reach for most often. Kept
#: out of :data:`ALTERNATIVE_MODELS` so that every study already in the repository writes
#: exactly what it wrote before. Opt in with ``build_all(include_extra_models=True)``.
EXTRA_ALTERNATIVE_MODELS = (LogNormalLikelihood,)


#: Criteria carried across every model. EPIG is excluded from the non-conjugate model: its
#: outer expectation needs a posterior predictive entropy at every node, and under a grid
#: posterior that is a quadrature inside a quadrature, which costs more than the whole rest
#: of the study put together. That exclusion is itself a result and is reported, not hidden.
CROSSED = (EIG, MaxEntropy, DOptimality, Neyman, TotalVariance, EpistemicVariance,
           Thompson)

#: Bayesian optimisation on a Gaussian process with a Poisson output. Kept out of the
#: default set on purpose: it is the only member that pools across contexts, it carries its
#: own surrogate rather than one of the mixing laws, and adding it silently would change
#: what every existing study writes. Opt in with ``build_all(include_bo=True)``.
BAYESOPT = (BayesOptEI, BayesOptUCB)


def build_all(cross="models", include_bo=False, include_extra_models=False):
    """Every agent in the study.

    ``cross`` selects how much of the criterion-by-model grid to run:

        ``False``     the ten criteria under the proposed model only.
        ``"models"``  those, plus the proposed criterion under each alternative mixing law.
                      This is the headline set: one axis at a time, twelve agents.
        ``True``      the full grid, every criterion in :data:`CROSSED` under every model.

    The belief-free designs (systematic, random) are run once whatever the setting, since
    they never consult a model.

    ``include_bo`` appends :data:`BAYESOPT`. It defaults off so that every study already in
    the repository writes exactly what it wrote before this was added.
    """
    out = [cls() for cls in CRITERIA]
    if cross == "models":
        out += [EIG(model=m()) for m in ALTERNATIVE_MODELS]
    elif cross is True:
        for m in ALTERNATIVE_MODELS:
            out += [cls(model=m()) for cls in CROSSED]
            out += [EPIG(model=m())] if m is not LognormalPoisson else []
    if include_extra_models:
        out += [EIG(model=m()) for m in EXTRA_ALTERNATIVE_MODELS]
    if include_bo:
        out += [cls() for cls in BAYESOPT]
    return out


NAMES = tuple(a.name for a in build_all())
LABELS = {a.name: a.label for a in build_all()}


def build(name):
    for a in build_all(include_bo=True, include_extra_models=True):
        if a.name == name:
            return a
    raise ValueError("unknown agent {!r}; known: {}".format(name, ", ".join(NAMES)))


#: Outer and inner sample sizes (N = M) of the nested estimators under the GIG prior, per
#: setting, chosen so that a round costs about what a round of the exact EIG costs, both
#: timed in one process on episode 0 (2026-10-04: exact 131 / 23 / 54 ms per round; PCE at
#: N = M = 1024 254 / 182 / 182 ms; cost scales as N M). Other settings fall back to 1024.
NMC_SIZES = {"bulk": 768, "gamma": 384, "hardxray": 576}


def build_comparison(setting=None):
    """Every agent of the head-to-head study on the three settings.

    The proposed criterion under two conjugate mixing laws, the non-conjugate route
    (log-skew-normal prior, MCMC posterior, nested Monte Carlo EIG), and Bayesian
    optimisation by expected improvement on a GP-Poisson surrogate under two kernels.
    The binned lognormal likelihood was dropped after the first round of studies: it cost
    two orders of magnitude more per round than any other agent and was never competitive.

    Added 2026-10-04 for the factorial of prior by allocation rule: D-optimality and the
    budget-matched systematic programme under the gamma and gamma-mixture priors; and the
    GIG prior with its EIG estimated by nested Monte Carlo, plain and as the PCE bound, at
    the sample sizes of :data:`NMC_SIZES` for ``setting``.
    """
    from methods.compoundgamma import CompoundGammaPoisson
    from methods.countmodels import GIGPoissonNMC
    from methods.gppoisson import GPPoisson
    from methods.gammamixture import GammaMixturePoisson
    from methods.logskewnormal import LogSkewNormalPoisson, LogSkewNormalQuad
    n_nmc = NMC_SIZES.get(setting, 1024)
    return [EIG(),
            EIG(model=GammaPoisson()),
            EIG(model=LogSkewNormalPoisson()),
            BayesOptEI(),
            BayesOptEI(model=GPPoisson(kernel="categorical")),
            # Non-adaptive allocation, predicting with the proposed model.
            BudgetedSystematic(),
            RandomAgent(),
            # Other acquisitions under the proposed model, and the top-m rule under both
            # conjugate models, which is where the mixing law can reach the regret.
            DOptimality(),
            Thompson(),
            BayesLUCB(),
            BayesLUCB(model=GammaPoisson()),
            # A flexible conjugate rival, the exact non-conjugate route, and the sampled
            # route with an inner sample eight times larger.
            EIG(model=GammaMixturePoisson()),
            EIG(model=LogSkewNormalQuad()),
            EIG(model=LogSkewNormalPoisson(n_chains=1024 + 8192, n_outer=1024,
                                           name="logskewnormal-poisson-m8k",
                                           label="log-skew-normal Poisson (MCMC + NMC, M = 8192)")),
            # An amortised, non-myopic policy trained offline on the agent's own model.
            AmortisedDesign(),
            # The rest of the prior-by-rule factorial: the two conjugate rivals under
            # D-optimality and under the systematic programme.
            DOptimality(model=GammaPoisson()),
            DOptimality(model=GammaMixturePoisson()),
            BudgetedSystematic(model=GammaPoisson()),
            BudgetedSystematic(model=GammaMixturePoisson()),
            # The estimator ablation: the GIG prior and its exact posterior, with the EIG
            # estimated by nested Monte Carlo instead of computed.
            EIG(model=GIGPoissonNMC(n_nmc, n_nmc, pce=False)),
            EIG(model=GIGPoissonNMC(n_nmc, n_nmc, pce=True)),
            # In-regime rivals: the hierarchical gamma with its rate integrated out (a
            # compound gamma, with a power-law tail), and the skew-normal prior fitted by
            # maximum penalised likelihood so that its shape stays finite.
            EIG(model=CompoundGammaPoisson()),
            EIG(model=LogSkewNormalQuad(
                penalised=True, name="logskewnormal-quad-pen",
                label="log-skew-normal Poisson (exact quadrature, penalised fit)"))]
