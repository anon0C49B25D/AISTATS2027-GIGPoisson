"""Expected information gain about the grade. The method this paper proposes.

``EIG(u) = H[y | u] - A(u)``, both terms closed form under the Sichel predictive: the first
a sum over the truncated support, the second a one-dimensional quadrature over the mixing
law. Nothing is estimated, so the value carries no Monte-Carlo error and no sample size to
tune.

This is the parameter-oriented criterion, the quantity BALD scores in the active learning
literature \\citep{houlsby2011bald}. Its prediction-oriented counterpart is
:mod:`agents.epig`.
"""

from .base import ScoredAgent

__all__ = ["EIG"]


class EIG(ScoredAgent):
    criterion = "eig"
    criterion_label = "EIG (ours)"
    closed_form = True

    def score(self, model, belief, f, env):
        return model.eig(belief, f)
