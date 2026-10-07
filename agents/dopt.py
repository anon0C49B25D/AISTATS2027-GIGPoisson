"""Local D-optimality on the Fisher information \\citep{chaloner1995bayesian}.

Reported in the variance-stabilising coordinate. D-optimality is not invariant under
reparameterisation, since the Fisher information transforms by the square of the Jacobian,
so quoting one arbitrary coordinate would attribute to the criterion what belongs to the
choice of coordinate. In the stabilising coordinate the information about the rate no longer
depends on the rate, and the criterion weights a block only by how much exposure it has
already received.
"""

from .base import ScoredAgent

__all__ = ["DOptimality"]


class DOptimality(ScoredAgent):
    criterion = "d-optimality"
    criterion_label = "D-optimality"
    closed_form = True

    def score(self, model, belief, f, env):
        return model.fisher_dopt(belief, f)
