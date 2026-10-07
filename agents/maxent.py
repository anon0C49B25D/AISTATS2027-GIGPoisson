"""Maximum entropy sampling: score a sample by how unpredictable its outcome is.

``H[y | u]``, the predictive entropy \\citep{sebastiani2000maximum}. It agrees with the
expected information gain exactly when the aleatoric term ``A(u)`` does not depend on the
action. In this model it does depend on the action, through the exposure, so the two
criteria differ and the difference is the whole of ``A(u)``.

The consequence is worth stating because it is not a subtlety: the entropy of a count grows
with its mean, so this criterion sends the budget to the block with the most stones, which
is the block a sample tells you least about per unit of gravel.
"""

from .base import ScoredAgent

__all__ = ["MaxEntropy"]


class MaxEntropy(ScoredAgent):
    criterion = "maxent"
    criterion_label = "max-entropy"
    closed_form = True

    def score(self, model, belief, f, env):
        return model.predictive_entropy(belief, f)
