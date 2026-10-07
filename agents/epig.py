"""Expected predictive information gain: information about the next observation.

``EPIG(u) = H[y* | u*] - E_{y ~ p(y|u)}[ H[y* | u*, y] ]``, the information a sample at ``u``
carries about an observation ``y*`` at a target action ``u*`` \\citep{smith2023prediction}.
Where the expected information gain of :mod:`agents.eig` measures information about the
parameter, this measures it in the space of predictions.

Included because it is the baseline this paper most needs. The study is scored on held-out
predictive performance, and a criterion that targets the parameter is not automatically the
right one for that: what is most informative about a rate need not be most informative about
the next count. Omitting it would leave the obvious objection unanswered.

The target is the evaluation action at the same block, since beliefs factorise across blocks
and a sample at one block changes predictions nowhere else.

**This one has no closed form.** The outer expectation runs over the candidate's own
predictive, so each value costs a quadrature over ``y`` with a posterior predictive entropy
at every node. That is a nested computation of exactly the kind the closed forms elsewhere
avoid, and its cost is reported rather than hidden.
"""

import numpy as np

from .base import ScoredAgent

__all__ = ["EPIG"]


class EPIG(ScoredAgent):
    criterion = "epig"
    criterion_label = "EPIG"
    closed_form = False

    def __init__(self, model=None, n_nodes=12):
        super(EPIG, self).__init__(model)
        #: Support points of the candidate predictive used for the outer expectation. The
        #: highest-mass counts are kept and renormalised, which is where the estimator's
        #: error comes from and why it is not exact.
        self.n_nodes = int(n_nodes)

    def score(self, model, belief, f, env):
        target_f = env.target_exposure
        h_prior = model.predictive_entropy(belief, target_f)

        y, w = self._quadrature(model, belief, f)
        h_post = 0.0
        for yj, wj in zip(y, w):
            if wj <= 0.0:
                continue
            h_post += wj * model.predictive_entropy(model.update(belief, yj, f), target_f)
        return float(max(h_prior - h_post, 0.0))

    def _quadrature(self, model, belief, f):
        """The ``n_nodes`` most probable counts, renormalised.

        Truncating to the highest-mass points is what makes this criterion an approximation
        rather than an evaluation, and is where its error comes from.
        """
        y = model.support(belief, f)
        w = np.exp(model.logpmf(belief, f, y))
        if y.size > self.n_nodes:
            keep = np.argsort(w)[-self.n_nodes:]
            y, w = y[keep], w[keep]
        total = w.sum()
        return y, (w / total if total > 0 else w)
