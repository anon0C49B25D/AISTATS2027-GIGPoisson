"""Bayes-LUCB for the top-m contexts: an acquisition aimed at the regret, not at the rates.

Every other criterion here tries to learn every rate, and the decision the regret scores, the
set of the ``m`` largest rates, is read off the beliefs afterwards. This one targets that
set directly, following LUCB \\citep{kalyanakrishnan2012pac} with the confidence bounds of
Bayes-UCB \\citep{kaufmann2012bayesian}: posterior quantiles of the rate under the agent's
own model.

At each step the current top-``m`` set ``J`` is the ``m`` contexts of highest posterior mean.
The weakest member, ``l = argmin_{k in J} q_low(k)``, and the strongest challenger,
``u = argmax_{k not in J} q_high(k)``, are the two contexts whose order is least settled, and
both are measured, one per round. Bounds are the ``1 - 1/(t + 1)`` posterior quantiles, ``t``
the number of measurements so far, floored at 0.9 so the first rounds are not decided by
medians.

**Why it belongs in a comparison of models.** The bounds are tail quantiles, so a model's
tail decides which challengers look worth chasing. A heavy-tailed posterior keeps a lightly
observed context in contention longer than a gamma posterior with the same mean would. If the
mixing law reaches the regret anywhere, it reaches it through a rule like this one.

Each measurement takes the shortest action on the grid. The cost per unit of action is the
same for every action in every study here, so the shortest gives the finest adaptivity for
the same budget, and LUCB is defined in unit pulls.
"""

import numpy as np

from .base import Agent

__all__ = ["BayesLUCB"]


class BayesLUCB(Agent):
    criterion = "lucb"
    criterion_label = "Bayes-LUCB (top-m)"
    closed_form = True

    def __init__(self, model=None, m=5, floor=0.9):
        super(BayesLUCB, self).__init__(model)
        self.m = int(m)
        self.floor = float(floor)

    def reset(self, env, prior_moments, recovery, rng=None):
        super(BayesLUCB, self).reset(env, prior_moments, recovery, rng=rng)
        self._t = 0
        self._queue = []

    def observe(self, k, y, f):
        super(BayesLUCB, self).observe(k, y, f)
        self._t += 1

    def act(self, env, rng):
        shortest = float(min(env.action_values))
        if self._queue:
            return self._queue.pop(), shortest
        n = env.n_contexts
        q = max(self.floor, 1.0 - 1.0 / (self._t + 1.0))
        means = np.array([self.model.rate_mean(b) for b in self.beliefs])
        top = np.argsort(-means, kind="stable")[: min(self.m, n - 1)]
        rest = np.setdiff1d(np.arange(n), top)
        low = np.array([float(self.model.rate_quantile(self.beliefs[k], 1.0 - q)) for k in top])
        high = np.array([float(self.model.rate_quantile(self.beliefs[k], q)) for k in rest])
        weakest, challenger = int(top[np.argmin(low)]), int(rest[np.argmax(high)])
        self._queue.append(challenger)
        return weakest, shortest
