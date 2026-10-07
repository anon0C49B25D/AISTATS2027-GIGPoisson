"""Neyman optimal allocation, the classical survey-sampling answer to how much per stratum.

Neyman's rule allocates effort to a stratum in proportion to its standard deviation and in
inverse proportion to the square root of its unit cost. Specialised to this problem it has a
closed form. Proposition 2 gives the variance of the grade estimate at block ``k`` as
``lam_k / F_k`` once exposure has accumulated, so minimising the total

    sum_k lam_k / F_k     subject to     sum_k F_k / r_k = B

over the exposures gives ``F_k`` proportional to ``sqrt(lam_k r_k)``, and hence a target
share of the budget in cubic metres proportional to

    sqrt( lam_k / r_k ) .

More gravel goes to blocks that are richer, because a Poisson count is noisier there, and to
blocks the plant recovers well from, because gravel buys more exposure there.

This is the strongest non-information baseline in the study and the fairest one. It uses the
same model, the same prior and the same posterior as the proposed method; it simply spends
the budget to minimise a variance rather than to maximise an information gain. It is also
adaptive here, recomputing the target from the current posterior each round, which is more
than the textbook rule does and is deliberately generous.

The unknown grade is replaced by its posterior mean, which is the standard plug-in and the
standard weakness: the classical rule needs stratum variances it does not have.
"""

import numpy as np

from .base import Agent

__all__ = ["Neyman"]


class Neyman(Agent):
    criterion = "neyman"
    criterion_label = "Neyman alloc."
    closed_form = True

    def reset(self, env, prior_moments, recovery):
        super(Neyman, self).reset(env, prior_moments, recovery)
        self._spent = np.zeros(env.n_contexts)

    def act(self, env, rng):
        grades = np.array([self.model.rate_mean(b) for b in self.beliefs])
        share = np.sqrt(np.maximum(grades, 1e-12) / np.asarray(self.recovery))
        share = share / share.sum()
        # Whichever block is furthest behind its target share of what has been spent.
        target = share * (self._spent.sum() + float(min(env.action_values)))
        k = int(np.argmax(target - self._spent))
        v = float(min(env.action_values))
        self._spent[k] += v
        return k, v
