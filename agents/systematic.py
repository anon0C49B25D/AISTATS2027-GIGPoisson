"""The systematic programme: every block in turn, at one fixed sample volume.

The incumbent. A grid or round-robin survey at a constant sample size is what a sampling
campaign actually runs when nothing adaptive is in place, and any adaptive scheme has to
beat it to be worth the trouble. It is a stronger baseline than it looks, because spreading
effort evenly is close to optimal whenever the quantity of interest is a coarse ranking.
"""

from .base import Agent

__all__ = ["Systematic", "BudgetedSystematic"]


class Systematic(Agent):
    criterion = "uniform"
    criterion_label = "systematic"
    closed_form = True

    def reset(self, env, prior_moments, recovery, rng=None):
        super(Systematic, self).reset(env, prior_moments, recovery, rng=rng)
        self._next = 0

    def act(self, env, rng):
        k = self._next % env.n_contexts
        self._next = k + 1
        return k, float(env.action_values[len(env.action_values) // 2])


class BudgetedSystematic(Systematic):
    """Round-robin at the largest action that still lets the budget visit every context once.

    The fixed median action of :class:`Systematic` is the right incumbent when the budget
    covers several passes. Under a scarce budget it is not: it reaches a handful of contexts
    and leaves the rest unobserved, which no survey planner would schedule. A planner sizes
    the action to the area and the time available, and so does this: the largest action
    whose cost, times the number of contexts, fits the budget, or the smallest action if
    none does. It is the strongest honest version of spreading the effort evenly.
    """

    criterion = "systematic"
    criterion_label = "systematic (budget-matched)"

    def reset(self, env, prior_moments, recovery, rng=None):
        super(BudgetedSystematic, self).reset(env, prior_moments, recovery, rng=rng)
        grid = sorted(float(v) for v in env.action_values)
        fits = [v for v in grid if env.cost((0, v)) * env.n_contexts <= env.budget + 1e-9]
        self._value = fits[-1] if fits else grid[0]

    def act(self, env, rng):
        k = self._next % env.n_contexts
        self._next = k + 1
        return k, self._value
