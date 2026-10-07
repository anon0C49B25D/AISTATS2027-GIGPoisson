"""What a sensing agent is: a model, a criterion, and a belief it carries itself.

Agents are written against a small environment protocol rather than against any one problem,
so the same agent runs on the bulk-sampling and photon-counting studies unchanged:
``n_contexts``, ``action_values``, ``exposure(u, recovery)``, ``cost(u)`` and
``target_exposure``. A study supplies those and keeps its own domain names beside them.

An agent is a pair. The **model** says what a count is and how a belief about a rate moves
when one arrives. The **criterion** ranks candidate actions given that belief. Varying the
first with the second held fixed asks whether the modelling choice matters; varying the
second with the first held fixed asks whether the acquisition choice matters. They are
different questions and this package keeps them separable.

The belief belongs to the agent rather than to the environment, because two agents with
different models do not carry the same kind of object: a conjugate agent carries a few
numbers and a non-conjugate one carries a grid. The environment supplies the same prior
*information* to everyone, as a mean and a variance per block, and each agent expresses that
in its own family.

Three things are fixed here rather than left to the individual agent, because varying them
would make the comparison meaningless rather than merely different:

**Cost normalisation.** The budget is spent in cubic metres, so a criterion is ranked by its
value per cubic metre. Applied identically to every criterion, ours included.

**Cache invalidation.** Beliefs factorise across blocks, so an observation at one block
changes the score of that block alone.

**Prior information.** Every agent is initialised from the same two moments and none is told
the true parameters.
"""

__all__ = ["Agent", "ScoredAgent"]


class Agent(object):
    """Base class. Subclasses set ``criterion`` and implement :meth:`act`."""

    #: Short name of the acquisition rule.
    criterion = "agent"
    #: Human-readable name of the acquisition rule.
    criterion_label = "agent"
    #: Whether the acquisition value has a closed form under a conjugate model.
    closed_form = True

    def __init__(self, model=None):
        from methods.countmodels import GIGPoisson
        self.model = GIGPoisson() if model is None else model

    # -- identity ----------------------------------------------------------

    @property
    def name(self):
        """``criterion`` on its own for the default model, else ``criterion@model``."""
        if self.model.name == "gig-poisson":
            return self.criterion
        return "{}@{}".format(self.criterion, self.model.name)

    @property
    def label(self):
        if self.model.name == "gig-poisson":
            return self.criterion_label
        return "{} + {}".format(self.criterion_label, self.model.label)

    # -- belief ------------------------------------------------------------

    def reset(self, env, prior_moments, recovery, rng=None):
        """Express the shared prior information in this agent's own family.

        An environment that has a population of comparable rates to hand supplies
        ``initial_beliefs`` and each family is fitted to it; otherwise every family is
        matched to the same two moments. Either way the information is the same for
        everyone and no agent is told the truth.

        ``rng`` seeds a model that draws random numbers of its own, such as a sampler.
        """
        if rng is not None and hasattr(self.model, "set_rng"):
            self.model.set_rng(rng)
        if hasattr(env, "initial_beliefs"):
            self.beliefs = env.initial_beliefs(self.model)
        else:
            self.beliefs = [self.model.prior_from_moments(m, v) for m, v in prior_moments]
        self.recovery = recovery
        self._cache = {}

    def observe(self, k, y, f):
        self.beliefs[k] = self.model.update(self.beliefs[k], y, f)
        self._cache.pop(k, None)

    # -- scoring hooks used by the study -----------------------------------

    def rate_mean(self, k):
        return self.model.rate_mean(self.beliefs[k])

    def rate_map(self, k):
        return self.model.rate_map(self.beliefs[k])

    def logpmf(self, k, f, y):
        return self.model.logpmf(self.beliefs[k], f, y)

    # -- acting ------------------------------------------------------------

    def act(self, env, rng):
        raise NotImplementedError


class ScoredAgent(Agent):
    """An agent that ranks every candidate action by a scalar, per unit cost."""

    def score(self, model, belief, f, env):
        raise NotImplementedError

    def _block_scores(self, env, k):
        if k not in self._cache:
            self._cache[k] = [
                self.score(self.model, self.beliefs[k],
                           env.exposure((k, v), self.recovery), env)
                / env.cost((k, v))
                for v in env.action_values
            ]
        return self._cache[k]

    def act(self, env, rng):
        best = None
        for k in range(env.n_contexts):
            for j, s in enumerate(self._block_scores(env, k)):
                if best is None or s > best[0]:
                    best = (s, k, float(env.action_values[j]))
        return best[1], best[2]
