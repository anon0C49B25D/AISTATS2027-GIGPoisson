"""Uniform over the action set: the floor any criterion must clear."""

from .base import Agent

__all__ = ["RandomAgent"]


class RandomAgent(Agent):
    criterion = "random"
    criterion_label = "random"
    closed_form = True

    def act(self, env, rng):
        k = int(rng.integers(env.n_contexts))
        return k, float(env.action_values[int(rng.integers(len(env.action_values)))])
