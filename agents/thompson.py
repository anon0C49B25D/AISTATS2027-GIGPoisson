"""Posterior sampling: commit to the block that looks richest.

One grade is drawn per block and the largest draw wins, at the largest sample on the
schedule, since a policy seeking reward has no reason to hold back. This is reward-seeking
rather than information-seeking, so it is a poor estimator by construction. It is here
because it is what a prospector does, and because it makes the point that the axis is
different: concentrating on the best block is the right move for a bandit and the wrong one
for a survey.
"""

import numpy as np

from .base import Agent

__all__ = ["Thompson"]


class Thompson(Agent):
    criterion = "thompson"
    criterion_label = "Thompson"
    closed_form = True

    def act(self, env, rng):
        draws = [float(self.model.sample_rate(b, 1, rng)[0]) for b in self.beliefs]
        return int(np.argmax(draws)), float(env.action_values[-1])
