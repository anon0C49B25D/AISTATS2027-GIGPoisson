"""Uncertainty sampling, in its two forms.

``TotalVariance`` scores the whole predictive variance, which is the plain uncertainty
sampling analogue. ``EpistemicVariance`` scores only the part attributable to not knowing
the grade, which is the acquisition used in heteroscedastic active learning specifically to
avoid paying for irreducible noise \\citep{griffiths2019heteroscedastic}.

Both are moments rather than information, so both are cheap; the law of total variance gives
them in closed form from the mixing law's first two moments.
"""

from .base import ScoredAgent

__all__ = ["TotalVariance", "EpistemicVariance"]


class TotalVariance(ScoredAgent):
    criterion = "predictive-variance"
    criterion_label = "total var."
    closed_form = True

    def score(self, model, belief, f, env):
        return sum(model.variance_decomposition(belief, f))


class EpistemicVariance(ScoredAgent):
    criterion = "epistemic"
    criterion_label = "epistemic var."
    closed_form = True

    def score(self, model, belief, f, env):
        return model.variance_decomposition(belief, f)[1]
