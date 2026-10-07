"""An amortised, non-myopic design policy in the manner of deep adaptive design.

Deep adaptive design \\citep{foster2021deep} replaces the per-step optimisation of an
information criterion with a policy trained once, offline, to maximise a lower bound on the
information gathered over the whole experiment, and then deployed at the cost of a forward
pass. This is that idea, kept light enough to need no autodiff framework, close to the
policy-gradient form of \\citet{blau2022optimizing}:

**Policy.** A linear softmax over every (context, action) pair, ``pi(u | h) ~ exp(w . phi(u, h))``,
on features of the agent's GIG-Poisson belief at the context (posterior mean, dispersion,
counts and exposure so far), of its prior, of the action (cost, expected count, a
D-optimality-like score per unit cost), and of the budget left, which interacts with the
action features so that the policy can plan rather than act greedily.

**Objective.** The sequential prior contrastive estimate (sPCE) of the information gathered
by the end of the budget, a lower bound on the total information gain. The prior factorises
over contexts, so the bound is taken per context and summed:

    sPCE = sum_k log p(D_k | lam_k0) / ((1/(L+1)) sum_{l=0}^{L} p(D_k | lam_kl)),

with ``lam_k0`` the rate that generated the data and ``lam_kl`` independent prior draws.
A single bound over all contexts would saturate at ``log(L + 1)`` nats, far below the
hundreds a campaign over 192 targets gathers; per context it saturates at ``log(L + 1)``
each, well above what one context gathers here. ``p(D_k | lam)`` depends on the history only
through the accumulated count and exposure, so the bound is cheap.

**Training.** REINFORCE with a running-mean baseline and normalised advantages, Adam on
``w``, on experiments simulated from the agent's own model: the episode's contexts and their
fitted priors as the environment supplies them, rates drawn from those priors, counts from
the Poisson. Training seeds are disjoint from the evaluation episodes. The trained weights
are stored per setting and loaded at reset; training cost is reported separately, since it
is paid once, which is the point of amortisation.

**At deployment** the policy acts greedily, and predicts with the GIG-Poisson posterior.
"""

import json
import os

import numpy as np
from scipy.special import logsumexp

from methods.gigpoisson import gig_moments, gig_sample

from .base import Agent

__all__ = ["AmortisedDesign", "env_key", "train_policy", "WEIGHTS_DIR"]

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments",
                           "comparison", "dad")
N_FEATURES = 13


def env_key(env):
    """A name for the setting, from what defines its decision problem."""
    parts = [type(env).__name__, str(env.n_contexts), "%g" % env.budget]
    for attr in ("eval_volume", "eval_time", "eval_dwell", "counts_per_ks"):
        if hasattr(env, attr):
            parts.append("%s%g" % (attr, getattr(env, attr)))
    if hasattr(env, "population"):
        parts.append("pop%d" % sum(len(v) for v in env.population.values()))
    return "_".join(parts)


class _State(object):
    """Per-context sufficient statistics, beliefs and moments, kept in step with observations."""

    def __init__(self, model, env, beliefs, prior_moments, recovery):
        self.model, self.env = model, env
        self.beliefs = list(beliefs)
        self.recovery = recovery
        k = env.n_contexts
        self.T, self.F = np.zeros(k), np.zeros(k)
        mom = np.array([gig_moments(b) for b in self.beliefs])
        self.m0, self.v0 = mom[:, 0].copy(), mom[:, 1].copy()
        self.m, self.v = mom[:, 0].copy(), mom[:, 1].copy()
        acts = np.asarray(env.action_values, dtype=float)
        self.acts = acts
        self.cost = np.array([env.cost((0, a)) for a in acts])
        self.expo = np.array([[env.exposure((kk, a), recovery) for a in acts]
                              for kk in range(k)])
        self.spent = 0.0

    def observe(self, k, y, f, c):
        self.beliefs[k] = self.model.update(self.beliefs[k], y, f)
        self.m[k], self.v[k] = gig_moments(self.beliefs[k])
        self.T[k] += y
        self.F[k] += f
        self.spent += c

    def features(self):
        """``(K, A, N_FEATURES)``: one feature vector per context and action."""
        left = max(0.0, 1.0 - self.spent / self.env.budget)
        e_count = self.expo * self.m[:, None]
        fano = np.log1p(self.expo * (self.v / self.m)[:, None])
        lc = np.log(self.cost)[None, :]
        per_cost = fano - lc
        K, A = self.expo.shape
        ctx = np.stack([np.log(self.m), np.log(self.v / self.m ** 2), np.log1p(self.T),
                        np.log1p(self.F * self.m0), np.log(self.m0),
                        np.log(self.v0 / self.m0 ** 2), (self.F == 0).astype(float)], axis=1)
        out = np.empty((K, A, N_FEATURES))
        out[..., 0] = lc
        out[..., 1] = np.log(e_count)
        out[..., 2] = fano
        out[..., 3] = per_cost
        out[..., 4:11] = ctx[:, None, :]
        out[..., 11] = left * per_cost
        out[..., 12] = left * lc
        return out


def _scores(w, phi):
    return phi @ w


def simulate(w, env, model, beliefs, prior_moments, recovery, rng, L=1023, greedy=False):
    """One simulated experiment under the policy. Returns ``(sPCE, sum of grad log pi)``."""
    st = _State(model, env, beliefs, prior_moments, recovery)
    K, A = st.expo.shape
    lam = np.array([float(gig_sample(b, 1, rng)[0]) for b in beliefs])
    grad = np.zeros(N_FEATURES)
    smallest = int(np.argmin(st.cost))
    while True:
        remaining = env.budget - st.spent
        if st.cost[smallest] > remaining + 1e-9:
            break
        phi = st.features().reshape(K * A, N_FEATURES)
        s = _scores(w, phi)
        s -= s.max()
        p = np.exp(s)
        p /= p.sum()
        i = int(np.argmax(p)) if greedy else int(rng.choice(p.size, p=p))
        grad += phi[i] - p @ phi
        k, j = divmod(i, A)
        if st.cost[j] > remaining + 1e-9:
            j = smallest
        f = st.expo[k, j]
        y = float(rng.poisson(f * lam[k]))
        st.observe(k, y, f, st.cost[j])
    # Per-context sPCE: only observed contexts contribute.
    seen = np.flatnonzero(st.F > 0)
    if seen.size == 0:
        return 0.0, grad
    contrast = np.stack([gig_sample(beliefs[k], L, rng) for k in seen])        # (S, L)
    lam_all = np.concatenate([lam[seen][:, None], contrast], axis=1)              # (S, L+1)
    ll = st.T[seen][:, None] * np.log(lam_all) - st.F[seen][:, None] * lam_all
    spce = float(np.sum(ll[:, 0] - (logsumexp(ll, axis=1) - np.log(L + 1))))
    return spce, grad


def train_policy(episode, n_iter=150, batch=16, lr=0.05, n_jobs=12, log=print,
                 validate=None, every=25):
    """REINFORCE on sPCE. ``episode(w, i)`` simulates training episode ``i`` under weights
    ``w`` and returns ``(sPCE, sum of grad log pi)``; it must be picklable, since episodes
    run in parallel worker processes.

    REINFORCE improves the *stochastic* policy it samples from, and the deployed policy is
    the greedy one, which need not improve with it. ``validate(w, pool)``, if given, scores
    the greedy policy on a fixed set of simulated episodes; it is called at the start and
    every ``every`` iterations, and the best weights seen are returned. The start is
    D-optimality per unit cost, so the returned policy is never worse than that on the
    agent's own simulator. Returns ``(w, curve, validation)``."""
    from joblib import Parallel, delayed
    w = np.zeros(N_FEATURES)
    w[3] = 1.0                       # start at D-optimality per unit cost
    m_adam, v_adam = np.zeros_like(w), np.zeros_like(w)
    base, curve, checks = None, [], []
    with Parallel(n_jobs=n_jobs) as pool:
        best_w, best_v = w.copy(), (validate(w, pool) if validate else None)
        if validate:
            checks.append((0, best_v))
            log("  start   greedy validation sPCE %.2f" % best_v)
        for it in range(n_iter):
            out = pool(delayed(episode)(w.copy(), it * batch + b) for b in range(batch))
            R = np.array([o[0] for o in out])
            G = np.array([o[1] for o in out])
            base = R.mean() if base is None else 0.9 * base + 0.1 * R.mean()
            adv = (R - base) / (R.std() + 1e-8)
            g = (adv[:, None] * G).mean(axis=0)
            m_adam = 0.9 * m_adam + 0.1 * g
            v_adam = 0.999 * v_adam + 0.001 * g * g
            mh, vh = m_adam / (1 - 0.9 ** (it + 1)), v_adam / (1 - 0.999 ** (it + 1))
            w = w + lr * mh / (np.sqrt(vh) + 1e-8)
            curve.append(float(R.mean()))
            if it % 10 == 0 or it == n_iter - 1:
                log("  iter %3d  mean sPCE %.2f  |w| %.2f" % (it, R.mean(), np.linalg.norm(w)))
            if validate and ((it + 1) % every == 0 or it == n_iter - 1):
                v = validate(w, pool)
                checks.append((it + 1, v))
                log("  iter %3d  greedy validation sPCE %.2f (best %.2f)" % (it + 1, v, best_v))
                if v > best_v:
                    best_w, best_v = w.copy(), v
    if not validate:
        best_w = w
    return best_w, curve, checks


class AmortisedDesign(Agent):
    criterion = "dad"
    criterion_label = "amortised policy (DAD-style)"
    closed_form = True

    def reset(self, env, prior_moments, recovery, rng=None):
        super(AmortisedDesign, self).reset(env, prior_moments, recovery, rng=rng)
        path = os.path.join(WEIGHTS_DIR, env_key(env) + ".json")
        if not os.path.exists(path):
            raise FileNotFoundError("no trained policy for this setting: run "
                                    "experiments/comparison/train_dad.py first ({})".format(path))
        with open(path) as fh:
            self.w = np.asarray(json.load(fh)["w"], dtype=float)
        self._state = _State(self.model, env, self.beliefs, prior_moments, recovery)

    def observe(self, k, y, f):
        super(AmortisedDesign, self).observe(k, y, f)
        st = self._state
        st.beliefs[k] = self.beliefs[k]
        st.m[k], st.v[k] = gig_moments(self.beliefs[k])
        st.T[k] += y
        st.F[k] += f
        # The harness charges the cost; keep the policy's own budget clock in step with it.
        st.spent += self._last_cost

    def act(self, env, rng):
        st = self._state
        K, A = st.expo.shape
        s = _scores(self.w, st.features().reshape(K * A, N_FEATURES))
        k, j = divmod(int(np.argmax(s)), A)
        self._last_cost = float(st.cost[j])
        return k, float(st.acts[j])
