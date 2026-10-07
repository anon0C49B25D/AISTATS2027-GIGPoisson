"""One sequential campaign under one agent, scored identically for every method.

Shared by every study that compares agents head to head, so that no method is timed or
scored by a different loop than another. Written against the environment protocol of
``agents.base`` (``n_contexts``, ``action_values``, ``exposure``, ``cost``) plus the three
draws every environment offers (``blocks``, ``stone_fields``, ``held_out``).

**Timing.** Wall-clock seconds of ``agent.act`` (design selection) and ``agent.observe``
(belief update), accumulated separately and nothing else: not the environment, not the
scoring, not the initial prior fit in ``reset``. Whatever an agent caches between rounds it
is entitled to, and the cost of maintaining the cache is inside one of the two calls.

**Scores at each checkpoint.**

    nlpd          average surprisal of held-out counts at the evaluation exposure, under
                  the agent's own predictive, in nats.
    regret_top{m} simple top-m regret, m = 1..5: ``1 - sum_{k in S} lam_k / sum_{k in S*} lam_k``,
                  where ``S`` is the agent's m contexts of highest posterior mean rate and
                  ``S*`` the m of highest true rate. Scale-free, in ``[0, 1]``. The
                  posterior mean is used because it is the one point estimate every model
                  here, sampled or Laplace or conjugate, provides without approximation.
    rate_rmse     root-mean-square error of the posterior mean rate (diagnostic).
    rate_dex      the same in ``log10`` (diagnostic).
    touched       fraction of contexts observed at least once.
    nlpd_obs      ``nlpd`` over the contexts observed at least once (NaN if none).
    nlpd_unobs    ``nlpd`` over the contexts never observed (NaN if none), which rests on
                  the prior alone. The two sets differ between agents, since each agent
                  decides which contexts it observes.
    mean_dwell    mean action value over the rounds so far, in the setting's own unit.
    frac_shortest fraction of the rounds so far spent at the smallest action.
"""

import time

import numpy as np

__all__ = ["TOP_M", "draw_episode", "prior_scores", "run_episode", "top_m_regret"]

TOP_M = (1, 2, 3, 4, 5)


def top_m_regret(means, truths, m):
    chosen = np.argsort(-np.asarray(means), kind="stable")[:m]
    best = np.sort(truths)[::-1][:m]
    return float(1.0 - truths[chosen].sum() / best.sum())


def draw_episode(env, e, n_heldout=24):
    """The ground truth of episode ``e``, identical for every agent."""
    rng = np.random.default_rng(1000 + e)
    moments, truths, recovery = env.blocks(rng)
    streams = env.stone_fields(truths, recovery, rng)
    heldout = env.held_out(truths, recovery, rng, n=n_heldout)
    ev = next(getattr(env, n) for n in ("eval_volume", "eval_time", "eval_dwell")
              if hasattr(env, n))
    f_eval = np.array([env.exposure((k, ev), recovery) for k in range(env.n_contexts)])
    return moments, truths, recovery, streams, heldout, f_eval


def _evaluate(agent, truths, heldout, f_eval, allocated):
    n = truths.size
    nlpd = np.array([-float(np.mean(agent.logpmf(k, f_eval[k], np.asarray(heldout[k], float))))
                     for k in range(n)])
    means = np.array([agent.rate_mean(k) for k in range(n)])
    rmse = float(np.sqrt(np.mean((means - truths) ** 2)))
    dex = float(np.sqrt(np.mean((np.log10(np.maximum(means, 1e-12))
                                 - np.log10(truths)) ** 2)))
    regrets = tuple(top_m_regret(means, truths, m) for m in TOP_M)
    seen = allocated > 0
    obs = float(nlpd[seen].mean()) if seen.any() else float("nan")
    unobs = float(nlpd[~seen].mean()) if (~seen).any() else float("nan")
    return ((float(np.mean(nlpd)),) + regrets + (rmse, dex, float(seen.mean()))
            + (obs, unobs))


def prior_scores(env, agent, episode, n_heldout=24):
    """The scores of :func:`run_episode` before any budget is spent: the prior alone.

    Same episode, same held-out counts, so a row here pairs with that episode's rows there.
    Returns ``(nlpd, regret_top1 .. regret_top5, rate_rmse, rate_dex)``.
    """
    moments, truths, recovery, _, heldout, f_eval = draw_episode(env, episode, n_heldout)
    agent.reset(env, moments, recovery, rng=np.random.default_rng(70_000 + episode))
    return _evaluate(agent, truths, heldout, f_eval, np.zeros(env.n_contexts))[:8]


def run_episode(env, agent, episode, checkpoints, n_heldout=24):
    """Spend the budget under one agent. One row per checkpoint.

    Row: ``(spent, rounds, act_seconds, observe_seconds, nlpd, regret_top1 .. regret_top5,
    rate_rmse, rate_dex, touched, nlpd_obs, nlpd_unobs, mean_dwell, frac_shortest)``.
    """
    moments, truths, recovery, streams, heldout, f_eval = draw_episode(env, episode, n_heldout)
    rng = np.random.default_rng(50_000 + episode)
    agent.reset(env, moments, recovery, rng=np.random.default_rng(70_000 + episode))
    n = env.n_contexts
    consumed, allocated = np.zeros(n), np.zeros(n)
    spent, rounds, t_act, t_obs = 0.0, 0, 0.0, 0.0
    smallest = float(min(env.action_values))
    dwell_sum, n_shortest = 0.0, 0

    def dwell():
        return (dwell_sum / rounds if rounds else float("nan"),
                n_shortest / rounds if rounds else float("nan"))

    rows, nxt = [], 0

    while nxt < len(checkpoints):
        t0 = time.perf_counter()
        k, v = agent.act(env, rng)
        t_act += time.perf_counter() - t0

        if spent + env.cost((k, v)) > checkpoints[-1] + 1e-9:
            v = smallest
            if spent + env.cost((k, v)) > checkpoints[-1] + 1e-9:
                break

        f = env.exposure((k, v), recovery)
        lo, hi = consumed[k], consumed[k] + f
        y = int(np.searchsorted(streams[k], hi) - np.searchsorted(streams[k], lo))
        consumed[k] = hi

        t0 = time.perf_counter()
        agent.observe(k, y, f)
        t_obs += time.perf_counter() - t0

        allocated[k] += env.cost((k, v))
        spent += env.cost((k, v))
        rounds += 1
        dwell_sum += float(v)
        n_shortest += int(abs(float(v) - smallest) < 1e-12)
        while nxt < len(checkpoints) and spent >= checkpoints[nxt] - 1e-9:
            rows.append((checkpoints[nxt], rounds, t_act, t_obs)
                        + _evaluate(agent, truths, heldout, f_eval, allocated) + dwell())
            nxt += 1

    # A schedule whose costs do not divide the budget stops a little short of the last
    # checkpoint; its final state is recorded there so no agent drops out of the average.
    while nxt < len(checkpoints):
        rows.append((checkpoints[nxt], rounds, t_act, t_obs)
                    + _evaluate(agent, truths, heldout, f_eval, allocated) + dwell())
        nxt += 1
    return rows
