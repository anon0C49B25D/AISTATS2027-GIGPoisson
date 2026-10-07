"""Sequential bulk sampling under a processing budget: two comparisons in one study.

The agents span two axes. Holding the model fixed and varying the acquisition asks whether
the criterion matters. Holding the acquisition fixed and varying the mixing law asks whether
the model matters, which is the question the paper is actually about. Both are run on the
same properties, the same recovery factors and the same gravel.

The gravel is shared exactly: each block's stone field is realised once as a Poisson process
in effective processed volume, and a sample consumes the next ``v r_k`` metres of it. Two
agents that process the same gravel recover the same stones.

Three things are scored, the metrics of the problem statement, plus one diagnostic:

    NLPD          average surprisal of held-out stone counts under the agent's own
                  predictive, in nats. An agent with the wrong model is penalised here more
                  directly than anywhere else.
    regret        cumulative rate gap, ``sum_t v_t (lam_{k*} - lam_{k_t})``: after every
                  sample the block with the highest posterior mode is the agent's pick, and
                  the grade it forgoes is charged for the gravel that sample processed.
                  Stones, not a fraction.
    inference     wall-clock seconds spent choosing a design and updating the belief on its
                  outcome, cumulative. Divide by ``rounds`` for the cost per design.
    grade RMSE    error of the posterior mean grade over blocks; a diagnostic only.

Writes to ``data/`` and ``results/``. Figures come from ``visualize.py``.

Run: python experiments/bulk-sampling/run.py
"""

import csv
import os
import sys
import time

import numpy as np
from scipy.special import gammaln, logsumexp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import agents                                                         # noqa: E402
from domain import BulkSampling                                       # noqa: E402
from methods.countmodels import MODELS                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0
N_ENV = 48              # properties
N_HELDOUT = 24          # held-out samples per block
CHECKPOINTS = (30.0, 60.0, 120.0, 180.0, 240.0)   # budget spent, m^3


def save_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("  wrote {} ({} rows)".format(os.path.relpath(path, HERE), len(rows)))


# --------------------------------------------------------------------------
# Correctness gate
# --------------------------------------------------------------------------

def _nested_mc_eig(model, belief, f, rng, n=4000):
    lam = model.sample_rate(belief, n, rng)
    y = rng.poisson(f * lam).astype(float)
    ll_out = -f * lam + y * np.log(f * lam) - gammaln(y + 1.0)
    lam_in = model.sample_rate(belief, n, rng)
    mu = f * lam_in
    ll_in = -mu[None, :] + y[:, None] * np.log(mu)[None, :] - gammaln(y + 1.0)[:, None]
    return float(np.mean(ll_out - (logsumexp(ll_in, axis=1) - np.log(n))))


def gate(env, verbose=True):
    """Every model's acquisition against nested Monte Carlo, before anything is run on it.

    A misspecified baseline is only a fair baseline if its own closed forms are right, so
    each model is checked inside its own family rather than against ours.
    """
    rng = np.random.default_rng(SEED)
    moments, _, recovery = env.blocks(rng)
    rows, bad = [], 0
    for cls in MODELS:
        model = cls()
        for k in (0, env.n_blocks // 2):
            belief = model.prior_from_moments(*moments[k])
            for v in (env.volumes[0], env.volumes[-1]):
                f = env.exposure((k, v), recovery)
                exact = model.eig(belief, f)
                lo = [_nested_mc_eig(model, belief, f, rng, 1500) for _ in range(4)]
                hi = [_nested_mc_eig(model, belief, f, rng, 6000) for _ in range(4)]
                corrected = (4.0 * np.mean(hi) - np.mean(lo)) / 3.0
                sem = float(np.sqrt((16.0 * np.var(hi, ddof=1) / 4
                                     + np.var(lo, ddof=1) / 4) / 9.0))
                z = (exact - corrected) / max(sem, 1e-12)
                rows.append((model.name, k, float(v), f, exact, corrected, sem, z))
                if abs(z) > 4.0 or exact < -1e-12:
                    bad += 1
    ok = bad == 0
    if verbose:
        print("Correctness gate: each model's EIG against nested Monte Carlo")
        print("  {:<20} {:>5} {:>7} {:>11} {:>11} {:>9} {:>7}".format(
            "model", "block", "volume", "eig exact", "bias-corr.", "sem", "z"))
        for name, k, v, f, e, c, s, z in rows:
            print("  {:<20} {:5d} {:7.3g} {:11.6f} {:11.6f} {:9.2e} {:+7.2f}"
                  .format(name, k, v, e, c, s, z))
        print("  ->  {} ({} of {} outside 4 sigma)".format(
            "PASS" if ok else "FAIL", bad, len(rows)))
    return ok, rows


# --------------------------------------------------------------------------
# One property, one agent
# --------------------------------------------------------------------------

def evaluate(env, agent, truths, recovery, heldout):
    """``(nlpd, grade_rmse)`` under the agent's own model."""
    nlpd, sq = [], []
    for k in range(env.n_blocks):
        f_eval = env.eval_volume * float(recovery[k])
        nlpd.append(-float(np.mean(agent.logpmf(k, f_eval,
                                                np.asarray(heldout[k], dtype=float)))))
        sq.append((agent.rate_mean(k) - truths[k]) ** 2)
    return float(np.mean(nlpd)), float(np.sqrt(np.mean(sq)))


def best_block(agent, modes):
    """The block with the highest posterior mode.

    A gamma posterior with shape below one has its mode at the origin, so an unsampled
    block can tie with another at zero. Ties go to the higher posterior mean rather than to
    the lower index, which would reward whatever order the blocks happen to be listed in.
    """
    tied = np.flatnonzero(modes == modes.max())
    if tied.size == 1:
        return int(tied[0])
    return int(tied[np.argmax([agent.rate_mean(k) for k in tied])])


def run_agent(env, agent, moments, truths, recovery, fields, heldout, rng,
              checkpoints_at=None):
    """Spend the budget under one agent, recording the score at each checkpoint.

    ``checkpoints_at`` overrides the module-level schedule, so a sweep that only needs the
    end of the campaign does not pay for the intermediate evaluations.
    """
    cps_at = CHECKPOINTS if checkpoints_at is None else tuple(checkpoints_at)
    agent.reset(env, moments, recovery)
    consumed = np.zeros(env.n_blocks)
    allocated = np.zeros(env.n_blocks)
    spent, rounds, infer_time, regret = 0.0, 0, 0.0, 0.0
    best = float(np.max(truths))
    modes = np.array([agent.rate_map(k) for k in range(env.n_blocks)])
    checkpoints, next_cp = [], 0

    while next_cp < len(cps_at):
        t0 = time.perf_counter()
        k, v = agent.act(env, rng)
        infer_time += time.perf_counter() - t0

        if spent + v > cps_at[-1]:
            v = float(min(env.volumes))
            if spent + v > cps_at[-1] + 1e-9:
                break

        f = env.exposure((k, v), recovery)
        lo, hi = consumed[k], consumed[k] + f
        y = int(np.searchsorted(fields[k], hi) - np.searchsorted(fields[k], lo))
        consumed[k] = hi
        t0 = time.perf_counter()
        agent.observe(k, y, f)
        infer_time += time.perf_counter() - t0
        allocated[k] += v
        spent += v
        rounds += 1

        # Only block k's belief moved, so only its mode needs recomputing.
        modes[k] = agent.rate_map(k)
        regret += v * (best - float(truths[best_block(agent, modes)]))

        while next_cp < len(cps_at) and spent >= cps_at[next_cp] - 1e-9:
            checkpoints.append((cps_at[next_cp], rounds, infer_time,
                                *evaluate(env, agent, truths, recovery, heldout), regret))
            next_cp += 1

    # An agent whose volumes do not divide the budget stops a little short of the last
    # checkpoint. Record its final state there anyway, so that every agent is compared at
    # the same budget rather than silently dropped from the average.
    while next_cp < len(cps_at):
        checkpoints.append((cps_at[next_cp], rounds, infer_time,
                            *evaluate(env, agent, truths, recovery, heldout), regret))
        next_cp += 1
    return checkpoints, allocated


# --------------------------------------------------------------------------

def main():
    dd, rd = os.path.join(HERE, "data"), os.path.join(HERE, "results")
    env = BulkSampling()

    ok, gate_rows = gate(env)
    save_csv(os.path.join(rd, "gate_eig.csv"),
             ["model", "block", "volume", "exposure", "eig_exact", "eig_corrected",
              "sem", "z"], gate_rows)
    if not ok:
        print("\nABORT: an acquisition failed its correctness gate.")
        return 1

    print("\nSequential study: {} properties, {} blocks, budget {:.0f} m^3, {} agents"
          .format(N_ENV, env.n_blocks, env.budget, len(agents.build_all(cross=True))))
    rows, env_rows, alloc_rows = [], [], []
    for e in range(N_ENV):
        rng_env = np.random.default_rng(1000 + e)
        moments, truths, recovery = env.blocks(rng_env)
        fields = env.stone_fields(truths, recovery, rng_env)
        heldout = env.held_out(truths, recovery, rng_env, n=N_HELDOUT)
        for k in range(env.n_blocks):
            env_rows.append((e, k, moments[k][0], moments[k][1], float(recovery[k]),
                             float(truths[k])))
        for agent in agents.build_all(cross=True):
            rng_pol = np.random.default_rng(50_000 + e)
            checkpoints, allocated = run_agent(env, agent, moments, truths, recovery,
                                               fields, heldout, rng_pol)
            for spent, nrounds, dt, nlpd, rmse, regret in checkpoints:
                rows.append((agent.name, e, spent, nrounds, dt, nlpd, rmse, regret))
            for k in range(env.n_blocks):
                alloc_rows.append((agent.name, e, k, float(truths[k]),
                                   float(allocated[k])))
        print("  property {:2d} done".format(e))

    save_csv(os.path.join(dd, "properties.csv"),
             ["env", "block", "prior_mean", "prior_var", "recovery", "grade_true"],
             env_rows)
    save_csv(os.path.join(rd, "allocation.csv"),
             ["policy", "env", "block", "grade_true", "volume_allocated"], alloc_rows)
    save_csv(os.path.join(rd, "sequential.csv"),
             ["policy", "env", "spent", "rounds", "inference_seconds", "nlpd", "grade_rmse",
              "regret"], rows)

    print("\nAt the full budget of {:.0f} m^3, mean over {} properties"
          .format(CHECKPOINTS[-1], N_ENV))
    print("  {:<24} {:>9} {:>11} {:>11} {:>9}".format(
        "agent", "NLPD", "grade RMSE", "regret", "ms/design"))
    for agent in agents.build_all(cross=True):
        sel = [r for r in rows
               if r[0] == agent.name and abs(r[2] - CHECKPOINTS[-1]) < 1e-9]
        if not sel:
            continue
        vals = tuple(float(np.mean([s[i] for s in sel])) for i in (5, 6, 7))
        per = 1e3 * float(np.mean([s[4] / max(s[3], 1) for s in sel]))
        print("  {:<24} {:9.4f} {:11.4f} {:11.4f} {:9.2f}".format(
            agent.name, *vals, per))
    return 0


if __name__ == "__main__":
    sys.exit(main())
