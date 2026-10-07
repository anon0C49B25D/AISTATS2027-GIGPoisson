"""Allocating telescope time: the same agents, on a field of rates nobody simulated.

Identical in structure to the bulk-sampling study, and deliberately so: the same agents, the
same budget discipline, the same three scores. What changes is where the rates come from. In
the bulk-sampling study they are drawn from the GIG-Poisson law, so a model that assumes that
law is correct by construction. Here they are measured photon fluxes from the Fermi-LAT
catalogue, and every model in the comparison is misspecified to some degree, ours included.

Scores, the metrics of the problem statement plus one diagnostic:

    NLPD        average surprisal of held-out photon counts under the agent's own
                predictive, in nats.
    regret      cumulative rate gap, ``sum_t t_t (lam_{k*} - lam_{k_t})``: after every
                observation the source with the highest posterior mode is the agent's pick,
                and the flux it forgoes is charged for that observation's integration time.
                In 1e-10 cm^-2 s^-1 Ms.
    inference   wall-clock seconds spent choosing a design and updating the belief on its
                outcome, cumulative. Divide by ``rounds`` for the cost per design.
    flux error  root-mean-square error of the posterior mean flux, in dex; a diagnostic
                only. Fluxes span decades, so an absolute error would be a report on the
                brightest source alone; a logarithmic one weights every source equally.

Run: python experiments/gamma-ray/run.py
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

#: The criterion-by-model cross is established in the bulk-sampling study; here the ten
#: criteria run under the proposed model and the proposed criterion runs under each model,
#: which is what this environment is for.
from environment import PhotonCounting                                # noqa: E402
from methods.countmodels import MODELS                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0
N_ENV = 24
N_HELDOUT = 24
CHECKPOINTS = (15.0, 30.0, 60.0, 90.0, 120.0)     # telescope time spent, Ms


def save_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("  wrote {} ({} rows)".format(os.path.relpath(path, HERE), len(rows)))


def _nested_mc_eig(model, belief, f, rng, n=4000):
    lam = model.sample_rate(belief, n, rng)
    y = rng.poisson(f * lam).astype(float)
    ll_out = -f * lam + y * np.log(f * lam) - gammaln(y + 1.0)
    lam_in = model.sample_rate(belief, n, rng)
    mu = f * lam_in
    ll_in = -mu[None, :] + y[:, None] * np.log(mu)[None, :] - gammaln(y + 1.0)[:, None]
    return float(np.mean(ll_out - (logsumexp(ll_in, axis=1) - np.log(n))))


def gate(env, verbose=True):
    """Every model's acquisition against nested Monte Carlo, under this environment's prior."""
    rng = np.random.default_rng(SEED)
    rows, bad = [], 0
    for cls in MODELS:
        model = cls()
        belief = env.initial_beliefs(model)[0]
        for t in (env.times[0], env.times[-1]):
            f = env.exposure((0, t))
            exact = model.eig(belief, f)
            lo = [_nested_mc_eig(model, belief, f, rng, 1500) for _ in range(4)]
            hi = [_nested_mc_eig(model, belief, f, rng, 6000) for _ in range(4)]
            corrected = (4.0 * np.mean(hi) - np.mean(lo)) / 3.0
            sem = float(np.sqrt((16.0 * np.var(hi, ddof=1) / 4
                                 + np.var(lo, ddof=1) / 4) / 9.0))
            z = (exact - corrected) / max(sem, 1e-12)
            rows.append((model.name, float(t), f, exact, corrected, sem, z))
            if abs(z) > 4.0 or exact < -1e-12:
                bad += 1
    ok = bad == 0
    if verbose:
        print("Correctness gate: each model's EIG against nested Monte Carlo")
        print("  {:<20} {:>7} {:>11} {:>11} {:>9} {:>7}".format(
            "model", "time", "eig exact", "bias-corr.", "sem", "z"))
        for name, t, f, e, c, s, z in rows:
            print("  {:<20} {:7.3g} {:11.6f} {:11.6f} {:9.2e} {:+7.2f}"
                  .format(name, t, e, c, s, z))
        print("  ->  {} ({} of {} outside 4 sigma)".format(
            "PASS" if ok else "FAIL", bad, len(rows)))
    return ok, rows


def evaluate(env, agent, truths, heldout):
    nlpd, sq = [], []
    f_eval = env.eval_time * env.exposure((0, 1.0)) / 1.0
    for k in range(env.n_sources):
        nlpd.append(-float(np.mean(agent.logpmf(k, f_eval,
                                                np.asarray(heldout[k], dtype=float)))))
        est = max(agent.rate_mean(k), 1e-12)
        sq.append((np.log10(est) - np.log10(truths[k])) ** 2)
    return float(np.mean(nlpd)), float(np.sqrt(np.mean(sq)))


def best_source(agent, modes):
    """The source with the highest posterior mode.

    A gamma posterior with shape below one has its mode at the origin, so unobserved
    sources can tie at zero. Ties go to the higher posterior mean rather than to the lower
    index, which would reward whatever order the catalogue happens to list them in.
    """
    tied = np.flatnonzero(modes == modes.max())
    if tied.size == 1:
        return int(tied[0])
    return int(tied[np.argmax([agent.rate_mean(k) for k in tied])])


def run_agent(env, agent, moments, truths, recovery, streams, heldout, rng):
    agent.reset(env, moments, recovery)
    consumed = np.zeros(env.n_sources)
    allocated = np.zeros(env.n_sources)
    spent, rounds, infer_time, regret = 0.0, 0, 0.0, 0.0
    best = float(np.max(truths))
    modes = np.array([agent.rate_map(k) for k in range(env.n_sources)])
    checkpoints, next_cp = [], 0

    while next_cp < len(CHECKPOINTS):
        t0 = time.perf_counter()
        k, t = agent.act(env, rng)
        infer_time += time.perf_counter() - t0

        if spent + t > CHECKPOINTS[-1]:
            t = float(min(env.times))
            if spent + t > CHECKPOINTS[-1] + 1e-9:
                break

        f = env.exposure((k, t))
        lo, hi = consumed[k], consumed[k] + f
        y = int(np.searchsorted(streams[k], hi) - np.searchsorted(streams[k], lo))
        consumed[k] = hi
        t0 = time.perf_counter()
        agent.observe(k, y, f)
        infer_time += time.perf_counter() - t0
        allocated[k] += t
        spent += t
        rounds += 1

        # Only source k's belief moved, so only its mode needs recomputing.
        modes[k] = agent.rate_map(k)
        regret += t * (best - float(truths[best_source(agent, modes)]))

        while next_cp < len(CHECKPOINTS) and spent >= CHECKPOINTS[next_cp] - 1e-9:
            checkpoints.append((CHECKPOINTS[next_cp], rounds, infer_time,
                                *evaluate(env, agent, truths, heldout), regret))
            next_cp += 1

    while next_cp < len(CHECKPOINTS):
        checkpoints.append((CHECKPOINTS[next_cp], rounds, infer_time,
                            *evaluate(env, agent, truths, heldout), regret))
        next_cp += 1
    return checkpoints, allocated


def main():
    dd, rd = os.path.join(HERE, "data"), os.path.join(HERE, "results")
    env = PhotonCounting()

    ok, gate_rows = gate(env)
    save_csv(os.path.join(rd, "gate_eig.csv"),
             ["model", "time", "exposure", "eig_exact", "eig_corrected", "sem", "z"],
             gate_rows)
    if not ok:
        print("\nABORT: an acquisition failed its correctness gate.")
        return 1

    print("\nObserving campaigns: {} programmes, {} sources, budget {:.0f} Ms "
          "({:.1f} Ms per source), {} agents".format(
              N_ENV, env.n_sources, env.budget, env.budget / env.n_sources,
              len(agents.build_all(cross="models"))))
    rows, env_rows, alloc_rows = [], [], []
    for e in range(N_ENV):
        rng_env = np.random.default_rng(1000 + e)
        moments, truths, recovery = env.blocks(rng_env)
        streams = env.stone_fields(truths, recovery, rng_env)
        heldout = env.held_out(truths, recovery, rng_env, n=N_HELDOUT)
        for k in range(env.n_sources):
            env_rows.append((e, k, float(truths[k])))
        for agent in agents.build_all(cross="models"):
            rng_pol = np.random.default_rng(50_000 + e)
            checkpoints, allocated = run_agent(env, agent, moments, truths, recovery,
                                               streams, heldout, rng_pol)
            for spent, nrounds, dt, nlpd, dex, regret in checkpoints:
                rows.append((agent.name, e, spent, nrounds, dt, nlpd, dex, regret))
            for k in range(env.n_sources):
                alloc_rows.append((agent.name, e, k, float(truths[k]),
                                   float(allocated[k])))
        # Written after every programme, so a run that is interrupted still leaves usable
        # results rather than nothing at all.
        save_csv(os.path.join(rd, "sequential.csv"),
                 ["policy", "env", "spent", "rounds", "inference_seconds", "nlpd",
                  "flux_dex", "regret"], rows)
        print("  programme {:2d} done".format(e))

    save_csv(os.path.join(dd, "programmes.csv"), ["env", "source", "flux_true"], env_rows)
    save_csv(os.path.join(rd, "allocation.csv"),
             ["policy", "env", "source", "flux_true", "time_allocated"], alloc_rows)
    save_csv(os.path.join(rd, "sequential.csv"),
             ["policy", "env", "spent", "rounds", "inference_seconds", "nlpd", "flux_dex",
              "regret"], rows)

    print("\nAt the full budget of {:.0f} Ms, mean over {} programmes"
          .format(CHECKPOINTS[-1], N_ENV))
    print("  {:<24} {:>9} {:>11} {:>11} {:>9}".format(
        "agent", "NLPD", "flux (dex)", "regret", "ms/design"))
    for agent in agents.build_all(cross="models"):
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
