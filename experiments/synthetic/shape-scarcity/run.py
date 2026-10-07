"""When does an exact information criterion pay over a two-moment surrogate?

The two allocation studies find expected information gain, EPIG and local D-optimality
indistinguishable on accuracy. That is not an accident of those problems. D-optimality reads
the belief through ``V[lam]/E[lam]`` alone, and as the expected count grows the exact
criterion satisfies ``EIG -> (1/2) D-opt + c``, with ``c`` set by the *shape* of the mixing
law. Both studies hand every context a prior of the same order, so ``c`` is shared, the two
criteria are affinely related, and no ranking can separate them.

This study varies the shape across contexts, which makes ``c`` differ between them, and
sweeps the budget at two count levels. It reports where the difference reaches a decision
and where it does not, including the budget at which the proposed rule *loses*.

A gate runs first and aborts the study unless the design premise holds: that the achieved
moments are the requested ones, and that D-optimality and expected information gain have
strict and opposite preferences over the two classes at the priors they are given. Without
that the sweep would be measuring nothing in particular.

Writes ``results/``. Figures come from ``visualize.py``.

Run: python experiments/synthetic/shape-scarcity/run.py
"""

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

import agents                                                         # noqa: E402
from domain import ShapeScarcity, REGIMES                             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0
N_ENV = 80             # episodes per (regime, budget)
N_HELDOUT = 24          # held-out counts per context
TOP_M = 4               # contexts a commitment would take
N_CONTEXTS = 12

#: Total budget, in exposure, shared by the contexts. The per-context share runs from a
#: fraction of the shortest measurement on offer up to several times the largest.
BUDGETS = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)

#: The criteria, all under the proposed model, so the acquisition is the only thing varying.
#: This is the upper block of Tables 2 and 4.
CROSS = False


def save_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("  wrote {} ({} rows)".format(os.path.relpath(path, HERE), len(rows)))


# --------------------------------------------------------------------------
# Design gate
# --------------------------------------------------------------------------

def gate(verbose=True):
    """Check the premise: moments as requested, and strict opposite preferences.

    Returns ``(ok, rows)``. A failure here means the sweep below would be comparing two
    criteria that happen to agree, which is exactly the situation this study exists to get
    out of, so the study aborts rather than reporting it.
    """
    from methods.countmodels import GIGPoisson
    rows, bad = [], 0
    for regime in REGIMES:
        env = ShapeScarcity(n_contexts=N_CONTEXTS, regime=regime, seed=SEED)
        model = GIGPoisson()
        beliefs = env.initial_beliefs(model)
        by_class = {}
        for c, bel in zip(env._classes, beliefs):
            by_class.setdefault(c, bel)
        for f in (float(min(env.action_values)), float(max(env.action_values))):
            e = {c: model.eig(b, f) for c, b in by_class.items()}
            d = {c: model.fisher_dopt(b, f) for c, b in by_class.items()}
            # Per unit cost is what the policy ranks by, and cost is the exposure itself.
            eig_pick = max(e, key=lambda c: e[c] / f)
            dopt_pick = max(d, key=lambda c: d[c] / f)
            opposite = (eig_pick == "mild" and dopt_pick == "heavy")
            rows.append((regime, f, e["heavy"], e["mild"], d["heavy"], d["mild"],
                         eig_pick, dopt_pick, bool(opposite)))
            if not opposite:
                bad += 1
    if verbose:
        print("Design gate: are the two criteria given strict, opposite preferences?")
        for regime in REGIMES:
            env = ShapeScarcity(n_contexts=N_CONTEXTS, regime=regime, seed=SEED)
            for c, order, m, phi in env.class_summary():
                want = REGIMES[regime][c]
                flag = "OK" if abs(phi - want[1]) < 1e-3 * want[1] else "MISMATCH"
                print("  {:<5} {:<6} order {:+5.2f}  E[lam]={:7.4f}  V/E={:7.4f}  ({})"
                      .format(regime, c, order, m, phi, flag))
        print("  {:<5} {:>6} {:>10} {:>10} {:>10} {:>10}  {:<12}".format(
            "", "f", "EIG heavy", "EIG mild", "Dopt heavy", "Dopt mild", "picks"))
        for r in rows:
            print("  {:<5} {:>6g} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f}  "
                  "EIG->{:<5} Dopt->{:<5} {}".format(
                      r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
                      "opposite" if r[8] else "**AGREE**"))
        print("  ->  {} ({} of {} not opposite)".format(
            "PASS" if bad == 0 else "FAIL", bad, len(rows)))
    return bad == 0, rows


# --------------------------------------------------------------------------
# One episode
# --------------------------------------------------------------------------

def evaluate(env, agent, truths, heldout):
    nlpd, sq = [], []
    for k in range(env.n_contexts):
        nlpd.append(-float(np.mean(agent.logpmf(k, env.eval_exposure,
                                                np.asarray(heldout[k], dtype=float)))))
        est = max(agent.rate_mean(k), 1e-12)
        sq.append((np.log10(est) - np.log10(truths[k])) ** 2)
    means = np.array([agent.rate_mean(k) for k in range(env.n_contexts)])
    chosen = np.argsort(-means)[:TOP_M]
    best = np.sort(truths)[::-1][:TOP_M]
    regret = float(best.sum() - truths[chosen].sum()) / float(best.sum())
    return float(np.mean(nlpd)), float(np.sqrt(np.mean(sq))), regret


def run_agent(env, agent, moments, truths, streams, heldout, rng):
    agent.reset(env, moments, np.ones(env.n_contexts))
    consumed = np.zeros(env.n_contexts)
    allocated = np.zeros(env.n_contexts)
    spent, rounds, decide_time = 0.0, 0, 0.0

    while spent < env.budget - 1e-9:
        t0 = time.perf_counter()
        k, v = agent.act(env, rng)
        decide_time += time.perf_counter() - t0
        if spent + v > env.budget + 1e-9:
            v = float(min(env.action_values))
            if spent + v > env.budget + 1e-9:
                break
        f = env.exposure((k, v))
        lo, hi = consumed[k], consumed[k] + f
        y = int(np.searchsorted(streams[k], hi) - np.searchsorted(streams[k], lo))
        consumed[k] = hi
        agent.observe(k, y, f)
        allocated[k] += f
        spent += v
        rounds += 1

    nlpd, dex, regret = evaluate(env, agent, truths, heldout)
    heavy = np.array([c == "heavy" for c in env._classes])
    share = float(allocated[heavy].sum() / max(allocated.sum(), 1e-12))
    touched = float((allocated > 0).mean())
    ms = 1000.0 * decide_time / max(rounds, 1)
    return nlpd, dex, regret, share, touched, ms, allocated


# --------------------------------------------------------------------------

def main():
    rd = os.path.join(HERE, "results")
    os.makedirs(rd, exist_ok=True)

    ok, gate_rows = gate()
    save_csv(os.path.join(rd, "gate_design.csv"),
             ["regime", "exposure", "eig_heavy", "eig_mild", "dopt_heavy", "dopt_mild",
              "eig_picks", "dopt_picks", "opposite"], gate_rows)
    if not ok:
        print("\nABORT: the two criteria do not have opposite preferences under this "
              "design, so the sweep would measure nothing.")
        return 1

    built = agents.build_all(cross=CROSS)
    print("\nSweep: {} regimes x {} budgets x {} episodes x {} agents".format(
        len(REGIMES), len(BUDGETS), N_ENV, len(built)))

    rows, alloc_rows = [], []
    t_start = time.time()
    for regime in REGIMES:
        for budget in BUDGETS:
            env = ShapeScarcity(n_contexts=N_CONTEXTS, regime=regime,
                                budget=budget, seed=SEED)
            per_agent = {}
            for e in range(N_ENV):
                rng_env = np.random.default_rng(1000 + e)
                moments, truths, _ = env.blocks(rng_env)
                streams = env.streams(truths, rng_env)
                heldout = env.held_out(truths, rng_env, n=N_HELDOUT)
                for agent in agents.build_all(cross=CROSS):
                    out = run_agent(env, agent, moments, truths, streams, heldout,
                                    np.random.default_rng(5000 + e))
                    per_agent.setdefault(agent.name, []).append(out[:6])
                    rows.append((regime, budget, e, agent.name) + tuple(out[:6]))
            ref = np.array(per_agent["eig"], dtype=float)
            print("  {:<5} budget {:>5g} ({:>5.2f} per context)  [{:.0f}s]".format(
                regime, budget, budget / N_CONTEXTS, time.time() - t_start))
            for name in sorted(per_agent):
                a = np.array(per_agent[name], dtype=float)
                ts = []
                for col in (0, 1, 2):
                    d = ref[:, col] - a[:, col]
                    ts.append(float(np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d))
                                                  + 1e-15)))
                alloc_rows.append((regime, budget, name, a[:, 0].mean(), a[:, 1].mean(),
                                   a[:, 2].mean(), a[:, 3].mean(), a[:, 4].mean(),
                                   a[:, 5].mean(), ts[0], ts[1], ts[2],
                                   a[:, 0].std(ddof=1) / np.sqrt(len(a)),
                                   a[:, 1].std(ddof=1) / np.sqrt(len(a))))

    save_csv(os.path.join(rd, "episodes.csv"),
             ["regime", "budget", "episode", "agent", "nlpd", "dex", "regret",
              "heavy_share", "touched", "ms"], rows)
    save_csv(os.path.join(rd, "summary.csv"),
             ["regime", "budget", "agent", "nlpd", "dex", "regret", "heavy_share",
              "touched", "ms", "t_nlpd", "t_dex", "t_regret", "sem_nlpd", "sem_dex"],
             alloc_rows)
    print("\nDone in {:.0f}s.".format(time.time() - t_start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
