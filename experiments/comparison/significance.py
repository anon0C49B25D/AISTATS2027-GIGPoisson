"""Significance tests between the methods, per setting, on NLPD and on top-m regret.

Every agent in an episode faces the same ground truth, the same photon or stone streams and
the same held-out counts, so per-episode scores are paired by episode. Per setting and per
metric:

    Friedman      omnibus test that the methods' scores differ, over episodes.
    pairwise      every pair of methods, by a paired t-test on the per-episode differences
                  and by a Wilcoxon signed-rank test, which does not assume they are normal.
    Holm          both families of pairwise p-values corrected within the setting and the
                  metric, so each table controls the family-wise error rate at its level.

Metrics: held-out NLPD at the full budget, and top-m regret both at the full budget and as
the area under the regret curve (its mean over the checkpoints), for m = 1 and m = 5.

Reported per pair: mean difference (row minus column, so negative means the row is better),
its 95 per cent confidence interval, the number of episodes in which the row is better, and
the Holm-adjusted p-values. Holm is applied over every pair in a setting; the printout shows
the pairs against the proposed method, the CSV all of them.

Writes ``results/significance.csv`` and prints one block per setting and metric.

Run: python experiments/comparison/significance.py
"""

import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

SETTINGS = [
    ("bulk", "diamond bulk sampling: 96 blocks, 240 m^3, NLPD at 20 m^3"),
    ("gamma", "gamma-ray survey: 192 sources, 120 Ms, NLPD at 25 Ms"),
    ("hardxray", "hard X-ray follow-up: 192 sources, 120 ks, NLPD at 20 ks"),
]
#: (column, label, how): ``final`` scores the last checkpoint, ``auc`` the mean over all.
METRICS = [("nlpd", "NLPD at the full budget", "final"),
           ("regret_top1", "top-1 regret at the full budget", "final"),
           ("regret_top5", "top-5 regret at the full budget", "final"),
           ("regret_top1", "top-1 regret, area under the budget curve", "auc"),
           ("regret_top5", "top-5 regret, area under the budget curve", "auc")]
METHODS = ["eig", "eig@gamma-poisson", "eig@gammamix-poisson", "eig@logskewnormal-quad",
           "eig@logskewnormal-poisson", "eig@logskewnormal-poisson-m8k", "systematic", "random",
           "d-optimality", "thompson", "lucb", "lucb@gamma-poisson", "dad",
           "bo-ei@gp-poisson", "bo-ei@gp-poisson-cat",
           # Added 2026-10-04: the rest of the prior-by-rule factorial, the nested estimators
           # under the GIG prior, and the in-regime rivals.
           "d-optimality@gamma-poisson", "d-optimality@gammamix-poisson",
           "systematic@gamma-poisson", "systematic@gammamix-poisson",
           "eig@gig-nmc", "eig@gig-pce", "eig@compoundgamma-poisson",
           "eig@logskewnormal-quad-pen"]
SHORT = {"eig": "GIG", "eig@gamma-poisson": "gamma", "eig@gammamix-poisson": "gamma-mix",
         "eig@logskewnormal-quad": "LSN-quad", "eig@logskewnormal-poisson": "LSN-MCMC",
         "eig@logskewnormal-poisson-m8k": "LSN-MCMC-8k", "systematic": "systematic",
         "random": "random", "d-optimality": "D-opt", "thompson": "Thompson",
         "lucb": "LUCB-GIG", "lucb@gamma-poisson": "LUCB-gamma", "dad": "DAD",
         "bo-ei@gp-poisson": "BO-RBF", "bo-ei@gp-poisson-cat": "BO-cat",
         "d-optimality@gamma-poisson": "D-opt-gamma",
         "d-optimality@gammamix-poisson": "D-opt-mix",
         "systematic@gamma-poisson": "sys-gamma", "systematic@gammamix-poisson": "sys-mix",
         "eig@gig-nmc": "GIG-NMC", "eig@gig-pce": "GIG-PCE",
         "eig@compoundgamma-poisson": "cgamma", "eig@logskewnormal-quad-pen": "LSN-pen"}
#: The reference every method is printed against; the CSV holds every pair.
REFERENCE = "eig"


def holm(p):
    """Holm step-down adjustment of a vector of p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (p.size - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def setting_table(name, column="nlpd", how="final"):
    d = pd.read_csv(os.path.join(RESULTS, name + ".csv"))
    if how == "final":
        sub = d[np.isclose(d.spent, d.spent.max())]
        wide = sub.pivot(index="episode", columns="agent", values=column)
    else:
        wide = d.groupby(["episode", "agent"])[column].mean().unstack("agent")
    methods = [m for m in METHODS if m in wide.columns]
    wide = wide[methods].dropna()
    try:
        fried = stats.friedmanchisquare(*[wide[m].values for m in methods])
    except ValueError:
        fried = None
    rows = []
    for a, b in itertools.combinations(methods, 2):
        x = wide[a].values - wide[b].values
        n = x.size
        se = x.std(ddof=1) / np.sqrt(n)
        half = stats.t.ppf(0.975, n - 1) * se
        if np.all(x == 0):
            pt, tt, pw = 1.0, 0.0, 1.0
        else:
            t = stats.ttest_rel(wide[a].values, wide[b].values)
            w = stats.wilcoxon(wide[a].values, wide[b].values, zero_method="pratt")
            pt, tt, pw = t.pvalue, t.statistic, w.pvalue
        rows.append(dict(setting=name, metric=column, how=how, a=a, b=b, n=n,
                         mean_diff=x.mean(), ci_lo=x.mean() - half, ci_hi=x.mean() + half,
                         a_better=int((x < 0).sum()), b_better=int((x > 0).sum()), t=tt,
                         p_t=pt, p_wilcoxon=pw))
    tab = pd.DataFrame(rows)
    tab["p_t_holm"] = holm(tab.p_t)
    tab["p_wilcoxon_holm"] = holm(tab.p_wilcoxon)
    means = wide.mean()
    return tab, fried, means, wide.shape[0]


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="",
                    help="results prefix, e.g. fact_ (reads results/<prefix><setting>.csv, "
                         "writes results/<prefix>significance.csv)")
    prefix = ap.parse_args().prefix
    out = []
    for name, label in SETTINGS:
        path = os.path.join(RESULTS, prefix + name + ".csv")
        if not os.path.exists(path):
            continue
        for column, mlabel, how in METRICS:
            tab, fried, means, n = setting_table(prefix + name, column, how)
            out.append(tab)
            print("\n== {}: {} ({} episodes)".format(label, mlabel, n))
            print("   mean: " + ", ".join("{} {:.4f}".format(SHORT[m], v)
                                         for m, v in means.items()))
            if fried is not None:
                print("   Friedman chi2 = {:.1f}, p = {:.1e}".format(fried.statistic,
                                                                   fried.pvalue))
            print("   {:<15} {:>9} {:>21} {:>7} {:>7} {:>10} {:>10}".format(
                "A vs B", "A - B", "95% CI", "A wins", "t", "p_t Holm", "p_W Holm"))
            for r in tab[(tab.a == REFERENCE) | (tab.b == REFERENCE)].itertuples():
                print("   {:<15} {:>+9.4f} [{:>+8.4f}, {:>+8.4f}] {:>4d}/{:<2d} {:>+7.1f} "
                      "{:>7.1e} {:<4} {:>7.1e} {:<4}".format(
                          SHORT[r.a] + " vs " + SHORT[r.b], r.mean_diff, r.ci_lo, r.ci_hi,
                          r.a_better, r.n, r.t, r.p_t_holm, stars(r.p_t_holm),
                          r.p_wilcoxon_holm, stars(r.p_wilcoxon_holm)))
    res = pd.concat(out, ignore_index=True)
    path = os.path.join(RESULTS, prefix + "significance.csv")
    res.to_csv(path, index=False)
    print("\nwrote {}".format(os.path.relpath(path)))


if __name__ == "__main__":
    main()
