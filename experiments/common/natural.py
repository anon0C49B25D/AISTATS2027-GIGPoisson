"""One protocol for measured count data: class priors from small catalogues, scored on new contexts.

A natural count data set offers contexts, each with a chronological run of counts at known
exposures, and a class that is known before anything is measured. Nothing is simulated, so
nothing can be redrawn: an *episode* is a random split, within each class, of the contexts
into a calibration pool and a test half.

Per episode and catalogue size ``n``:

* each law's prior for a class is fitted by marginal likelihood (``countfit.py``) to all
  counts of the first ``n`` contexts of the class's calibration pool. Catalogues are nested,
  so the 20 contexts of ``n = 20`` are among the 40 of ``n = 40``, and a catalogue larger
  than the pool is the whole pool;
* every test context has its readings ``J_MAX + 1`` to ``J_MAX + HOLD`` held out, and they
  are predicted after its first ``j`` readings have been observed, for ``j`` in
  ``CONDITION``. ``j = 0`` is the prior predictive of a context never measured;
* two scores per test context: the mean surprisal of the held-out readings one at a time
  (``reading``), and the surprisal of their sum (``integration``). The second is a single
  long exposure, which is where the tails of the predictive matter most (Lemma 3 of the
  paper).

Scores are averaged over the test contexts of all classes, every context weighing the same,
and laws are compared paired by episode: a paired t-test and a Wilcoxon signed-rank test,
Holm-corrected over the rivals at each catalogue size, condition and score. Episodes overlap
(they resample one data set), so the tests are a guide to consistency across splits, not to
sampling from a population.
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import countfit                                                       # noqa: E402

__all__ = ["build_cells", "describe", "population_fits", "run_episodes", "summarise",
           "CATALOGUES", "CONDITION", "HOLD", "J_MAX", "EPISODES", "REFERENCE"]

CATALOGUES = (5, 10, 20, 40, 100, 400)
CONDITION = (0, 1, 2, 5)
J_MAX = max(CONDITION)
HOLD = 10
EPISODES = 50
MAX_TEST_PER_CLASS = 500
REFERENCE = "gig"
SEED = 0


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def build_cells(cell, t, count, dist_km, bands, min_readings=J_MAX + 1):
    """Contexts from readings: per cell, its counts in time order and its class.

    ``cell`` is each reading's cell index, ``t`` its time, ``dist_km`` its distance from the
    study centre. A cell's class is the distance band of its mean distance. Cells with fewer
    than ``min_readings`` readings are dropped, since they have nothing to hold out.
    """
    order = np.lexsort((t, cell))
    cell, count, dist_km = cell[order], count[order], dist_km[order]
    starts = np.flatnonzero(np.r_[True, cell[1:] != cell[:-1]])
    ends = np.r_[starts[1:], cell.size]
    counts, classes, ids, dists = [], [], [], []
    for s, e in zip(starts, ends):
        if e - s < min_readings:
            continue
        d = float(np.mean(dist_km[s:e]))
        k = int(np.digitize(d, bands[1:-1]))
        counts.append(count[s:e].astype(float))
        classes.append(k)
        ids.append(int(cell[s]))
        dists.append(d)
    return counts, np.asarray(classes), np.asarray(ids), np.asarray(dists)


def describe(counts, classes, band_labels, per_minute):
    """One row per class (and one for all): sizes, the spread of the cell rates, dispersion.

    ``within_fano`` is each cell's variance-to-mean ratio of its counts, over cells with at
    least ten readings, which is about 1 if counts are Poisson about a fixed cell rate.
    ``fano_1`` and ``fano_10`` are the variance-to-mean of a cell's first reading and of the
    sum of its first ten, across cells: the overdispersion a model has to carry, which grows
    with the exposure as ``1 + f V / E``.
    """
    rows = []
    for k in list(range(len(band_labels))) + [None]:
        sel = [i for i in range(len(counts)) if k is None or classes[i] == k]
        c = [counts[i] for i in sel]
        rate = np.array([x.mean() for x in c]) * per_minute
        rich = [x for x in c if x.size >= 10]
        wf = np.array([x.var(ddof=1) / x.mean() for x in rich if x.mean() > 0])
        first = np.array([x[0] for x in c])
        ten = np.array([x[:10].sum() for x in rich])
        rows.append(dict(
            cls="all" if k is None else band_labels[k], cells=len(c),
            readings=int(sum(x.size for x in c)),
            rate_median_cpm=float(np.median(rate)), rate_p90_cpm=float(np.percentile(rate, 90)),
            rate_p99_cpm=float(np.percentile(rate, 99)), rate_max_cpm=float(rate.max()),
            p99_over_median=float(np.percentile(rate, 99) / np.median(rate)),
            within_fano_median=float(np.median(wf)) if wf.size else np.nan,
            within_fano_q25=float(np.percentile(wf, 25)) if wf.size else np.nan,
            within_fano_q75=float(np.percentile(wf, 75)) if wf.size else np.nan,
            fano_1=float(first.var(ddof=1) / first.mean()),
            fano_10=float(ten.var(ddof=1) / ten.mean()) if ten.size > 1 else np.nan))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Population fits: does the third parameter earn its place on these counts?
# --------------------------------------------------------------------------

def population_fits(counts, classes, band_labels, laws=countfit.LAWS):
    """Each law fitted to every cell of each class, with AIC and BIC.

    Log-likelihoods include the term ``-sum log y!`` common to every law, so they are the
    log-probability of the counts themselves.
    """
    rows = []
    for k, lab in enumerate(band_labels):
        c = [counts[i] for i in range(len(counts)) if classes[i] == k]
        T = np.array([x.sum() for x in c])
        F = np.array([float(x.size) for x in c])
        const = -float(sum(np.sum(_lgamma1(x)) for x in c))
        for law in laws:
            th, M = countfit.fit(law, T, F)
            k_ = countfit.n_params(law, th)
            ll = M + const
            rows.append(dict(cls=lab, law=law, cells=len(c), k=k_, loglik=ll,
                             aic=2 * k_ - 2 * ll, bic=k_ * np.log(len(c)) - 2 * ll,
                             params=json.dumps([float(v) for v in th]),
                             **_describe_fit(law, th)))
    df = pd.DataFrame(rows)
    df["daic"] = df["aic"] - df.groupby("cls")["aic"].transform("min")
    df["dbic"] = df["bic"] - df.groupby("cls")["bic"].transform("min")
    return df


def _lgamma1(x):
    from scipy.special import gammaln
    return gammaln(np.asarray(x, float) + 1.0)


def _describe_fit(law, th):
    """Interpretable numbers for a fit: the GIG's order and concentration, a tail index."""
    out = {}
    if law == "gig":
        nu, a, b = countfit.gig_params(th)
        out.update(gig_order=float(nu), gig_omega=float(np.sqrt(a * b)))
    if law == "compoundgamma":
        out.update(cg_tail_index=float(1.0 + np.exp(th[1])))
    if law == "gammamix":
        out.update(mix_components=int((len(th) + 1) // 3))
    return out


# --------------------------------------------------------------------------
# Episodes
# --------------------------------------------------------------------------

def _test_arrays(counts, idx):
    """Per test context: observed (T_j, F_j) for each j, and the held-out readings."""
    H = np.full((idx.size, HOLD), np.nan)
    Tj = {j: np.zeros(idx.size) for j in CONDITION}
    for r, i in enumerate(idx):
        x = counts[i]
        h = x[J_MAX:J_MAX + HOLD]
        H[r, :h.size] = h
        for j in CONDITION:
            Tj[j][r] = x[:j].sum()
    return Tj, H


def _score(law, th, Tj, H):
    """Per context: mean surprisal of the held-out readings, and surprisal of their sum."""
    mask = np.isfinite(H)
    Hz = np.where(mask, H, 0.0)
    n = mask.sum(axis=1).astype(float)
    out = {}
    for j in CONDITION:
        T = Tj[j][:, None]
        F = np.full_like(T, float(j))
        lp = countfit.predictive(law, th, T, F, Hz, 1.0)
        reading = -np.sum(np.where(mask, lp, 0.0), axis=1) / n
        integ = -countfit.predictive(law, th, Tj[j], np.full(n.shape, float(j)),
                                     Hz.sum(axis=1), n)
        # The surprisal of a count is never negative. One that is, or is not finite, is a
        # numerical failure, and it stops the run rather than entering an average.
        for name, v in (("reading", reading), ("integration", integ)):
            if not (np.all(np.isfinite(v)) and float(np.min(v)) > -1e-9):
                raise FloatingPointError(
                    "{} score of law {} at j={} is {:.4g} (theta={})".format(
                        name, law, j, float(np.nanmin(v)), list(np.round(th, 4))))
        out[j] = (reading, integ)
    return out


def _episode(args):
    e, counts, classes, n_classes, catalogues, laws = args
    rng = np.random.default_rng(SEED + 1000 + e)
    splits = []
    for k in range(n_classes):
        idx = rng.permutation(np.flatnonzero(classes == k))
        half = idx.size // 2
        splits.append((idx[:half], idx[half:][:MAX_TEST_PER_CLASS]))
    tests = [_test_arrays(counts, test) for _, test in splits]
    rows, fits = [], []
    prev = {}
    t0 = time.time()
    for n in sorted(catalogues):
        acc = {(law, j, s): [] for law in laws for j in CONDITION for s in (0, 1)}
        for k, (pool, test) in enumerate(splits):
            cal = pool[:n]
            T = np.array([counts[i].sum() for i in cal])
            F = np.array([float(counts[i].size) for i in cal])
            Tj, H = tests[k]
            for law in laws:
                # Catalogues are nested, so the fit to the next smaller one is a warm start.
                th, M = countfit.fit(law, T, F, init=prev.get((k, law)))
                prev[(k, law)] = th
                fits.append(dict(episode=e, catalogue=n, cls=k, law=law, cells=int(cal.size),
                                 marginal=M, params=json.dumps([float(v) for v in th]),
                                 **_describe_fit(law, th)))
                sc = _score(law, th, Tj, H)
                for j in CONDITION:
                    acc[(law, j, 0)].append(sc[j][0])
                    acc[(law, j, 1)].append(sc[j][1])
        for (law, j, s), v in acc.items():
            v = np.concatenate(v)
            rows.append(dict(episode=e, catalogue=n, law=law, condition=j,
                             score=("reading", "integration")[s], nlpd=float(np.mean(v)),
                             contexts=int(v.size)))
    return rows, fits, time.time() - t0


def run_episodes(counts, classes, n_classes, episodes=EPISODES, catalogues=CATALOGUES,
                 laws=countfit.LAWS, workers=None, log=print, partial=None,
                 fresh_after=0.0):
    """Every episode, in parallel. Returns ``(scores, fits)`` as data frames.

    With ``partial`` (a directory), each episode is written there as it finishes and an
    episode already there is read back instead of rerun, so an interrupted run resumes
    where it stopped. An episode is a pure function of its index, so this changes nothing
    in the result. Files older than ``fresh_after`` (a time stamp: the code that wrote them
    has changed since) are ignored and rewritten.
    """
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 2))
    done = {}
    if partial:
        os.makedirs(partial, exist_ok=True)
        for e in range(episodes):
            fs = [os.path.join(partial, "e{:03d}_{}.csv".format(e, k))
                  for k in ("scores", "fits")]
            if all(os.path.exists(f) and os.path.getmtime(f) > fresh_after for f in fs):
                done[e] = (pd.read_csv(fs[0]), pd.read_csv(fs[1]))
        if done:
            log("  {} episodes read back from {}".format(len(done), partial))
    todo = [e for e in range(episodes) if e not in done]
    args = [(e, counts, classes, n_classes, catalogues, laws) for e in todo]
    with ProcessPoolExecutor(workers) as ex:
        for e, (r, f, sec) in zip(todo, ex.map(_episode, args)):
            r, f = pd.DataFrame(r), pd.DataFrame(f)
            if partial:
                for k, df in (("scores", r), ("fits", f)):
                    fn = os.path.join(partial, "e{:03d}_{}.csv".format(e, k))
                    df.to_csv(fn + ".part", index=False)
                    os.replace(fn + ".part", fn)
            done[e] = (r, f)
            log("  episode {:>2}: {:.0f} s".format(e, sec))
    order = sorted(done)
    return (pd.concat([done[e][0] for e in order], ignore_index=True),
            pd.concat([done[e][1] for e in order], ignore_index=True))


# --------------------------------------------------------------------------
# Paired comparison
# --------------------------------------------------------------------------

def _holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (p.size - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def summarise(scores, reference=REFERENCE):
    """Mean NLPD per law, and each rival's paired difference from the reference law.

    ``diff`` is rival minus reference, so positive means the reference predicted better;
    ``ahead`` counts the episodes in which it did.
    """
    out = []
    for (n, j, s), g in scores.groupby(["catalogue", "condition", "score"]):
        piv = g.pivot(index="episode", columns="law", values="nlpd")
        ref = piv[reference]
        block = []
        for law in piv.columns:
            d = piv[law] - ref
            row = dict(catalogue=n, condition=j, score=s, law=law, nlpd=float(piv[law].mean()),
                       nlpd_sd=float(piv[law].std(ddof=1)), diff=float(d.mean()),
                       diff_se=float(d.std(ddof=1) / np.sqrt(d.size)),
                       ahead=int(np.sum(d > 0)), episodes=int(d.size))
            if law != reference and np.any(d != 0):
                row["t"] = float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))
                row["p_t"] = float(stats.ttest_1samp(d, 0.0).pvalue)
                row["p_wilcoxon"] = float(stats.wilcoxon(d).pvalue)
            block.append(row)
        rivals = [r for r in block if "p_t" in r]
        if rivals:
            pt = _holm([r["p_t"] for r in rivals])
            pw = _holm([r["p_wilcoxon"] for r in rivals])
            for r, a, b in zip(rivals, pt, pw):
                r["p_t_holm"], r["p_wilcoxon_holm"] = float(a), float(b)
                r["significant"] = bool(a < 0.01 and b < 0.01)
        out += block
    return pd.DataFrame(out)
