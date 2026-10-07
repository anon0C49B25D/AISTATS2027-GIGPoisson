"""Regenerate every results table in the paper from the recorded CSVs.

One script, so a number in the manuscript cannot drift away from the run that produced
it. Each table is written as a LaTeX fragment into ``tables/`` and pulled into
the manuscript with ``\\input``. Captions stay in the manuscript, since they carry the
design of the study rather than its results.

Run from this directory::

    python make_tables.py

Writes ``tables/bulk.tex``, ``tables/photon.tex``, ``tables/rate_fields.tex`` and
``tables/grid.tex``, and prints a short summary of what each one contains.
"""

import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tables")

#: Display name for each policy, and the order rows appear in. An agent is a criterion,
#: optionally at a non-default mixing law, written ``criterion@model``.
CRITERIA = [
    ("eig", "EIG (ours)"),
    ("epig", "EPIG"),
    ("d-optimality", "D-opt."),
    ("neyman", "Neyman"),
    ("uniform", "systematic"),
    ("epistemic", "epistemic"),
    ("predictive-variance", "total var."),
    ("random", "random"),
    ("thompson", "Thompson"),
    ("maxent", "max-ent."),
]
MODELS = [
    ("eig@gamma-poisson", "gamma-Poisson"),
    ("eig@lognormal-poisson", "lognormal-Pois."),
]

#: The point of the shape-and-scarcity sweep the manuscript tabulates. Fixed here rather
#: than chosen as the sweep's best point, so the table reports a budget decided in advance.
SHAPE_REGIME = "low"
SHAPE_BUDGET = 8.0


def read(path):
    with open(os.path.join(HERE, path), newline="") as fh:
        return list(csv.DictReader(fh))


def at_budget(rows, budget):
    """Rows at the final checkpoint, indexed by policy and then by environment."""
    out = {}
    for r in rows:
        if abs(float(r["spent"]) - budget) < 1e-9:
            out.setdefault(r["policy"], {})[int(r["env"])] = r
    return out


def paired_t(ours, theirs):
    """Paired t of ``ours - theirs`` over shared environments.

    Sign convention follows the manuscript: negative favours our agent, so the
    difference is taken as ours minus theirs on a loss.
    """
    d = [a - b for a, b in zip(ours, theirs)]
    n = len(d)
    if n < 2:
        return float("nan")
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    if var <= 0.0:
        return 0.0
    return m / math.sqrt(var / n)


def column(by, policy, field, envs):
    return [float(by[policy][e][field]) for e in envs]


def ms_per_decision(by, policy, envs):
    return [1000.0 * float(by[policy][e]["inference_seconds"]) / float(by[policy][e]["rounds"])
            for e in envs]


def fmt_ms(v):
    """Milliseconds at two significant figures, with a floor so a fast agent is not 0.

    Returns a complete cell, since the floor case needs text mode for its inequality.
    """
    if v < 0.001:
        return "$<$\\,0.001"
    if v >= 10.0:
        return "$%d$" % round(v)
    if v >= 1.0:
        return "$%.1f$" % v
    if v >= 0.01:
        return "$%.2f$" % v
    return "$%.3f$" % v


def value_cell(value, t, digits, best):
    """A number, bolded if it leads its column outright, with its paired t as a superscript."""
    txt = ("%." + str(digits) + "f") % value
    body = "\\mathbf{%s}" % txt if best else txt
    if t is None:
        return "$%s$" % body
    return "$%s^{%+.1f}$" % (body, t)


def outright_leader(by, field, keys, envs):
    """The agent with the best mean, but only when it beats every other at ``|t| >= 2``.

    Bolding a mean that is a statistical tie would claim a win the data does not support,
    so a column whose leader ties anyone is left unbolded and the superscripts carry the
    comparison.
    """
    means = {k: sum(column(by, k, field, envs)) / len(envs) for k in keys}
    lead = min(means, key=means.get)
    for k in keys:
        if k == lead:
            continue
        t = paired_t(column(by, lead, field, envs), column(by, k, field, envs))
        if not (abs(t) >= 2.0):
            return None
    return lead


def sequential_table(csv_path, budget, columns, path, label_extra=None):
    """Build a criterion block and a mixing-law block from one sequential study."""
    by = at_budget(read(csv_path), budget)
    present = [(k, lab) for k, lab in CRITERIA if k in by]
    models = [(k, lab) for k, lab in MODELS if k in by]
    envs = sorted(set(by["eig"]))
    for k, _ in present + models:
        envs = [e for e in envs if e in by[k]]

    ref = {f: column(by, "eig", f, envs) for f, _, _ in columns}
    rows, cells = [], {}
    for key, lab in present + models:
        cells[key] = {}
        for field, _head, digits in columns:
            vals = column(by, key, field, envs)
            mean = sum(vals) / len(vals)
            t = None if key == "eig" else paired_t(ref[field], vals)
            cells[key][field] = (mean, t, digits)

    best = {}
    for field, _head, _d in columns:
        best[field] = outright_leader(by, field, [k for k, _ in present], envs)

    def block(items):
        out = []
        for key, lab in items:
            parts = []
            for field, _head, _d in columns:
                mean, t, digits = cells[key][field]
                parts.append(value_cell(mean, t, digits, best[field] == key))
            parts.append(fmt_ms(sum(ms_per_decision(by, key, envs)) / len(envs)))
            out.append("        %s & %s \\\\" % (lab, " & ".join(parts)))
        return out

    heads = " & ".join(h for _f, h, _d in columns)
    rows.append("    \\begin{tabular}{l%sr}" % ("c" * len(columns)))
    rows.append("        \\toprule")
    rows.append("        agent & %s & ms \\\\" % heads)
    rows.append("        \\midrule")
    rows.append("        \\multicolumn{%d}{l}{\\emph{acquisition, under the GIG-Poisson model}} \\\\"
                % (len(columns) + 2))
    rows += block(present)
    rows.append("        \\midrule")
    rows.append("        \\multicolumn{%d}{l}{\\emph{mixing law, under the EIG acquisition}} \\\\"
                % (len(columns) + 2))
    rows += block(models)
    rows.append("        \\bottomrule")
    rows.append("    \\end{tabular}")
    write(path, rows)
    return len(envs), [k for k, _ in present + models]


def rate_fields_table(path, source="rate-fields/results/rate_field_fits.csv"):
    """One row per field, the three laws across the columns.

    The earlier layout gave each law its own row and spanned ``n`` and the per-observation
    advantage over the three, which cost nine rows and two rules to say what three rows say.
    Laws in columns also puts the three $\\Delta$AIC values side by side, which is the
    comparison the section is about.
    """
    rows = read(source)
    fields = []
    for r in rows:
        if r["field"] not in [f for f, _ in fields]:
            fields.append((r["field"], []))
        dict(fields)[r["field"]].append(r)

    short = {"Fermi-LAT sources": "Fermi-LAT", "Safecast cells": "Safecast",
             "20 Newsgroups terms": "Newsgroups"}

    out = ["    \\begin{tabular}{lrrrrrr}", "        \\toprule",
           "        & & \\multicolumn{3}{c}{$\\Delta$AIC} & & \\\\",
           "        \\cmidrule(lr){3-5}",
           "        field & $n$ & gamma & lognorm. & GIG & order & per obs. \\\\",
           "        \\midrule"]
    for name, group in fields:
        rec = {r["law"]: r for r in group}
        n = int(float(group[0]["n"]))
        cells = []
        for key in ("gamma", "lognormal", "GIG"):
            d = float(rec[key]["delta_aic"])
            cells.append("$\\mathbf{0}$" if d == 0.0 else "$%d$" % round(d))
        order = "$%+.2f$" % float(
            rec["GIG"]["fitted"].split("order=")[1].split(",")[0])
        gain = "$%.2f$" % (float(rec["gamma"]["delta_aic"]) / n)
        out.append("        %s & $%d$ & %s & %s & %s & %s & %s \\\\"
                   % (short.get(name, name), n, cells[0], cells[1], cells[2],
                      order, gain))
    out += ["        \\bottomrule", "    \\end{tabular}"]
    write(path, out)
    return [short.get(n, n) for n, _ in fields]


def tail_mass_table(path):
    """What the third parameter moves, at a predictive mean and variance held fixed.

    Ratios against the moment-matched negative binomial, which is the law at the top of
    the attainable order range rather than one point inside it.
    """
    rows = read("tail-mass/results/tail_mass.csv")   # archived study
    setting = [r for r in rows if abs(float(r["mean"]) - 6.0) < 1e-9]
    near = int(float(setting[0]["tail_from"]))
    far_from = int(float(setting[0]["far_from"]))

    def far(r):
        v = float(r["far_tail_ratio_vs_negbinom"])
        return "%.0f" % v if v >= 10.0 else "%.1f" % v

    out = ["    \\begin{tabular}{rrrrr}", "        \\toprule",
           "        order & $\\omega$ & $p(0)$ & $p(y \\geq %d)$ & $p(y \\geq %d)$ \\\\"
           % (near, far_from),
           "        \\midrule"]
    for r in setting:
        if float(r["order_over_max"]) == 1.0:
            out.append("        \\midrule")
            out.append("        NB & $0$ & $1$ & $1$ & $1$ \\\\")
            continue
        out.append("        $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ & $%s$ \\\\" % (
            float(r["order"]), float(r["omega"]),
            float(r["zero_ratio_vs_negbinom"]), float(r["tail_ratio_vs_negbinom"]),
            far(r)))
    out += ["        \\bottomrule", "    \\end{tabular}"]
    write(path, out)
    return near, far_from


def slopes_table(path):
    """Fitted log-log slopes by regime, with the misspecified arm beside the correct one."""
    sl = read("synthetic/verification/results/prediction_slopes.csv")
    err = read("synthetic/verification/results/prediction_error.csv")
    order = ["light", "moderate", "heavy", "sparse"]

    got = {}
    for r in sl:
        got[(r["regime"], r["quantity"])] = float(r["loglog_slope"])
    vom, worst = {}, {}
    for r in err:
        g = r["regime"]
        vom[g] = float(r["omega"])            # placeholder, replaced below
        ratio = float(r["kl_negbinom_mean"]) / float(r["kl_mean"])
        worst[g] = max(worst.get(g, 0.0), ratio)
    # Prior variance-to-mean ratio, which is the regime label the manuscript uses.
    vm = {"light": "0.8", "moderate": "4.0", "heavy": "40", "sparse": "0.5"}

    out = ["    \\begin{tabular}{lrrrrr}", "        \\toprule",
           "        regime & $\\mathbb{V}/\\mathbb{E}$ & KL & rate err. & "
           "$\\mathbb{V}[\\lambda \\given \\mathcal{D}_n]$ & NB cost \\\\",
           "        \\midrule"]
    for g in order:
        out.append("        %s & $%s$ & $%.3f$ & $%.3f$ & $%.3f$ & $\\times %.2f$ \\\\"
                   % (g, vm[g], got[(g, "kl")], got[(g, "rate_error")],
                      got[(g, "post_var")], worst[g]))
    out += ["        \\midrule",
            "        theory & & $-1$ & $-1/2$ & $-1$ & \\\\",
            "        \\bottomrule", "    \\end{tabular}"]
    write(path, out)
    return {g: worst[g] for g in order}


def grid_table(path):
    """Held-out NLPD of each criterion relative to EIG under the same mixing law."""
    by = at_budget(read("bulk-sampling/results/sequential.csv"), 240.0)
    models = [("", "GIG-Poisson"), ("@gamma-poisson", "gamma-Poisson"),
              ("@lognormal-poisson", "lognormal-Poisson")]
    order = ["epig", "d-optimality", "neyman", "epistemic", "predictive-variance",
             "thompson", "maxent"]
    labels = {"epig": "EPIG", "d-optimality": "D-optimality", "neyman": "Neyman allocation",
              "epistemic": "epistemic variance", "predictive-variance": "total variance",
              "thompson": "Thompson sampling", "maxent": "maximum entropy"}

    out = ["    \\begin{tabular}{lrrr}", "        \\toprule",
           "        criterion & %s \\\\" % " & ".join(lab for _s, lab in models),
           "        \\midrule"]
    for crit in order:
        cells = []
        for suffix, _lab in models:
            key, ref = crit + suffix, "eig" + suffix
            if key not in by or ref not in by:
                cells.append("not run")
                continue
            envs = sorted(set(by[key]) & set(by[ref]))
            mk = sum(float(by[key][e]["nlpd"]) for e in envs) / len(envs)
            mr = sum(float(by[ref][e]["nlpd"]) for e in envs) / len(envs)
            cells.append("$%+.3f$" % (mk - mr))
        out.append("        %s & %s \\\\" % (labels[crit], " & ".join(cells)))
    out += ["        \\bottomrule", "    \\end{tabular}"]
    write(path, out)


def shape_scarcity_table(regime, budget, path):
    """Criterion block for the shape-and-scarcity study at one point of its sweep.

    Same layout as the other two allocation tables, with one column added: the share of the
    budget each criterion sent to the heavy-shape class. That column is the mechanism, and
    without it the accuracy differences look arbitrary rather than earned.
    """
    by = {}
    for r in read("synthetic/shape-scarcity/results/episodes.csv"):
        if r["regime"] == regime and abs(float(r["budget"]) - budget) < 1e-9:
            by.setdefault(r["agent"], {})[int(r["episode"])] = r
    present = [(k, lab) for k, lab in CRITERIA if k in by]
    envs = sorted(set(by["eig"]))
    for k, _ in present:
        envs = [e for e in envs if e in by[k]]

    columns = [("nlpd", "NLPD", 3), ("dex", "dex", 3), ("regret", "regret", 3)]
    ref = {f: column(by, "eig", f, envs) for f, _h, _d in columns}
    best = {f: outright_leader(by, f, [k for k, _ in present], envs)
            for f, _h, _d in columns}

    rows = ["    \\begin{tabular}{lcccrr}", "        \\toprule",
            "        agent & NLPD & dex & regret & to heavy & ms \\\\", "        \\midrule"]
    for key, lab in present:
        parts = []
        for field, _h, digits in columns:
            vals = column(by, key, field, envs)
            mean = sum(vals) / len(vals)
            t = None if key == "eig" else paired_t(ref[field], vals)
            parts.append(value_cell(mean, t, digits, best[field] == key))
        share = 100.0 * sum(column(by, key, "heavy_share", envs)) / len(envs)
        parts.append("$%.0f$\\,\\%%" % share)
        parts.append(fmt_ms(sum(column(by, key, "ms", envs)) / len(envs)))
        rows.append("        %s & %s \\\\" % (lab, " & ".join(parts)))
    rows += ["        \\bottomrule", "    \\end{tabular}"]
    write(path, rows)
    return len(envs), len(present)


def write(path, lines):
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    full = os.path.join(OUT, path)
    with open(full, "w") as fh:
        fh.write("% Generated by experiments/make_tables.py. Do not edit by hand.\n")
        fh.write("\n".join(lines) + "\n")
    print("  wrote tables/%s" % path)


def main():
    print("Regenerating the paper's results tables from the recorded runs.\n")
    n, ags = sequential_table(
        "bulk-sampling/results/sequential.csv", 240.0,
        [("nlpd", "NLPD", 3), ("regret", "regret", 1)],
        "bulk.tex")
    print("    bulk: %d agents over %d properties" % (len(ags), n))

    n, ags = sequential_table(
        "gamma-ray/results/sequential.csv", 120.0,
        [("nlpd", "NLPD", 3), ("regret", "regret", 1)],
        "photon.tex")
    print("    photon: %d agents over %d programmes" % (len(ags), n))

    fields = rate_fields_table("rate_fields.tex")
    print("    rate fields: %s" % ", ".join(fields))

    news_csv = os.path.join(HERE, "newsgroups", "results",
                            "newsgroup_fits.csv")
    if os.path.isfile(news_csv):
        got = rate_fields_table("newsgroups.tex",
                                "newsgroups/results/newsgroup_fits.csv")
        print("    newsgroups: %s" % ", ".join(got))

    # The tail-mass study is archived: the manuscript no longer inputs its table. The
    # builder is kept in case the study is restored.
    if os.path.isdir(os.path.join(HERE, "tail-mass")):
        near, far = tail_mass_table("tail_mass.tex")
        print("    tail mass: ratios at y >= %d and y >= %d" % (near, far))

    worst = slopes_table("slopes.tex")
    print("    slopes: misspecification costs up to %s"
          % ", ".join("%s x%.2f" % (g, w) for g, w in worst.items()))

    grid_table("grid.tex")

    if os.path.isfile(os.path.join(HERE, "synthetic/shape-scarcity",
                                   "results", "episodes.csv")):
        n, k = shape_scarcity_table(SHAPE_REGIME, SHAPE_BUDGET, "shape_scarcity.tex")
        print("    shape scarcity: %d criteria over %d episodes at the %s-count regime, "
              "budget %g" % (k, n, SHAPE_REGIME, SHAPE_BUDGET))

    print("\nDone. The manuscript inputs these from experiments/tables/.")


if __name__ == "__main__":
    main()
