"""The Safecast studies end to end: fetch, run and summarise one region.

``experiments/safecast-fukushima`` and ``experiments/safecast-osaka`` are the same study on two
populations of rates, so they share this driver and differ only in their ``config.py``:

    name        label used in printouts and figures
    centre      (lat, lon) of the study centre
    radius_km   readings further than this from the centre are dropped
    after       earliest measurement date of a log (ISO), or None for any date
    before      latest measurement date of a log, or None
    bands       class edges in km from the centre: a cell's class is its distance band,
                which a surveyor knows before going out
    cell_m      side of a cell, in metres
    discover    (radius_m, pages) for the measurement-API search that finds the logs

Each study writes only into its own ``data/``, ``results/`` and ``figures/``.
"""

import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import countfit                                                       # noqa: E402
import natural                                                        # noqa: E402
import safecast                                                       # noqa: E402

#: Readings per minute: one reading is a five-second count.
PER_MINUTE = 60.0 / safecast.READING_S


def _paths(here):
    out = {k: os.path.join(here, k) for k in ("data", "results", "figures")}
    for p in out.values():
        os.makedirs(p, exist_ok=True)
    return out


def band_labels(cfg):
    b = cfg["bands"]
    return ["{:g}-{:g} km".format(b[i], b[i + 1]) for i in range(len(b) - 1)]


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------

def fetch(cfg, here, discover=False):
    """Write ``data/imports.csv`` (the logs used) and ``data/readings.npz`` (every valid
    reading inside the study radius). Without ``discover``, an existing ``imports.csv`` is
    reused, so a rerun reads exactly the same logs."""
    p = _paths(here)
    fn_imp = os.path.join(p["data"], "imports.csv")
    if discover or not os.path.exists(fn_imp):
        print("searching the measurement API for drive logs near {} ...".format(cfg["name"]))
        radius_m, pages = cfg["discover"]
        dates = safecast.find_imports(cfg["centre"], radius_m, after=cfg.get("after"),
                                      max_pages=pages)
        lo, hi = cfg.get("after") or "0000", cfg.get("before") or "9999"
        dates = {i: d for i, d in dates.items() if lo <= d <= hi}
        with open(fn_imp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["import_id", "first_date"])
            for i in sorted(dates):
                w.writerow([i, dates[i]])
    imports = pd.read_csv(fn_imp)
    print("{}: {} drive logs; reading them (cache: {})".format(
        cfg["name"], len(imports), safecast.CACHE))
    t0 = time.time()
    r = safecast.read_logs(imports["import_id"].tolist(), cfg["centre"], cfg["radius_km"])
    if r["missing"]:
        print("  {} logs could not be read: {}".format(len(r["missing"]), r["missing"][:5]))
    np.savez_compressed(os.path.join(p["data"], "readings.npz"),
                        import_id=r["import"].astype(np.int32), t=r["t"].astype(np.int64),
                        count=r["count"].astype(np.int32), lat=r["lat"].astype(np.float64),
                        lon=r["lon"].astype(np.float64))
    print("  {} readings in {:.0f} s; wrote data/readings.npz".format(r["count"].size,
                                                                     time.time() - t0))


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

def load_contexts(cfg, here):
    """Readings -> quality control -> cells -> contexts with at least ``J_MAX + 1`` readings."""
    p = _paths(here)
    d = np.load(os.path.join(p["data"], "readings.npz"))
    rd = {"import": d["import_id"].astype(np.int64), "t": d["t"], "count": d["count"],
          "lat": d["lat"], "lon": d["lon"]}
    keep, dropped = safecast.qc(rd)
    rd = {k: v[keep] for k, v in rd.items()}
    cell, clat, clon = safecast.assign_cells(rd["lat"], rd["lon"], cfg["centre"],
                                             cfg["cell_m"])
    dist = safecast.haversine_km(rd["lat"], rd["lon"], *cfg["centre"])
    counts, classes, ids, dists = natural.build_cells(cell, rd["t"], rd["count"], dist,
                                                      cfg["bands"])
    info = dict(readings=int(keep.size), readings_kept=int(keep.sum()),
                logs=int(np.unique(d["import_id"]).size),
                logs_dropped=[int(i) for i in dropped], cells=int(clat.size),
                contexts=len(counts))
    table = pd.DataFrame(dict(cell=ids, lat=clat[ids], lon=clon[ids], dist_km=dists,
                              cls=classes, readings=[c.size for c in counts],
                              rate_cpm=[c.mean() * PER_MINUTE for c in counts]))
    return counts, classes, table, info


def run(cfg, here, episodes=natural.EPISODES, catalogues=(10, 20, 40, 100, 400),
        workers=None):
    p = _paths(here)
    t0 = time.time()
    print("=== {} ===".format(cfg["name"]))
    print("countfit gate ...")
    if countfit.gate(verbose=True):
        raise SystemExit("countfit gate failed: nothing is scored")
    counts, classes, table, info = load_contexts(cfg, here)
    labels = band_labels(cfg)
    print("  {readings} readings from {logs} logs; quality control dropped logs "
          "{logs_dropped}; {contexts} cells with >= {m} readings".format(m=natural.J_MAX + 1,
                                                                       **info))
    with open(os.path.join(p["results"], "info.json"), "w") as fh:
        json.dump(dict(info, config={k: v for k, v in cfg.items()}, episodes=episodes,
                       catalogues=list(catalogues), condition=list(natural.CONDITION),
                       hold=natural.HOLD), fh, indent=1)
    table.to_csv(os.path.join(p["results"], "cells.csv"), index=False)

    desc = natural.describe(counts, classes, labels, PER_MINUTE)
    desc.to_csv(os.path.join(p["results"], "describe.csv"), index=False)
    print(desc.round(3).to_string(index=False))
    rich = [(k, c) for k, c in zip(classes, counts) if c.size >= 10 and c.mean() > 0]
    pd.DataFrame(dict(cls=[k for k, _ in rich], readings=[c.size for _, c in rich],
                      mean=[c.mean() for _, c in rich],
                      var=[c.var(ddof=1) for _, c in rich])).to_csv(
        os.path.join(p["results"], "dispersion.csv"), index=False)

    print("population fits (every cell of a class) ...")
    pop = natural.population_fits(counts, classes, labels)
    pop.to_csv(os.path.join(p["results"], "population.csv"), index=False)
    print(pop[["cls", "law", "k", "loglik", "daic", "dbic"]].round(1).to_string(index=False))

    print("{} episodes, catalogues {} ...".format(episodes, list(catalogues)))
    # Each episode is saved as it finishes, so an interrupted run resumes; episodes saved by
    # an older version of the fitting or scoring code are recomputed.
    here_c = os.path.dirname(os.path.abspath(__file__))
    code = max(os.path.getmtime(os.path.join(here_c, f)) for f in ("countfit.py", "natural.py"))
    scores, fits = natural.run_episodes(counts, classes, len(labels), episodes=episodes,
                                        catalogues=catalogues, workers=workers,
                                        partial=os.path.join(p["results"], "partial"),
                                        fresh_after=code)
    scores.to_csv(os.path.join(p["results"], "scores.csv"), index=False)
    fits.to_csv(os.path.join(p["results"], "fits.csv"), index=False)
    summ = natural.summarise(scores)
    summ.to_csv(os.path.join(p["results"], "summary.csv"), index=False)
    show = summ[(summ.condition.isin([0, 5])) & (summ.law != natural.REFERENCE)]
    cols = ["catalogue", "condition", "score", "law", "diff", "diff_se", "ahead", "t",
            "significant"]
    print(show[cols].round(4).to_string(index=False))
    print("done in {:.0f} min".format((time.time() - t0) / 60.0))
