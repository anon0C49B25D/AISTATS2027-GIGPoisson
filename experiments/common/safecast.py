"""Safecast bGeigie drive logs: Geiger counts measured in the world, at a known exposure.

Safecast is a citizen-science radiation survey whose data are released into the public domain
(CC0). Its bGeigie Nano carries an LND 7317 pancake Geiger-Mueller tube and a GPS receiver,
and is driven around on a car; every five seconds it writes one line of its log,

    $BNRDD,<device>,<UTC time>,<CPM>,<pulses in last 5 s>,<total pulses>,<A|V>,
           <lat ddmm.mmmm>,<N|S>,<lon dddmm.mmmm>,<E|W>,<altitude>,<A|V>,<hdop>,<fix>*<checksum>

The fifth field is the number of tube pulses in the last five seconds: a count, at an exposure
of five seconds, from a process whose physics is Poisson. That is the observation model of the
paper with nothing simulated. The CPM field is a moving sum over the last twelve such windows,
so successive values overlap and are not independent; it is not used.

**Fetching.** The measurement API is asked for measurements within a radius of a centre and,
optionally, after a date. The drive logs (imports) those measurements came from are then
downloaded whole from Safecast's public bucket and parsed. A log is about half a megabyte, so
the raw logs are cached outside the repository (``SAFECAST_CACHE``, default
``~/.cache/safecast-bgeigie``); a study stores only the parsed readings and the list of logs,
which is enough to rerun it offline and to refetch exactly the same logs.

**What is kept.** A line is kept when both its radiation and its GPS flags read ``A`` (valid),
its timestamp is not a repeat of the previous line's, and it lies within the study radius.

**Quality control.** One rule, applied identically in every study: a log whose median five-second
count inside the study region exceeds three times the median over all logs is dropped. Such a
log was not measuring the ambient field (a check source, a sample on the passenger seat); the
rule is relative, so it cannot remove a genuinely hot region that several drives pass through.
"""

import ast
import json
import os
import socket
import time
import urllib.parse
import urllib.request

import numpy as np

__all__ = ["find_imports", "download_log", "parse_log", "read_logs", "qc", "assign_cells",
           "haversine_km", "CACHE"]

API = "https://api.safecast.org/measurements.json"
IMPORT_API = "https://api.safecast.org/bgeigie_imports/{}.json"
HEADERS = {"User-Agent": "activecounting-research (python urllib)"}
CACHE = os.environ.get("SAFECAST_CACHE",
                       os.path.join(os.path.expanduser("~"), ".cache", "safecast-bgeigie"))
#: Seconds per reading: the exposure of one count.
READING_S = 5.0
EPOCH = np.datetime64("2011-01-01T00:00:00", "s")


def _get(url, tries=5):
    err = None
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS)).read()
        except Exception as e:                  # the API returns intermittent 5xx
            err = e
            time.sleep(3.0 * (attempt + 1))
    raise err


def find_imports(centre, radius_m, after=None, max_pages=400, verbose=True):
    """Drive logs with measurements within ``radius_m`` of ``centre`` (and after ``after``).

    Returns ``{import_id: earliest captured date seen}``. The API pages through measurements
    rather than logs, so this samples the logs by how many measurements they hold near the
    centre; it is not guaranteed to find every log, and :mod:`fetch` scripts store the list
    they used so that a rerun reads the same ones.
    """
    socket.setdefaulttimeout(120)
    dates = {}
    for page in range(1, max_pages + 1):
        q = {"latitude": centre[0], "longitude": centre[1], "distance": int(radius_m),
             "per_page": 1000, "page": page}
        if after:
            q["captured_after"] = after
        try:
            recs = json.loads(_get(API + "?" + urllib.parse.urlencode(q)))
        except Exception as e:
            if verbose:
                print("  page {} failed: {}".format(page, e))
            continue
        if not recs:
            break
        for r in recs:
            i = r.get("measurement_import_id")
            if i is not None:
                d = str(r.get("captured_at", ""))[:10]
                dates[int(i)] = min(dates.get(int(i), d), d)
        if verbose and page % 10 == 0:
            print("  page {}: {} logs".format(page, len(dates)), flush=True)
    return dates


def download_log(import_id, cache=CACHE):
    """The raw text of one drive log, from the cache or Safecast's bucket."""
    os.makedirs(cache, exist_ok=True)
    fn = os.path.join(cache, "{}.log".format(int(import_id)))
    if not os.path.exists(fn):
        meta = json.loads(_get(IMPORT_API.format(int(import_id))))
        src = meta["source"]
        url = src["url"] if isinstance(src, dict) else ast.literal_eval(src)["url"]
        data = _get(url)
        with open(fn + ".part", "wb") as fh:
            fh.write(data)
        os.replace(fn + ".part", fn)
    with open(fn, "rb") as fh:
        return fh.read().decode("utf-8", "replace")


def _degrees(v, hemi):
    v = float(v)
    d = int(v // 100)
    out = d + (v - 100.0 * d) / 60.0
    return -out if hemi in ("S", "W") else out


def parse_log(text):
    """``(seconds since 2011-01-01, count in 5 s, lat, lon)`` of every valid line."""
    t, c, la, lo = [], [], [], []
    prev = None
    for line in text.splitlines():
        if not line.startswith("$B"):
            continue
        p = line.split("*")[0].split(",")
        if len(p) < 13 or not p[0].endswith("RDD"):
            continue
        if p[6] != "A" or p[12] != "A" or p[2] == prev:
            continue
        try:
            ts = np.datetime64(p[2].replace("Z", ""), "s")
            count = int(p[4])
            lat, lon = _degrees(p[7], p[8]), _degrees(p[9], p[10])
        except (ValueError, IndexError):
            continue
        if count < 0 or not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue
        prev = p[2]
        t.append(int((ts - EPOCH).astype(np.int64)))
        c.append(count)
        la.append(lat)
        lo.append(lon)
    return (np.asarray(t, np.int64), np.asarray(c, np.int64), np.asarray(la, float),
            np.asarray(lo, float))


def haversine_km(lat, lon, lat0, lon0):
    p = np.pi / 180.0
    a = (np.sin((lat - lat0) * p / 2.0) ** 2
         + np.cos(lat * p) * np.cos(lat0 * p) * np.sin((lon - lon0) * p / 2.0) ** 2)
    return 12742.0 * np.arcsin(np.sqrt(a))


def read_logs(import_ids, centre, radius_km, cache=CACHE, verbose=True):
    """Every valid reading of the given logs that lies within ``radius_km`` of ``centre``.

    Returns a dict of equal-length arrays: ``import`` (log id), ``t`` (seconds since
    2011-01-01, from the device clock), ``count`` (pulses in 5 s), ``lat``, ``lon``.
    """
    cols = {k: [] for k in ("import", "t", "count", "lat", "lon")}
    missing = []
    for n, i in enumerate(import_ids):
        try:
            t, c, la, lo = parse_log(download_log(i, cache))
        except Exception as e:
            missing.append((int(i), repr(e)[:80]))
            continue
        if t.size == 0:
            continue
        ok = haversine_km(la, lo, centre[0], centre[1]) < radius_km
        cols["import"].append(np.full(ok.sum(), int(i), np.int64))
        cols["t"].append(t[ok])
        cols["count"].append(c[ok])
        cols["lat"].append(la[ok])
        cols["lon"].append(lo[ok])
        if verbose and (n + 1) % 50 == 0:
            print("  parsed {} of {} logs".format(n + 1, len(import_ids)), flush=True)
    out = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}
    out["missing"] = missing
    return out


def qc(readings, factor=3.0):
    """Keep-mask of the readings, and the ids of the logs the rule drops (see module doc)."""
    imp = readings["import"]
    ids = np.unique(imp)
    med = np.array([np.median(readings["count"][imp == i]) for i in ids])
    bad = ids[med > factor * np.median(med)]
    return ~np.isin(imp, bad), bad


def assign_cells(lat, lon, centre, cell_m=100.0):
    """Integer cell index of each reading on a ``cell_m`` grid, and each cell's centre.

    The grid is in degrees, with the longitude spacing stretched by ``1 / cos(latitude)`` of
    the study centre, so that cells are close to square on the ground over a study region.
    """
    dlat = cell_m / 111_000.0
    dlon = dlat / np.cos(np.radians(centre[0]))
    ij = np.stack([np.round(lat / dlat), np.round(lon / dlon)], axis=1).astype(np.int64)
    uniq, cell = np.unique(ij, axis=0, return_inverse=True)
    return cell.ravel(), uniq[:, 0] * dlat, uniq[:, 1] * dlon
