"""Fetch the Swift-BAT 105-month hard X-ray survey: a second measured field of photon rates.

The catalogue lists, for each source detected in the 14-195 keV band over 105 months of
Swift's Burst Alert Telescope, a time-averaged flux and a source type from optical and
soft-X-ray follow-up (Seyfert 1 and 2, beamed AGN, X-ray binaries, cataclysmic variables,
and so on). The survey is all-sky and close to uniform in exposure, which is why it is used
here rather than the INTEGRAL/IBIS catalogue, whose exposure is concentrated on the Galactic
plane and many of whose sources are flagged as transients.

Source: Oh et al., *The 105-month Swift-BAT all-sky hard X-ray survey*, ApJS 235, 4 (2018),
served by VizieR as ``J/ApJS/235/4/table3``. Retrieved over the public TSV interface and
cached, so the study reruns offline.

As for 4FGL: the catalogue is flux-limited, so it is the detected population, and its
fluxes are estimates. Neither affects a comparison between mixing laws on the same numbers.

Run: python experiments/hard-xray/fetch_bat.py
"""

import os
import socket
import urllib.parse
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "swift_bat105.npz")
VIZIER = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
QUERY = {"-source": "J/ApJS/235/4/table3", "-out": "Flux,Type",
         "-out.max": "unlimited", "-out.form": "|"}


def fetch(force=False):
    """``(flux, type)``: 14-195 keV flux in 1e-15 W m^-2 (= 1e-12 erg cm^-2 s^-1), type string."""
    if os.path.exists(CACHE) and not force:
        d = np.load(CACHE, allow_pickle=True)
        return d["flux"], d["type"]
    socket.setdefaulttimeout(120)
    text = urllib.request.urlopen(VIZIER + "?" + urllib.parse.urlencode(QUERY)).read()
    rows = [l for l in text.decode("utf-8", "replace").splitlines()
            if l and not l.startswith("#")]
    flux, kind = [], []
    for line in rows[3:]:                      # header, units, separator
        parts = line.split("|")
        try:
            value = float(parts[0])
        except (ValueError, IndexError):
            continue
        flux.append(value)
        kind.append(parts[1].strip() if len(parts) > 1 else "")
    flux, kind = np.asarray(flux), np.asarray(kind, dtype=object)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, flux=flux, type=kind)
    return flux, kind


if __name__ == "__main__":
    f, t = fetch(force=True)
    print("cached {} sources to {}".format(f.size, os.path.relpath(CACHE)))
