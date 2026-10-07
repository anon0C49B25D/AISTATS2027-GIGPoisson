"""Fetch the Fermi-LAT 4FGL source catalogue: a real field of photon rates.

The catalogue lists, for each detected gamma-ray source, an integral photon flux above
1 GeV in photons per square centimetre per second. That is a *rate*, and the instrument
that measured it is a photon counter, so the counting model of Section 2 is not an analogy
here: it is the measurement process.

Source: Abdollahi et al., *Fermi Large Area Telescope Fourth Source Catalog*, ApJS 247, 33
(2020), served by VizieR as ``J/ApJS/247/33/4FGL``. Retrieved over the public TSV interface
and cached, so the study reruns offline.

**What the catalogue is not.** It is flux-limited: sources below the detection threshold are
absent, so the distribution here is the detected population rather than the intrinsic one,
and the fluxes are themselves estimates with uncertainty rather than exact rates. Neither
affects a comparison *between* mixing laws on the same numbers, which is what this
experiment does, but both would matter to any claim about the true luminosity function.

Run: python experiments/gamma-ray/fetch.py
"""

import os
import socket
import sys
import urllib.parse
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "fermi_4fgl_flux.npy")
VIZIER = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
QUERY = {"-source": "J/ApJS/247/33/4FGL", "-out": "F1000i,Cl1,Sig",
         "-out.max": "unlimited", "-out.form": "|"}
CACHE_CLASS = os.path.join(HERE, "data", "fermi_4fgl_class.npy")


def fetch(force=False, with_class=False):
    """Photon fluxes above 1 GeV, in photons per square centimetre per second.

    With ``with_class``, also returns the catalogue's source class for each entry. The class
    is what an observer knows about a target before pointing at it, and it is the covariate
    the observing study conditions its priors on: a pulsar and a blazar candidate are not the
    same prior problem, and treating them as one leaves nothing for an adaptive schedule to
    exploit.
    """
    if os.path.exists(CACHE) and os.path.exists(CACHE_CLASS) and not force:
        flux = np.load(CACHE)
        return (flux, np.load(CACHE_CLASS, allow_pickle=True)) if with_class else flux
    socket.setdefaulttimeout(120)
    url = VIZIER + "?" + urllib.parse.urlencode(QUERY)
    text = urllib.request.urlopen(url).read().decode("utf-8", "replace")
    rows = [l for l in text.splitlines() if l and not l.startswith("#")]
    flux, klass = [], []
    for line in rows[3:]:                      # header, units, separator
        parts = line.split("|")
        if len(parts) < 2:
            continue
        try:
            value = float(parts[0])
        except ValueError:
            continue
        if value <= 0:
            continue
        flux.append(value)
        # An empty class field means the source has no identified counterpart. Those are
        # the unassociated sources, which are a class in their own right and the ones a
        # follow-up campaign most wants to characterise.
        label = parts[1].strip().lower() if len(parts) > 1 else ""
        klass.append(label if label else "unassoc")
    flux = np.asarray(flux, dtype=float)
    klass = np.asarray(klass, dtype=object)
    np.save(CACHE, flux)
    np.save(CACHE_CLASS, klass, allow_pickle=True)
    return (flux, klass) if with_class else flux


if __name__ == "__main__":
    f, c = fetch(force="--force" in sys.argv, with_class=True)
    print("4FGL sources with a positive photon flux: {}".format(f.size))
    print("  photons cm^-2 s^-1 above 1 GeV")
    print("  min {:.3e}   median {:.3e}   max {:.3e}".format(
        f.min(), np.median(f), f.max()))
    print("  max / median = {:.0f},  skewness = {:.1f}".format(
        f.max() / np.median(f), float(((f - f.mean()) ** 3).mean() / f.std() ** 3)))
    import collections
    print("  source classes:")
    for label, n in collections.Counter(c).most_common(8):
        sub = f[c == label] * 1e10
        print("    {:<10} n={:<5d} median {:6.2f}  variance-to-mean {:9.1f}".format(
            label, n, float(np.median(sub)), float(sub.var() / sub.mean())))
    print("cached at {}".format(os.path.relpath(CACHE, HERE)))
