"""Region of the Fukushima study: the contaminated field around Fukushima Daiichi.

Logs measured from 2024 on: by then caesium-134 (half-life 2.1 years) has decayed to about one
per cent of its 2011 activity, so a cell's rate changes by a few per cent a year rather than
by a third, and a cell can be treated as having one rate over the window.
"""

CONFIG = dict(
    name="Fukushima Daiichi, 60 km, 2024-2026",
    centre=(37.4211, 141.0328),
    radius_km=60.0,
    after="2024-01-01",
    before="2026-12-31",
    bands=[0.0, 10.0, 20.0, 40.0, 60.0],
    cell_m=100.0,
    discover=(60000, 400),
)

#: Figure options. The no-mixing Poisson is tens of nats behind here, which would flatten
#: every other line, so it is listed in the panels instead of drawn.
FIGURES = dict(offscale=["poisson"])
