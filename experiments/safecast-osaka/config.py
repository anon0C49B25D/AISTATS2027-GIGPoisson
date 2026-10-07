"""Region of the Osaka study: natural background radiation only.

Osaka lies 583 km from Fukushima Daiichi, too far for the 2011 fallout to raise its ambient
dose rate measurably, so a cell's rate is the natural background of the ground and the
building materials around it, plus the cosmic-ray component. That rate does not change over the years, so logs of every year are
used.
"""

CONFIG = dict(
    name="Osaka, 30 km, natural background",
    centre=(34.69, 135.50),
    radius_km=30.0,
    after=None,
    before=None,
    bands=[0.0, 7.5, 15.0, 22.5, 30.0],
    cell_m=100.0,
    discover=(30000, 30),
)

#: Figure options: every law is drawn.
FIGURES = dict(offscale=[])
