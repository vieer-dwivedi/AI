"""DBA: 'true_solar_arc' year mode (the Sun-return year; PyJHora's default, see progress.md V1).

Definition (own, exact in the dasha coordinate, numerical in the clock):
  A dasha year is one revolution of the Sun (360 deg of ecliptic longitude). The dasha coordinate tau
  (years since birth, negative before birth) maps to the instant at which the Sun has travelled
  360 x tau degrees from its birth longitude. The map is additive (arcs add), so every boundary of
  every level is a function of its tau alone: a child boundary is the same instant whichever parent
  it is reached through, children sum to the parent exactly, and sub-periods are proportional in
  Sun ARC (and therefore in dasha years), not in days. In days they vary with the Sun's speed
  (0.953 to 1.019 deg/day), by a few days at the Antar level.

Variants (explicit option, returned in provenance):
  true_solar_arc           tropical longitude (PyJHora TRUE_TROPICAL_YEAR, its default); mean year 365.24219 d
  true_solar_arc_sidereal  sidereal longitude, Lahiri (PyJHora TRUE_SIDEREAL_YEAR); mean year 365.256364 d

Known differences from PyJHora (see test/diff_solararc.py for the measured size):
  * PyJHora chains each boundary from the previous one and clamps the last child to the parent end;
    here every boundary comes from one global map. Equal in exact arithmetic, a little different in
    float round-off and where PyJHora falls back to proportional splitting (children under one day).
  * Moon-based balance at birth is unchanged (rule of three on the Moon's nakshatra fraction);
    only the clock that converts dasha years to days differs.
Boundaries are floats rounded to a Fraction (exact binary value); the Sun is solved to about 1e-9 deg,
well below the ephemeris accuracy (Moshier about 1 arcsec = 0.0003 d).
"""
from datetime import datetime
from fractions import Fraction

import swisseph as swe

from dba_core import to_fraction

MODES = ("true_solar_arc", "true_solar_arc_sidereal")


def _jd(utc: datetime):
    return swe.julday(utc.year, utc.month, utc.day,
                      utc.hour + utc.minute / 60.0 + utc.second / 3600.0 + utc.microsecond / 3.6e9, swe.GREG_CAL)


class SunArcMap:
    """Maps dasha years since birth (Fraction, may be negative) to days since birth (Fraction)."""

    def __init__(self, birth_utc: datetime, sidereal: bool = False):
        self.sidereal = bool(sidereal)
        self.name = MODES[1] if self.sidereal else MODES[0]
        self.nominal = Fraction("365.256364") if self.sidereal else Fraction("365.24219")
        self.jd0 = _jd(birth_utc)
        self._cache = {Fraction(0): Fraction(0)}
        self.lon0 = self._sun(0.0)[0]

    def __str__(self):
        return self.name

    __repr__ = __str__

    def _sun(self, t_days):
        jd = self.jd0 + t_days
        if self.sidereal:
            swe.set_sid_mode(swe.SIDM_LAHIRI)
            flag = swe.FLG_SWIEPH | swe.FLG_SIDEREAL | swe.FLG_SPEED
        else:
            flag = swe.FLG_SWIEPH | swe.FLG_TROPICAL | swe.FLG_SPEED
        res = swe.calc_ut(jd, swe.SUN, flag)[0]
        return res[0], res[3]

    def days(self, tau):
        """Days after birth at which the Sun has covered 360 x tau degrees since birth."""
        tau = to_fraction(tau)
        hit = self._cache.get(tau)
        if hit is not None:
            return hit
        target = (self.lon0 + float(tau * 360)) % 360.0
        t = float(tau * self.nominal)
        for _ in range(60):
            lon, speed = self._sun(t)
            err = (lon - target + 180.0) % 360.0 - 180.0
            step = err / speed
            t -= step
            if abs(step) < 1e-9:
                break
        else:
            raise RuntimeError("Sun-arc solve did not converge for tau=%s" % tau)
        out = Fraction(t)
        self._cache[tau] = out
        return out


def resolve_year_mode(mode, birth_utc=None):
    """Year option -> Fraction (fixed modes) or SunArcMap (true_solar_arc*, needs the birth instant)."""
    import dba_vimshottari as v
    if isinstance(mode, str) and mode.strip().lower() in MODES:
        if birth_utc is None:
            raise ValueError("%s needs the birth instant (utc datetime)" % mode)
        return SunArcMap(birth_utc, sidereal=mode.strip().lower().endswith("sidereal"))
    return v.year_days_from(mode)
