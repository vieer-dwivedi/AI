"""DBA exact Vimshottari core (own implementation, independent of PyJHora).

Facts only. Conventions are explicit options and are returned with every result:
  year_days  : length of a dasha 'year' in days (exact Fraction). Default 365.25 (Julian).
  Order/years: BPHS 46.12-15 (Krittika start, Sun 6 Moon 10 Mars 7 Rahu 18 Jupiter 16
               Saturn 19 Mercury 17 Ketu 7 Venus 20 = 120), same table as Ketu-first rotation.
  Balance    : rule of three applies to the FIRST Mahadasha only (BPHS 46.16).
  Children   : child length = parent length x child years / 120; first child = parent's lord.
  Boundaries : exact rationals; boundary belongs to the NEW period.
'true_solar_arc' / 'true_solar_arc_sidereal' (Sun-return year, PyJHora default, progress.md V1) need the birth
instant: build the option with dba_solararc.resolve_year_mode(mode, birth_utc) and pass the returned SunArcMap
wherever a year option is accepted. A bare string raises NotImplementedError.
"""
from fractions import Fraction
from math import floor

from dba_core import Period, to_fraction

LORDS = ("Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury")
YEARS = {"Ketu": 7, "Venus": 20, "Sun": 6, "Moon": 10, "Mars": 7,
         "Rahu": 18, "Jupiter": 16, "Saturn": 19, "Mercury": 17}
TOTAL_YEARS = 120
LEVEL_NAMES = ("Maha", "Antar", "Pratyantar", "Sookshma", "Prana", "Deha")
NAK_SPAN = Fraction(40, 3)          # 13 deg 20 min

YEAR_ALIASES = {
    "julian": "365.25",
    "savana": "360",
    "gregorian": "365.2425",
    "tropical": "365.24219",
    "sidereal": "365.256364",
}
DEFAULT_YEAR_MODE = "julian"


def is_map(x):
    """True for a non-linear year option (object with .days(tau) and .nominal, e.g. SunArcMap)."""
    return hasattr(x, "days") and callable(x.days) and hasattr(x, "nominal")


def nominal_year_days(yd):
    """Plain Fraction days per year for sizing loops (the mean year for a time map)."""
    return yd.nominal if is_map(yd) else yd


def year_days_from(mode=DEFAULT_YEAR_MODE):
    """Resolve a year option to an exact Fraction of days.

    Accepts an alias (julian, savana, gregorian, tropical, sidereal), 'fixed:<number>',
    or a plain number / numeric string. A time map (SunArcMap) is returned unchanged.
    """
    if is_map(mode):
        return mode
    if isinstance(mode, str):
        m = mode.strip().lower()
        if m in ("true_solar_arc", "true_solar_arc_sidereal"):
            raise NotImplementedError(
                "%s needs the birth instant: use dba_solararc.resolve_year_mode(mode, birth_utc)" % m)
        if m in YEAR_ALIASES:
            return to_fraction(YEAR_ALIASES[m])
        if m.startswith("fixed:"):
            return to_fraction(m[6:])
        return to_fraction(m)
    return to_fraction(mode)


def moon_start(moon_sidereal_deg):
    """(nakshatra index 0..26, Maha lord index into LORDS, fraction of the nakshatra elapsed)."""
    m = to_fraction(moon_sidereal_deg) % 360
    nak = floor(m / NAK_SPAN)
    frac = (m - nak * NAK_SPAN) / NAK_SPAN
    return nak, nak % 9, frac


def mahadashas(moon_sidereal_deg, year_days=DEFAULT_YEAR_MODE, count=9):
    """List of Maha periods. The first starts BEFORE birth by the elapsed share of its lord's years."""
    yd = year_days_from(year_days)
    _, i0, frac = moon_start(moon_sidereal_deg)
    if is_map(yd):
        tau = -frac * YEARS[LORDS[i0]]
        out = []
        for k in range(count):
            lord = LORDS[(i0 + k) % 9]
            nxt = tau + YEARS[lord]
            out.append(Period((lord,), yd.days(tau), yd.days(nxt), tau, nxt, yd))
            tau = nxt
        return out
    t = -frac * YEARS[LORDS[i0]] * yd
    out = []
    for k in range(count):
        lord = LORDS[(i0 + k) % 9]
        length = YEARS[lord] * yd
        out.append(Period((lord,), t, t + length))
        t += length
    return out


def children(parent):
    """Nine sub-periods of `parent`, parent's lord first, then the Vimshottari order. Exact."""
    i = LORDS.index(parent.lord)
    if parent.tmap is not None:
        tm, tau, out = parent.tmap, parent.tau_start, []
        span = parent.tau_end - parent.tau_start
        for k in range(9):
            lord = LORDS[(i + k) % 9]
            nxt = parent.tau_end if k == 8 else tau + span * YEARS[lord] / TOTAL_YEARS
            out.append(Period(parent.path + (lord,), tm.days(tau), tm.days(nxt), tau, nxt, tm))
            tau = nxt
        return out
    length = parent.duration
    t = parent.start
    out = []
    for k in range(9):
        lord = LORDS[(i + k) % 9]
        d = length * YEARS[lord] / TOTAL_YEARS
        out.append(Period(parent.path + (lord,), t, t + d))
        t += d
    return out


def ladder_at(moon_sidereal_deg, t, year_days=DEFAULT_YEAR_MODE, depth=6):
    """The running period at each level (Maha .. depth) at t days after birth (start inclusive)."""
    if not 1 <= depth <= len(LEVEL_NAMES):
        raise ValueError("depth must be 1..6")
    t = to_fraction(t)
    if t < 0:
        raise ValueError("t must be on or after birth")
    # Enough 9-Maha rounds to cover t (each round is 120 years of the chosen year length)
    yd = year_days_from(year_days)
    rounds = int(t / (TOTAL_YEARS * nominal_year_days(yd))) + 2
    current = None
    for p in mahadashas(moon_sidereal_deg, yd, count=9 * rounds):
        if p.contains(t):
            current = p
            break
    if current is None:
        raise ValueError("t outside generated range")
    ladder = [current]
    for _ in range(depth - 1):
        current = next(c for c in children(current) if c.contains(t))
        ladder.append(current)
    return ladder


def next_changes(ladder):
    """End instant of each running level (days from birth): when each level next changes."""
    return [p.end for p in ladder]
