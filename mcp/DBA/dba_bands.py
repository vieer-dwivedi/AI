"""DBA H.38: uncertainty bands on rule windows. Facts only.

A rule's windows depend on conventions and on measurement error. This module recomputes them under a set of scenarios
(birth-time error of -sigma / 0 / +sigma minutes, alternative ayanamsa Moon longitudes, alternative year lengths) and reports
  possible : union of the windows over all scenarios (dates a window MIGHT cover)
  certain  : intersection over all scenarios (dates every scenario marks)
and the exact coverage of each, so a narrow rule can be seen to be fragile (certain much smaller than possible).
All times are exact Fractions of a day measured from the NOMINAL birth instant.
Birth-time error: the true birth instant is delta days away, the Moon is speed x delta away from its nominal longitude,
and the timeline of that scenario is shifted by delta.
"""
from fractions import Fraction

import dba_algebra as alg
import dba_vimshottari as vim
from dba_core import to_fraction


def make_scenarios(moon_deg, moon_speed_deg_per_day, sigma_min=0, alt_moon_degs=None, year_modes=None):
    """Cartesian product of Moon variants x birth-time shifts x year modes."""
    base = to_fraction(moon_deg)
    speed = to_fraction(moon_speed_deg_per_day)
    moons = [("nominal", base)] + [("alt_ayanamsa_%d" % i, to_fraction(m)) for i, m in enumerate(alt_moon_degs or [])]
    sig = to_fraction(sigma_min)
    shifts = [Fraction(0)] if sig == 0 else [-sig / 1440, Fraction(0), sig / 1440]
    modes = list(year_modes or [vim.DEFAULT_YEAR_MODE])
    out = []
    for name, m in moons:
        for d in shifts:
            for ym in modes:
                out.append({"label": "%s, birth %+.1f min, year %s" % (name, float(d * 1440), ym),
                            "moon": m + speed * d, "shift_days": d, "year_mode": ym})
    return out


def scenario_windows(sc, predicate, level, t_from, t_to, max_results=300000):
    """Windows (nominal-birth time axis) of one scenario, clipped to [t_from, t_to)."""
    lo, hi = to_fraction(t_from), to_fraction(t_to)
    d = sc["shift_days"]
    r = alg.find_periods(sc["moon"], predicate, level, lo - d, hi - d, sc["year_mode"], max_results)
    if r["truncated"]:
        raise ValueError("too many periods in scenario %s; use a coarser level" % sc["label"])
    shifted = [alg.Window(p.start + d, p.end + d, (p.path,)) for p in r["periods"]]
    return alg.intersect(alg.merge_adjacent(shifted), [alg.Window(lo, hi)])


def window_bands(predicate, level, t_from, t_to, scenarios):
    lo, hi = to_fraction(t_from), to_fraction(t_to)
    per = [scenario_windows(sc, predicate, level, lo, hi) for sc in scenarios]
    possible = alg.union(*per)
    certain = per[0]
    for w in per[1:]:
        certain = alg.intersect(certain, w)
    cov = [alg.coverage_fraction(w, lo, hi) for w in per]
    return {"possible": possible, "certain": certain, "per_scenario": per,
            "coverage_possible": alg.coverage_fraction(possible, lo, hi),
            "coverage_certain": alg.coverage_fraction(certain, lo, hi),
            "coverage_by_scenario": [(sc["label"], c) for sc, c in zip(scenarios, cov)]}


def robust_hit(per_scenario_windows, a, b):
    """'all' / 'some' / 'none': is the interval [a, b) overlapped by a window in every / some / no scenario?"""
    a, b = to_fraction(a), to_fraction(b)
    hits = [any(w.start < b and w.end > a for w in ws) for ws in per_scenario_windows]
    return "all" if all(hits) else ("some" if any(hits) else "none")
