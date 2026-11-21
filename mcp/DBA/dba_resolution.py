"""DBA A.4: resolution report. How far do birth-time error, year length and ayanamsa move the dates?
Facts only. All shifts are in days; boundary shift from a birth-time error is the same for every boundary of the timeline.
"""
from fractions import Fraction

from dba_core import to_fraction
import dba_vimshottari as v


def resolution_report(moon_deg, moon_speed_deg_per_day, t_days=0, birth_sigma_min=0,
                      year_modes=("julian",), moon_deg_alternatives=(), depth=6):
    m = to_fraction(moon_deg)
    speed = to_fraction(moon_speed_deg_per_day)
    t = to_fraction(t_days)
    sigma = to_fraction(birth_sigma_min)
    nak, i0, frac = v.moon_start(m)
    first_lord = v.LORDS[i0]
    base_yd = v.nominal_year_days(v.year_days_from(year_modes[0]))

    shift_per_min = speed / 1440 / v.NAK_SPAN * v.YEARS[first_lord] * base_yd
    to_start = frac * v.NAK_SPAN / speed * 1440          # minutes since the Moon entered its nakshatra
    to_end = (1 - frac) * v.NAK_SPAN / speed * 1440      # minutes until it leaves
    nearest = min(to_start, to_end)
    side = "entered_earlier" if to_start <= to_end else "leaves_later"

    def ladder(mm, mode):
        return v.ladder_at(mm, t, mode, depth)

    base = ladder(m, year_modes[0])
    yr_spread = [Fraction(0)] * depth
    if len(year_modes) > 1:
        lads = [ladder(m, md) for md in year_modes]
        for k in range(depth):
            starts = [l[k].start for l in lads]
            yr_spread[k] = max(starts) - min(starts)
    ay_spread = [Fraction(0)] * depth
    if moon_deg_alternatives:
        lads = [base] + [ladder(to_fraction(a), year_modes[0]) for a in moon_deg_alternatives]
        for k in range(depth):
            starts = [l[k].start for l in lads]
            ay_spread[k] = max(starts) - min(starts)
    time_shift = shift_per_min * sigma

    levels = []
    for k, p in enumerate(base):
        res = max(time_shift, yr_spread[k], ay_spread[k])
        levels.append({
            "level": v.LEVEL_NAMES[k], "lord": p.lord, "duration_days": float(p.duration),
            "boundary_shift_days": {"birth_time": float(time_shift), "year_length": float(yr_spread[k]),
                                    "ayanamsa": float(ay_spread[k])},
            "below_resolution": bool(p.duration < res),
        })
    return {
        "first_maha_lord": first_lord,
        "nakshatra_index": nak,
        "shift_days_per_minute": float(shift_per_min),
        "moon_minutes_to_nakshatra_boundary": float(nearest),
        "nearest_boundary": side,
        "stable_for_minutes": float(nearest),
        "first_maha_lord_flips_within_sigma": bool(nearest <= sigma),
        "birth_sigma_min": float(sigma),
        "options_used": {"year_modes": [str(m) for m in year_modes], "moon_alternatives": len(moon_deg_alternatives)},
        "levels": levels,
    }
