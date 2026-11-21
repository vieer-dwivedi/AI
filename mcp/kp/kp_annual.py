"""KP annual chart (solar return), progress.md backlog item 25. Facts only.

The return moment is when the transiting Sun reaches its natal sidereal longitude (same ayanamsa). KP practice casts it for the
place of residence in that year (not necessarily the birth place) and reads it with cusp sub lords and the natal significators.
Tajika yogas, Muntha and Mudda dasha are NOT KP and are not computed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import kp_chart as ch
from kp_core import Ctx, KPError, fmt_dt
from kp_tables import kp_lords


def _w180(x: float) -> float:
    return ((x + 180.0) % 360.0) - 180.0


def solve_return(ctx: Ctx, natal_sun: float, year: int, birth_dt: datetime) -> float:
    """Julian day (UT) in `year` when the sidereal Sun equals `natal_sun` (bisection, +-4 days around the birthday)."""
    if not 1800 <= year <= 2300:
        raise KPError("year must be between 1800 and 2300.")
    try:
        approx = birth_dt.replace(year=year)
    except ValueError:  # 29 Feb
        approx = birth_dt.replace(year=year, day=28)
    c = ctx.jd_from_local(approx)
    lo, hi = c - 4.0, c + 4.0
    f = lambda jd: _w180(ctx.lon("Sun", jd) - natal_sun)
    if f(lo) > 0 or f(hi) < 0:
        raise KPError("Could not bracket the solar return (unexpected).")
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def compare(natal, ret) -> dict[str, Any]:
    rows = []
    for i in range(12):
        a, b = kp_lords(natal.cusps[i]), kp_lords(ret.cusps[i])
        rows.append({"cusp": i + 1, "natal_sub_lord": a["sub_lord"], "return_sub_lord": b["sub_lord"], "same_sub_lord": a["sub_lord"] == b["sub_lord"],
                     "return_cusp_sign": b["sign"], "return_cusp_star_lord": b["star_lord"]})
    pl = []
    for p, d in ret.planets.items():
        n = natal.planets.get(p)
        if n is None:
            continue
        pl.append({"planet": p, "return_house": d["house"], "natal_house": n["house"], "return_sign": d["sign"], "natal_sign": n["sign"],
                   "return_star_lord": d["star_lord"], "return_sub_lord": d["sub_lord"], "same_sign": d["sign"] == n["sign"],
                   "same_star_lord": d["star_lord"] == n["star_lord"]})
    return {"cusps": rows, "planets": pl}
