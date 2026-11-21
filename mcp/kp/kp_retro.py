"""Retrograde rules for KP judgments and station dates (progress.md backlog item 4; research note R16).

KP schools disagree on retrograde planets, so the rule is an option (`retrograde_rule`) and no rule ever hard-denies a matter:
  report_only      (default) only facts: which planets / star lords are retrograde.
  ksk_horary_cusp  the rule several teachers restrict to horary: applied to the CUSPAL sub lord only; triggered when the
                   CSL is retrograde itself or sits in the star of a retrograde planet. Effect = a delay flag.
  strict           the cusp rule above, plus: a dasha / bhukti / antara lord that is retrograde, or in the star of a
                   retrograde planet, flags the period as 'delayed: needs the planet to turn direct'.
Rahu and Ketu are always retrograde (mean node), so they never trigger these rules; Sun and Moon never are.
Retrograde status is the natal one (the chart's own speed sign). `find_stations` gives transit station dates so the caller can see when
a delayed result may mature (the planet turning direct in transit).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Optional

from kp_core import Ctx, KPError, fmt_dt
from kp_tables import kp_lords

NODES = ("Rahu", "Ketu")
RULES = ("report_only", "ksk_horary_cusp", "strict")
STATION_PLANETS = ("Mercury", "Venus", "Mars", "Jupiter", "Saturn")
SCAN_STEP_DAYS = {"Mercury": 4.0, "Venus": 10.0, "Mars": 15.0, "Jupiter": 20.0, "Saturn": 20.0}
MAX_WINDOW_DAYS = 365.25 * 60


def normalize_rule(value: Optional[str]) -> str:
    v = (value or "report_only").strip().lower().replace("-", "_").replace(" ", "_")
    v = {"report": "report_only", "none": "report_only", "ksk": "ksk_horary_cusp", "horary": "ksk_horary_cusp"}.get(v, v)
    if v not in RULES:
        raise ValueError(f"retrograde_rule must be one of {list(RULES)}")
    return v


def is_retro(chart: SimpleNamespace, planet: str) -> bool:
    return planet not in NODES and bool(chart.planets[planet]["retrograde"])


def cusp_check(chart: SimpleNamespace, csl: str, rule: str) -> dict[str, Any]:
    star = chart.star_of[csl]
    r_self, r_star = is_retro(chart, csl), is_retro(chart, star)
    applies = rule in ("ksk_horary_cusp", "strict")
    triggered = applies and (r_self or r_star)
    why = []
    if r_self:
        why.append(f"cuspal sub lord {csl} is retrograde")
    if r_star:
        why.append(f"cuspal sub lord {csl} is in the star of retrograde {star}")
    return {"rule": rule, "csl": csl, "csl_retrograde": r_self, "csl_star_lord": star, "csl_star_lord_retrograde": r_star,
            "triggered": triggered, "why": why if triggered else [],
            "effect": ("delay flag: per this rule the matter is held back until the retrograde planet turns direct; "
                       "the verdict is not changed and nothing is denied") if triggered else "",
            "facts_only": not applies}


def period_flags(chart: SimpleNamespace, lords: list[str], rule: str) -> dict[str, Any]:
    flags: list[dict[str, Any]] = []
    for lord in lords:
        star = chart.star_of[lord]
        if is_retro(chart, lord):
            flags.append({"lord": lord, "kind": "retrograde lord", "retrograde_planet": lord})
        if is_retro(chart, star) and star != lord:
            flags.append({"lord": lord, "kind": f"in the star of retrograde {star}", "retrograde_planet": star})
    return {"retrograde_flags": flags, "delayed": rule == "strict" and bool(flags)}


# --------------------------------------------------------------------------------------
# Station dates (transit): retrograde <-> direct
# --------------------------------------------------------------------------------------
def _scan(ctx: Ctx, planet: str, jd0: float, jd1: float) -> list[dict[str, Any]]:
    step = SCAN_STEP_DAYS[planet]
    out: list[dict[str, Any]] = []
    jd = jd0
    prev = ctx.lon_speed(planet, jd)[1]
    while jd < jd1:
        nxt = min(jd + step, jd1)
        sp = ctx.lon_speed(planet, nxt)[1]
        if (prev < 0) != (sp < 0):
            lo, hi = jd, nxt
            lo_neg = prev < 0
            for _ in range(60):
                mid = (lo + hi) / 2.0
                if (ctx.lon_speed(planet, mid)[1] < 0) == lo_neg:
                    lo = mid
                else:
                    hi = mid
            t = (lo + hi) / 2.0
            lon = ctx.lon(planet, t)
            r = kp_lords(lon)
            out.append({"station": "turns direct" if lo_neg else "turns retrograde", "datetime_local": fmt_dt(ctx.local_from_jd(t)),
                        "longitude_dms": r["dms"], "sign": r["sign"], "nakshatra": r["nakshatra"], "star_lord": r["star_lord"],
                        "sub_lord": r["sub_lord"]})
        prev = sp
        jd = nxt
    return out


def find_stations(ctx: Ctx, planet: str, start: datetime, end: datetime) -> dict[str, Any]:
    if planet in NODES:
        raise KPError("Rahu/Ketu have no stations (the mean node is always retrograde).")
    if planet not in STATION_PLANETS:
        raise KPError(f"planet must be one of {list(STATION_PLANETS)} (Sun and Moon are never retrograde).")
    jd0, jd1 = ctx.jd_from_local(start), ctx.jd_from_local(end)
    if jd1 <= jd0:
        raise KPError("`end` must be after `start`.")
    if jd1 - jd0 > MAX_WINDOW_DAYS:
        raise KPError("Window too long (max 60 years).")
    sp0 = ctx.lon_speed(planet, jd0)[1]
    stations = _scan(ctx, planet, jd0, jd1)
    nxt_direct = next((s for s in stations if s["station"] == "turns direct"), None)
    nxt_retro = next((s for s in stations if s["station"] == "turns retrograde"), None)
    beyond = False
    if (sp0 < 0 and nxt_direct is None) or (sp0 >= 0 and nxt_retro is None):
        more = _scan(ctx, planet, jd1, jd1 + 800.0)
        if sp0 < 0 and nxt_direct is None:
            nxt_direct = next((s for s in more if s["station"] == "turns direct"), None)
        if sp0 >= 0 and nxt_retro is None:
            nxt_retro = next((s for s in more if s["station"] == "turns retrograde"), None)
        beyond = True
    return {"planet": planet, "window": [fmt_dt(start), fmt_dt(end)], "retrograde_at_start": sp0 < 0,
            "speed_deg_per_day_at_start": round(sp0, 5), "count": len(stations), "stations": stations,
            "next_direct_station": nxt_direct, "next_retrograde_station": nxt_retro,
            "searched_beyond_window": beyond,
            "note": ("Station = the day the planet's apparent motion reverses (speed passes through zero), exact to about a second. "
                     "Under the strict retrograde rule a delayed result can mature once the planet has turned direct.")}
