"""Double transit (Jupiter + Saturn), progress.md backlog item 8. Facts only; no prediction.

School note: the "double transit" is a K.N. Rao / Parashari technique that many KP practitioners use beside KP. It is NOT
Krishnamurti's own rule, and it is a supporting condition, not a trigger: KP timing still needs the dasha lords to signify
the event houses (see find_event_periods).

Rule encoded: a natal house is "double transited" while BOTH Jupiter and Saturn occupy or aspect a target sign of that house.
  * Aspects are sign-based, counted from the transiting planet's own sidereal sign: every planet the 7th, Jupiter also the 5th
    and 9th, Saturn also the 3rd and 10th (same table as the node-agent and type-E code).
  * Target signs (`sign_scope`): 'cusp' = the sign of the house cusp (default); 'house' = every sign the unequal Placidus house
    touches. `include_lord` adds the natal sign of the house lord ("the house or its lord").
Retrograde handling: sign stretches follow the real longitude, so a planet that turns retrograde and slips back into a sign it
left shows up as a second stretch. The path is split at every station (the motion is monotonic between stations), so a short dip
over a sign boundary cannot be missed. The stationary-transit remark (events tend to cluster within about two months of a
Jupiter / Saturn station) is a heuristic and is only reported as nearby stations per window.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Optional

import kp_retro as rt
from kp_chart import ASPECTS
from kp_core import Ctx, KPError, fmt_dt
from kp_tables import HOUSE_THEMES, SIGNS, event_info

PAIR = ("Jupiter", "Saturn")
GRID_DAYS = 5.0
MAX_YEARS = 30.0
NEAR_STATION_DAYS = 60.0
SCOPES = ("cusp", "house")
SCHOOL = ("Double transit (K.N. Rao / Parashari technique used beside KP), not Krishnamurti's own rule. A supporting condition, "
          "not a trigger: the dasha lords must still signify the event houses.")


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}" + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def relation(planet: str, sign_now: int, target: int) -> Optional[str]:
    """How a transiting planet in sign `sign_now` touches sign `target`: 'occupies', 'aspects (Nth)' or None."""
    if sign_now == target:
        return "occupies"
    for off in ASPECTS.get(planet, (7,)):
        if (sign_now + off - 1) % 12 == target:
            return f"aspects ({_ordinal(off)})"
    return None


def resolve_houses(event: Optional[str], houses: Optional[list[int]], primary_only: bool) -> tuple[list[int], dict[str, Any]]:
    if houses:
        if any((not isinstance(h, int)) or not 1 <= h <= 12 for h in houses):
            raise KPError("houses must be integers 1..12.")
        return sorted(set(houses)), {"source": "houses"}
    if event:
        info = event_info(event)
        hs = [info["primary_cusp"]] if primary_only else sorted(set(info["favourable"]))
        return hs, {"source": "event", "event": info["key"], "label": info["label"], "primary_cusp": info["primary_cusp"],
                    "primary_only": primary_only}
    raise KPError("Give `event` or `houses`.")


def target_signs(chart: SimpleNamespace, house: int, scope: str, include_lord: bool) -> list[dict[str, Any]]:
    if scope not in SCOPES:
        raise KPError(f"sign_scope must be one of {list(SCOPES)}")
    c0 = chart.cusps[house - 1]
    order: list[int] = []
    why: dict[int, list[str]] = {}

    def add(s: int, reason: str) -> None:
        if s not in why:
            order.append(s)
            why[s] = []
        if reason not in why[s]:
            why[s].append(reason)

    add(int(c0 // 30.0) % 12, "sign of the house cusp")
    if scope == "house":
        width = (chart.cusps[house % 12] - c0) % 360.0 or 360.0
        for k in range(361):
            s = int(((c0 + width * k / 360.0) % 360.0) // 30.0) % 12
            add(s, "sign the house spans")
    if include_lord:
        lord = chart.owner[house]
        add(int(chart.planets[lord]["sign_index"]), f"natal sign of the house lord {lord}")
    return [{"sign_index": s, "sign": SIGNS[s], "why": why[s]} for s in order]


def _sign_of(ctx: Any, planet: str, jd: float) -> int:
    return int(ctx.lon(planet, jd) // 30.0) % 12


def sign_segments(ctx: Any, planet: str, jd0: float, jd1: float, station_jds: list[float], step: float = GRID_DAYS
                  ) -> list[tuple[float, float, int]]:
    """Constant-sign stretches [(jd_start, jd_end, sign_index)]. Sample points are a regular grid plus every station, so the path
    is monotonic between consecutive samples and a sign change between two samples is found exactly (bisection to 0.25 s)."""
    pts = {jd0, jd1}
    t = jd0 + step
    while t < jd1:
        pts.add(t)
        t += step
    pts.update(s for s in station_jds if jd0 < s < jd1)
    ordered = sorted(pts)
    segs: list[tuple[float, float, int]] = []
    seg_start, cur = jd0, _sign_of(ctx, planet, jd0)
    for a, b in zip(ordered, ordered[1:]):
        sb = _sign_of(ctx, planet, b)
        if sb == cur:
            continue
        lo, hi = a, b
        while hi - lo > 0.25 / 86400.0:
            mid = 0.5 * (lo + hi)
            if _sign_of(ctx, planet, mid) == cur:
                lo = mid
            else:
                hi = mid
        segs.append((seg_start, hi, cur))
        seg_start, cur = hi, sb
    segs.append((seg_start, jd1, cur))
    return segs


def _sign_at(segs: list[tuple[float, float, int]], jd: float) -> int:
    for a, b, s in segs:
        if a <= jd < b:
            return s
    return segs[-1][2]


def find_double_transit(ctx: Ctx, chart: SimpleNamespace, houses: list[int], source: dict[str, Any], start: datetime, end: datetime,
                        scope: str = "cusp", include_lord: bool = False, limit: int = 200) -> dict[str, Any]:
    jd0, jd1 = ctx.jd_from_local(start), ctx.jd_from_local(end)
    if jd1 <= jd0:
        raise KPError("`end` must be after `start`.")
    if (jd1 - jd0) / 365.25 > MAX_YEARS:
        raise KPError(f"Window too long (max {MAX_YEARS:.0f} years).")
    targets = {h: target_signs(chart, h, scope, include_lord) for h in houses}

    stations: dict[str, list[dict[str, Any]]] = {}
    station_jds: dict[str, list[float]] = {}
    margin = timedelta(days=NEAR_STATION_DAYS)
    for p in PAIR:
        st = rt.find_stations(ctx, p, start - margin, end + margin)["stations"]
        stations[p] = [{"station": x["station"], "datetime_local": x["datetime_local"], "sign": x["sign"], "nakshatra": x["nakshatra"]} for x in st]
        station_jds[p] = [ctx.jd_from_local(datetime.fromisoformat(x["datetime_local"])) for x in st]
    segs = {p: sign_segments(ctx, p, jd0, jd1, station_jds[p]) for p in PAIR}

    bps = sorted({jd0, jd1} | {a for p in PAIR for a, _, _ in segs[p] if jd0 < a < jd1})
    pieces = []
    for a, b in zip(bps, bps[1:]):
        if b - a < 1e-9:
            continue
        mid = 0.5 * (a + b)
        js, ss = _sign_at(segs["Jupiter"], mid), _sign_at(segs["Saturn"], mid)
        via: dict[int, list[dict[str, Any]]] = {}
        for h in houses:
            hit = []
            for t in targets[h]:
                jr, sr = relation("Jupiter", js, t["sign_index"]), relation("Saturn", ss, t["sign_index"])
                if jr and sr:
                    hit.append({"target_sign": t["sign"], "why": t["why"], "jupiter": jr, "saturn": sr})
            via[h] = hit
        pieces.append({"a": a, "b": b, "js": js, "ss": ss, "via": via})

    def iso(jd: float) -> str:
        return fmt_dt(ctx.local_from_jd(jd))

    st_all = [(p, x, j) for p in PAIR for x, j in zip(stations[p], station_jds[p])]
    house_rows = []
    flat = []
    for h in houses:
        wins: list[dict[str, Any]] = []
        cur: Optional[dict[str, Any]] = None
        for pc in pieces:
            if not pc["via"][h]:
                cur = None
                continue
            phase = {"start": iso(pc["a"]), "end": iso(pc["b"]),
                     "jupiter": {"sign": SIGNS[pc["js"]], "retrograde_at_start": ctx.lon_speed("Jupiter", pc["a"] + 1 / 86400.0)[1] < 0},
                     "saturn": {"sign": SIGNS[pc["ss"]], "retrograde_at_start": ctx.lon_speed("Saturn", pc["a"] + 1 / 86400.0)[1] < 0},
                     "via": pc["via"][h]}
            if cur is not None and abs(cur["_end"] - pc["a"]) < 1e-9:
                cur["phases"].append(phase)
                cur["_end"] = pc["b"]
            else:
                cur = {"_start": pc["a"], "_end": pc["b"], "phases": [phase]}
                wins.append(cur)
        out_wins = []
        for w in wins:
            near = [{"planet": p, "station": x["station"], "datetime_local": x["datetime_local"]} for p, x, j in st_all
                    if w["_start"] - NEAR_STATION_DAYS <= j <= w["_end"] + NEAR_STATION_DAYS]
            row = {"start": iso(w["_start"]), "end": iso(w["_end"]), "days": round(w["_end"] - w["_start"], 2),
                   "starts_at_window_start": abs(w["_start"] - jd0) < 1e-9, "continues_after_window_end": abs(w["_end"] - jd1) < 1e-9,
                   "phases": w["phases"], "stations_within_60_days_of_window": near}
            out_wins.append(row)
            flat.append({"house": h, "start": row["start"], "end": row["end"], "days": row["days"]})
        house_rows.append({"house": h, "theme": HOUSE_THEMES[h], "target_signs": targets[h], "window_count": len(out_wins), "windows": out_wins})
    flat.sort(key=lambda r: (r["start"], r["house"]))
    first = pieces[0] if pieces else None
    return {
        "school": SCHOOL, "source": source, "window": [fmt_dt(start), fmt_dt(end)], "sign_scope": scope, "include_lord": include_lord,
        "rule": ("A house counts while BOTH Jupiter and Saturn occupy or aspect a target sign of it. Aspects are sign-based from the "
                 "transiting planet's sidereal sign: all planets the 7th, Jupiter also 5th and 9th, Saturn also 3rd and 10th."),
        "at_start": {"jupiter": {"sign": SIGNS[first["js"]], "retrograde": ctx.lon_speed("Jupiter", jd0)[1] < 0},
                     "saturn": {"sign": SIGNS[first["ss"]], "retrograde": ctx.lon_speed("Saturn", jd0)[1] < 0},
                     "houses_double_transited": [h for h in houses if first["via"][h]]} if first else None,
        "houses": house_rows, "window_count": len(flat), "truncated": len(flat) > limit, "windows": flat[:limit],
        "stations": stations,
        "note": ("Stations are listed for the window plus 60 days on each side. Stationary-transit remark: events are said to cluster "
                 "within about two months of a Jupiter / Saturn station; reported as nearby stations only. Birth-time errors move "
                 "cusp signs only when a cusp is close to a sign boundary."),
    }
