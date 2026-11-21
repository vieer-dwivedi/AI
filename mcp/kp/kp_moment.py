"""KP work on a moment in time: ruling planets, horary (1-249), transit snapshot, boundary / trigger searches, rectification.

Rule-based only: every function returns the facts and the classical KP rule it applied; none predicts an outcome.
Dates are local clock time at the place used. Planet names: Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn, Rahu, Ketu.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Callable, Optional

from kp_chart import NODES, chart_at, house_of_lon, judge_event, signified
from kp_core import Ctx, KPError, fmt_dt, parse_dt
from kp_dasha import Dasha, DashaError, running_ladder
from kp_rectify import ANCHOR_EVENTS, ANCHOR_NOTE, MOON_LINK_MODES, event_weight, lagna_moon_link
from kp_tables import PLANETS, division_row, kp_lords

# --------------------------------------------------------------------------------------
# Ruling planets
# --------------------------------------------------------------------------------------
def ruling_planets(ctx: Ctx, dt: datetime, lat: float, lon: float, extended: bool = True,
                   natal: Optional[SimpleNamespace] = None, chart: Optional[SimpleNamespace] = None) -> dict[str, Any]:
    """Day lord + Moon and Lagna sign/star(/sub) lords at a moment. Lagna = the Ascendant of `chart` (horary passes its own)."""
    jd = ctx.jd_from_local(dt)
    ch = chart or chart_at(ctx, jd, lat, lon, dt)
    day = ctx.day_lord(dt)
    moon = ch.planets["Moon"]
    lag = kp_lords(ch.asc)
    entries = [("Day lord", day["day_lord"]), ("Moon sign lord", moon["sign_lord"]), ("Moon star lord", moon["star_lord"]),
               ("Lagna sign lord", lag["sign_lord"]), ("Lagna star lord", lag["star_lord"])]
    if extended:
        entries += [("Moon sub lord", moon["sub_lord"]), ("Lagna sub lord", lag["sub_lord"])]
    tally = Counter(p for _, p in entries)
    rp_set = set(tally)
    planets = []
    for p, n in tally.most_common():
        item: dict[str, Any] = {"planet": p, "count": n, "roles": [r for r, q in entries if q == p]}
        pd = ch.planets[p]
        item["moment_position"] = {"dms": pd["dms"], "sign": pd["sign"], "star_lord": pd["star_lord"], "sub_lord": pd["sub_lord"],
                                   "house": pd["house"], "retrograde": pd["retrograde"]}
        item["in_star_of_other_rp"] = pd["star_lord"] in rp_set and pd["star_lord"] != p
        item["in_sub_of_other_rp"] = pd["sub_lord"] in rp_set and pd["sub_lord"] != p
        item["in_own_sub"] = pd["sub_lord"] == p
        if p in NODES:
            item["represents"] = ch.agents[p]
        if natal is not None:
            item["natal_signifies"] = signified(natal, p)
        planets.append(item)
    top = tally.most_common(1)[0][1]
    return {
        "datetime_local": fmt_dt(dt), "day": day,
        "moon": {"dms": moon["dms"], "sign": moon["sign"], "sign_lord": moon["sign_lord"], "star_lord": moon["star_lord"],
                 "sub_lord": moon["sub_lord"], "nakshatra": moon["nakshatra"]},
        "lagna": {"dms": lag["dms"], "sign": lag["sign"], "sign_lord": lag["sign_lord"], "star_lord": lag["star_lord"],
                  "sub_lord": lag["sub_lord"], "nakshatra": lag["nakshatra"]},
        "extended_with_sub_lords": extended,
        "entries": [{"role": r, "planet": p} for r, p in entries],
        "ruling_planets": planets,
        "appearing_most": [p for p, n in tally.items() if n == top and top > 1],
        "notes": ["KP: a planet appearing several times, or placed in the star / sub of another ruling planet, is stronger.",
                  "Rahu/Ketu act for the planets they represent (see `represents`).",
                  "The Lagna changes about 1 degree every 4 minutes: its sub lord can change within minutes (see get_lagna_changes).",
                  "Day lord runs from sunrise to the next sunrise."],
    }


def ruling_planets_timeline(ctx: Ctx, dt: datetime, lat: float, lon: float, hours: float = 6.0, step_minutes: float = 5.0,
                            extended: bool = True) -> dict[str, Any]:
    """How the ruling planets change over the next hours (Lagna lords move fast; Moon lords slower; day lord at sunrise)."""
    if step_minutes < 1 or hours <= 0:
        raise KPError("hours must be positive and step_minutes at least 1.")
    n = int(hours * 60 // step_minutes)
    if n > MAX_CANDIDATES:
        raise KPError(f"Too many steps ({n}); use fewer hours or a larger step_minutes (max {MAX_CANDIDATES} steps).")
    out: list[dict[str, Any]] = []
    prev_entries: Optional[list[tuple[str, str]]] = None
    for i in range(n + 1):
        t = dt + timedelta(minutes=i * step_minutes)
        jd = ctx.jd_from_local(t)
        moon = kp_lords(ctx.lon("Moon", jd))
        lag = kp_lords(ctx.asc_only(jd, lat, lon))
        entries = [("Day lord", ctx.day_lord(t)["day_lord"]), ("Moon sign lord", moon["sign_lord"]), ("Moon star lord", moon["star_lord"]),
                   ("Lagna sign lord", lag["sign_lord"]), ("Lagna star lord", lag["star_lord"])]
        if extended:
            entries += [("Moon sub lord", moon["sub_lord"]), ("Lagna sub lord", lag["sub_lord"])]
        if prev_entries is not None and entries == prev_entries:
            out[-1]["last_seen"] = fmt_dt(t)
            continue
        tally = Counter(p for _, p in entries)
        out.append({"from": fmt_dt(t), "last_seen": fmt_dt(t), "entries": [{"role": r, "planet": p} for r, p in entries],
                    "ruling_planets": [{"planet": p, "count": c} for p, c in tally.most_common()],
                    "changed": [r for (r, p), (_, q) in zip(entries, prev_entries) if p != q] if prev_entries else []})
        prev_entries = entries
    for a, b in zip(out, out[1:]):
        a["until_about"] = b["from"]
    out[-1]["until_about"] = fmt_dt(dt + timedelta(minutes=n * step_minutes))
    return {"start": fmt_dt(dt), "hours": hours, "step_minutes": step_minutes, "extended_with_sub_lords": extended,
            "stretches": out,
            "note": "Change times are accurate to the step size. Use find_boundary_changes / get_lagna_changes for exact times."}


# --------------------------------------------------------------------------------------
# Horary (Prashna)
# --------------------------------------------------------------------------------------
LAGNA_POINTS = ("start", "middle", "end", "actual_if_inside")


def cast_horary(ctx: Ctx, number: int, dt: datetime, lat: float, lon: float, lagna_point: str = "start"
                ) -> tuple[SimpleNamespace, dict[str, Any]]:
    """KP horary chart: the number fixes the Lagna; planets are those of the question moment."""
    if lagna_point not in LAGNA_POINTS:
        raise KPError(f"lagna_point must be one of {list(LAGNA_POINTS)}")
    try:
        row = division_row(number)
    except ValueError as exc:
        raise KPError(str(exc)) from exc
    s, e = row["start"], row["end"]
    jd_q = ctx.jd_from_local(dt)
    actual = ctx.asc_only(jd_q, lat, lon)
    inside = s <= actual < e
    eps = 1e-4
    if lagna_point == "start" or (lagna_point == "actual_if_inside" and not inside):
        target = s + eps
    elif lagna_point == "middle":
        target = (s + e) / 2.0
    elif lagna_point == "end":
        target = e - eps
    else:
        target = actual
    jd_h = ctx.solve_time_for_ascendant(target, jd_q, lat, lon)
    houses = ctx.cusps(jd_h, lat, lon)
    chart = chart_at(ctx, jd_q, lat, lon, dt, jd_houses=jd_h, houses=houses)
    got = kp_lords(chart.asc)
    info = {
        "number": number, "division": row, "lagna_point": lagna_point, "target_longitude": round(target % 360.0, 6),
        "cusp_time_local": fmt_dt(ctx.local_from_jd(jd_h)), "question_time_local": fmt_dt(dt),
        "actual_ascendant_at_question": {"dms": kp_lords(actual)["dms"], "kp_number": kp_lords(actual)["kp_number"],
                                          "inside_the_numbered_division": bool(inside)},
        "lagna_check": {"kp_number": got["kp_number"], "matches_number": got["kp_number"] == number},
        "method": ("The house cusps are those of the moment (same place, within one day of the question) at which the Ascendant "
                   "falls at the chosen point of the numbered division; planets are at the question moment."),
    }
    return chart, info


def judge_horary(chart: SimpleNamespace, rp: dict[str, Any], event: Optional[str] = None,
                 custom_favourable: Optional[list[int]] = None, custom_denial: Optional[list[int]] = None,
                 primary_cusp: Optional[int] = None, levels: Optional[str] = None) -> dict[str, Any]:
    j = judge_event(chart, event, custom_favourable, custom_denial, primary_cusp, levels)
    rps = {p["planet"] for p in rp["ruling_planets"]}
    csl = j["csl"]["sub_lord"]
    star = j["csl"]["star_lord"]
    fav = set(j["favourable_houses"])
    lv = levels or chart.opts.significator_levels
    moon_sub = chart.planets["Moon"]["sub_lord"]
    ms = set(signified(chart, moon_sub, lv))
    j["ruling_planet_check"] = {
        "ruling_planets": sorted(rps),
        "csl_is_ruling_planet": csl in rps,
        "csl_star_lord_is_ruling_planet": star in rps,
        "csl_own_sub_lord_is_ruling_planet": chart.planets[csl]["sub_lord"] in rps,
        "rule": "KP horary: the cuspal sub lord of the matter's house, confirmed by the ruling planets, shows the answer.",
    }
    j["moon_sub_lord_check"] = {"moon_sub_lord": moon_sub, "signifies": sorted(ms), "favourable": sorted(ms & fav),
                                "note": "Textbook check on the querent's state of mind / genuineness of the question."}
    return j


# --------------------------------------------------------------------------------------
# Transit snapshot
# --------------------------------------------------------------------------------------
def transit_snapshot(ctx: Ctx, dt: datetime, natal: Optional[SimpleNamespace] = None) -> dict[str, Any]:
    jd = ctx.jd_from_local(dt)
    pos = ctx.positions(jd)
    rows = []
    for name, p in pos.items():
        r = kp_lords(p["lon"])
        row = {"planet": name, "longitude": r["longitude"], "dms": r["dms"], "sign": r["sign"], "nakshatra": r["nakshatra"],
               "sign_lord": r["sign_lord"], "star_lord": r["star_lord"], "sub_lord": r["sub_lord"],
               "sub_sub_lord": r["sub_sub_lord"], "kp_number": r["kp_number"], "retrograde": p["retro"],
               "speed_deg_per_day": round(p["speed"], 5)}
        if natal is not None:
            row["natal_house"] = house_of_lon(p["lon"], natal.cusps)
            if r["star_lord"] in PLANETS:
                row["star_lord_natal_signifies"] = signified(natal, r["star_lord"])
            if r["sub_lord"] in PLANETS:
                row["sub_lord_natal_signifies"] = signified(natal, r["sub_lord"])
        rows.append(row)
    return {"datetime_local": fmt_dt(dt), "day": ctx.day_lord(dt), "planets": rows,
            "note": "Transit KP reading uses the sign lord and star lord (sub lord optional); see find_transit_triggers."}


# --------------------------------------------------------------------------------------
# Boundary (sign / star / sub / sub-sub) change search
# --------------------------------------------------------------------------------------
LEVEL_SPAN = {"sign": 30.0, "star": 40.0 / 3.0, "sub": 2.0 / 3.0, "sub_sub": 1.0 / 30.0}   # smallest span in degrees
LEVEL_LORD = {"sign": "sign_lord", "star": "star_lord", "sub": "sub_lord", "sub_sub": "sub_sub_lord"}
MAX_SPEED = {"Sun": 1.1, "Moon": 15.5, "Mercury": 2.4, "Venus": 1.4, "Mars": 0.9, "Jupiter": 0.3, "Saturn": 0.2,
             "Rahu": 0.5, "Ketu": 0.5}                                                        # deg/day, with margin
MAX_STEPS = 200_000


def _key(r: dict[str, Any], level: str) -> Any:
    if level == "sign":
        return r["sign_index"]
    if level == "star":
        return r["nakshatra_index"]
    if level == "sub":
        return (r["nakshatra_index"], r["sub_lord"])
    if level == "sub_sub":
        return (r["nakshatra_index"], r["sub_lord"], r["sub_sub_lord"])
    raise KPError(f"level must be one of {list(LEVEL_SPAN)}")


def _segments(lon_fn: Callable[[float], float], jd0: float, jd1: float, step: float, level: str
              ) -> list[tuple[float, float, dict[str, Any]]]:
    """Constant-key stretches of a moving longitude between jd0 and jd1: [(jd_start, jd_end, kp_lords at start)]."""
    if jd1 <= jd0:
        raise KPError("`end` must be after `start`.")
    if (jd1 - jd0) / step > MAX_STEPS:
        raise KPError("This window is too long for this level: narrow the dates or use a coarser level (sign / star).")

    def st(jd: float) -> tuple[Any, dict[str, Any]]:
        r = kp_lords(lon_fn(jd))
        return _key(r, level), r

    out: list[tuple[float, float, dict[str, Any]]] = []
    t = jd0
    k, r = st(t)
    seg_start, seg_r = t, r
    while t < jd1:
        nt = min(t + step, jd1)
        k2, _ = st(nt)
        if k2 == k:
            t = nt
            continue
        lo, hi = t, nt
        while hi - lo > 0.25 / 86400.0:
            mid = 0.5 * (lo + hi)
            if st(mid)[0] == k:
                lo = mid
            else:
                hi = mid
        out.append((seg_start, hi, seg_r))
        seg_start = hi
        k, seg_r = st(hi)
        t = hi
    out.append((seg_start, jd1, seg_r))
    return out


def planet_segments(ctx: Ctx, planet: str, start: datetime, end: datetime, level: str) -> list[dict[str, Any]]:
    if level not in LEVEL_SPAN:
        raise KPError(f"level must be one of {list(LEVEL_SPAN)}")
    if planet not in MAX_SPEED:
        raise KPError(f"planet must be one of {list(MAX_SPEED)}")
    step = LEVEL_SPAN[level] / MAX_SPEED[planet] / 3.0
    segs = _segments(lambda jd: ctx.lon(planet, jd), ctx.jd_from_local(start), ctx.jd_from_local(end), step, level)
    rows = []
    for a, b, r in segs:
        sp = ctx.lon_speed(planet, a)[1]
        rows.append({"planet": planet, "level": level, "start": fmt_dt(ctx.local_from_jd(a)), "end": fmt_dt(ctx.local_from_jd(b)),
                     "lord": r[LEVEL_LORD[level]], "sign": r["sign"], "nakshatra": r["nakshatra"], "star_lord": r["star_lord"],
                     "sub_lord": r["sub_lord"], "sub_sub_lord": r["sub_sub_lord"],
                     "retrograde": True if planet in NODES else sp < 0,
                     "hours": round((b - a) * 24.0, 3)})
    return rows


def find_boundary_changes(ctx: Ctx, planet: str, start: datetime, end: datetime, level: str = "sub",
                          limit: int = 500) -> dict[str, Any]:
    rows = planet_segments(ctx, planet, start, end, level)
    changes = []
    for prev, cur in zip(rows, rows[1:]):
        changes.append({"time": cur["start"], "from": prev["lord"], "to": cur["lord"], "sign": cur["sign"],
                        "nakshatra": cur["nakshatra"], "star_lord": cur["star_lord"], "sub_lord": cur["sub_lord"],
                        "retrograde": cur["retrograde"]})
    return {"planet": planet, "level": level, "window": [fmt_dt(start), fmt_dt(end)], "at_start": rows[0],
            "change_count": len(changes), "truncated": len(changes) > limit, "changes": changes[:limit],
            "note": ("A change is reported when the stretch's lord changes; a planet turning retrograde exactly at a boundary "
                     "inside one scan step can be missed.")}


def find_transit_triggers(ctx: Ctx, planets: list[str], targets: dict[str, list[str]], start: datetime, end: datetime,
                          levels: list[str], limit: int = 400) -> dict[str, Any]:
    """Stretches in which a transiting planet's sign / star / sub lord is one of the target lords (dasha lords, ruling planets)."""
    hits: list[dict[str, Any]] = []
    for pl in planets:
        for lv in levels:
            for seg in planet_segments(ctx, pl, start, end, lv):
                if seg["lord"] in targets:
                    hits.append({"planet": pl, "level": lv, "lord": seg["lord"], "roles": targets[seg["lord"]],
                                 "start": seg["start"], "end": seg["end"], "hours": seg["hours"],
                                 "retrograde": seg["retrograde"], "star_lord": seg["star_lord"], "sub_lord": seg["sub_lord"]})
    hits.sort(key=lambda h: (h["start"], h["planet"]))
    return {"window": [fmt_dt(start), fmt_dt(end)], "planets": planets, "levels": levels, "targets": targets,
            "count": len(hits), "truncated": len(hits) > limit, "triggers": hits[:limit],
            "rule": ("KP transit: slow planets pick months, the Sun picks the week, the Moon picks the day; an event needs the "
                     "running dasha / bhukti / antara to allow it first. This tool lists when transiting planets touch the target lords.")}


# --------------------------------------------------------------------------------------
# Lagna sub-lord changes around a time (birth-time sensitivity)
# --------------------------------------------------------------------------------------
def lagna_changes(ctx: Ctx, dt: datetime, lat: float, lon: float, minutes: float = 30.0, level: str = "sub") -> dict[str, Any]:
    if level not in LEVEL_SPAN:
        raise KPError(f"level must be one of {list(LEVEL_SPAN)}")
    if not 1 <= minutes <= 720:
        raise KPError("minutes must be between 1 and 720")
    jd0 = ctx.jd_from_local(dt - timedelta(minutes=minutes))
    jd1 = ctx.jd_from_local(dt + timedelta(minutes=minutes))
    step = LEVEL_SPAN[level] / 361.0 / 3.0   # the Ascendant moves at most about 361 deg/day
    segs = _segments(lambda jd: ctx.asc_only(jd, lat, lon), jd0, jd1, step, level)
    rows = [{"start": fmt_dt(ctx.local_from_jd(a)), "end": fmt_dt(ctx.local_from_jd(b)), "lord": r[LEVEL_LORD[level]],
             "sign": r["sign"], "nakshatra": r["nakshatra"], "star_lord": r["star_lord"], "sub_lord": r["sub_lord"],
             "sub_sub_lord": r["sub_sub_lord"], "minutes": round((b - a) * 1440.0, 2)} for a, b, r in segs]
    return {"around": fmt_dt(dt), "level": level, "window_minutes": minutes, "stretches": rows,
            "note": "Each stretch has the same Lagna lord at this level; birth-time errors move the chart between stretches."}


# --------------------------------------------------------------------------------------
# Birth-time rectification scan (event based)
# --------------------------------------------------------------------------------------
_VERDICT_POINTS = {"strongly_promised": 3.0, "promised_with_obstacles": 2.0, "weakly_promised": 1.0, "doubtful": 0.5,
                   "not_promised": 0.0}
MAX_CANDIDATES = 1500


def _event_dt(value: Any) -> datetime:
    text = str(value).strip()
    dt = parse_dt(text)
    return dt if ("T" in text or ":" in text) else dt + timedelta(hours=12)


SENSITIVITY_WINDOW_MIN = 60.0


def _sensitivity(ctx: Ctx, rng: dict[str, Any], lat: float, lon: float) -> dict[str, Any]:
    """Minutes the Lagna sub lord stays unchanged around the middle of a candidate range (limited to +- 60 minutes)."""
    a, b = datetime.fromisoformat(rng["start"]), datetime.fromisoformat(rng["end"])
    mid = a + (b - a) / 2
    lc = lagna_changes(ctx, mid, lat, lon, SENSITIVITY_WINDOW_MIN, "sub")
    for s in lc["stretches"]:
        s0, s1 = datetime.fromisoformat(s["start"]), datetime.fromisoformat(s["end"])
        if s0 <= mid <= s1:
            before, after = (mid - s0).total_seconds() / 60.0, (s1 - mid).total_seconds() / 60.0
            return {"around": fmt_dt(mid), "lagna_sub_lord": s["sub_lord"], "stretch_start": s["start"], "stretch_end": s["end"],
                    "minutes_back_to_change": round(before, 2), "minutes_forward_to_change": round(after, 2),
                    "stretch_minutes": round(before + after, 2),
                    "cut_by_scan_window": s0 <= mid - timedelta(minutes=SENSITIVITY_WINDOW_MIN) + timedelta(seconds=2)
                    or s1 >= mid + timedelta(minutes=SENSITIVITY_WINDOW_MIN) - timedelta(seconds=2)}
    return {"around": fmt_dt(mid), "lagna_sub_lord": None}


def rectify(ctx: Ctx, birth_dt: datetime, lat: float, lon: float, events: list[dict[str, Any]], minutes: float = 20.0,
            step_seconds: int = 60, levels: Optional[str] = None, top: int = 8, moon_link: str = "report") -> dict[str, Any]:
    """Score trial birth times: does the event's cuspal sub lord promise it and do the running Dasha, Bhukti and Antara
    lords signify its houses on the event date? Events may carry a `weight` (default 1). `moon_link` 'report' (default) shows the
    Lagna-Moon lord link per range; 'require' only ranks ranges where the link holds. Each best range gets a Lagna sub-lord
    sensitivity."""
    if moon_link not in MOON_LINK_MODES:
        raise KPError(f"moon_link must be one of {list(MOON_LINK_MODES)}")
    if not events:
        raise KPError("Give at least one event: {'date': 'YYYY-MM-DD', 'event': 'marriage'} (or custom_favourable houses).")
    if step_seconds < 5:
        raise KPError("step_seconds must be at least 5.")
    n = int(minutes * 60 // step_seconds)
    if 2 * n + 1 > MAX_CANDIDATES:
        raise KPError(f"Too many trial times ({2 * n + 1}); use a smaller `minutes` or a larger `step_seconds` (max {MAX_CANDIDATES}).")
    parsed = []
    for ev in events:
        if "date" not in ev:
            raise KPError("Every event needs a `date`.")
        if not (ev.get("event") or ev.get("custom_favourable")):
            raise KPError("Every event needs `event` (a known name) or `custom_favourable` houses.")
        parsed.append((ev, _event_dt(ev["date"]), event_weight(ev)))
    cands: list[dict[str, Any]] = []
    for i in range(-n, n + 1):
        dt = birth_dt + timedelta(seconds=i * step_seconds)
        jd = ctx.jd_from_local(dt)
        ch = chart_at(ctx, jd, lat, lon, dt)
        d = Dasha(dt, ch.planets["Moon"]["longitude"], ctx.opts.dasha_year_days)
        total = 0.0
        det = []
        for ev, edt, wt in parsed:
            j = judge_event(ch, ev.get("event"), ev.get("custom_favourable"), ev.get("custom_denial"),
                            ev.get("primary_cusp"), levels)
            fav = set(j["favourable_houses"])
            try:
                ladder = running_ladder(d, ch, edt, 3)
            except DashaError as exc:
                raise KPError(f"Event {ev['date']}: {exc}") from exc
            lords = [{"lord": r["lords"][-1], "level": r["level"], "favourable": sorted(set(r["lord_signifies"]) & fav)}
                     for r in ladder]
            pts = (_VERDICT_POINTS[j["verdict"]] + sum(1.0 for x in lords if x["favourable"])) * wt
            total += pts
            det.append({"date": ev["date"], "event": j["event"], "csl": j["csl"]["sub_lord"], "verdict": j["verdict"],
                        "running_lords": lords, "weight": wt, "score": pts})
        asc = kp_lords(ch.asc)
        link = lagna_moon_link(ch)
        cands.append({"time": fmt_dt(dt), "score": total, "asc_star_lord": asc["star_lord"], "asc_sub_lord": asc["sub_lord"],
                      "events": det, "link": link,
                      "_sig": (asc["star_lord"], asc["sub_lord"], link["linked"],
                               tuple((e["csl"], tuple(bool(x["favourable"]) for x in e["running_lords"])) for e in det))})
    ranges: list[dict[str, Any]] = []
    for c in cands:
        if ranges and ranges[-1]["_sig"] == c["_sig"]:
            ranges[-1]["end"] = c["time"]
            ranges[-1]["trial_times"] += 1
        else:
            ranges.append({"start": c["time"], "end": c["time"], "trial_times": 1, "score": c["score"],
                           "asc_star_lord": c["asc_star_lord"], "asc_sub_lord": c["asc_sub_lord"], "events": c["events"],
                           "lagna_moon_link": c["link"], "_sig": c["_sig"]})
    for r in ranges:
        r.pop("_sig")
    eligible, filtered, fallback = ranges, [], False
    if moon_link == "require":
        eligible = [r for r in ranges if r["lagna_moon_link"]["linked"]]
        filtered = [r for r in ranges if not r["lagna_moon_link"]["linked"]]
        if not eligible:
            eligible, filtered, fallback = ranges, [], True
    best = max(r["score"] for r in eligible)
    ordered = sorted(eligible, key=lambda r: (-r["score"], r["start"]))
    best_ranges = [r for r in ordered if r["score"] == best][:top]
    for r in best_ranges:
        r["lagna_sub_lord_sensitivity"] = _sensitivity(ctx, r, lat, lon)
    return {"given_birth_time": fmt_dt(birth_dt), "scanned": {"minutes_each_side": minutes, "step_seconds": step_seconds,
                                                             "trial_times": len(cands)},
            "max_score_possible": 6.0 * sum(w for _, _, w in parsed), "best_score": best,
            "moon_link": {"mode": moon_link, "no_range_linked_so_filter_not_applied": fallback, "ranges_filtered_out": len(filtered),
                          "rule": ("Pre-filter: the Lagna star / sub lord connect with the Moon star / sub lord (same planet, same sign, "
                                   "or sign aspect). 'report' only shows it; 'require' ranks only linked ranges.")},
            "weights_used": [{"date": e["date"], "event": e.get("event"), "weight": w} for e, _, w in parsed],
            "anchor_events": {"events": list(ANCHOR_EVENTS), "note": ANCHOR_NOTE},
            "best_ranges": best_ranges, "other_ranges": [r for r in ordered if r["score"] < best][:top],
            "filtered_out_ranges": filtered[:top],
            "method": ("Score per event: (cuspal sub lord verdict (3 strongly promised .. 0 not promised) + 1 for each running Dasha, "
                       "Bhukti and Antara lord that signifies a favourable house on the event date) x the event weight (default 1). "
                       "Ranges group consecutive trial times with the same outcome and the same Lagna-Moon link. Sensitivity = how long "
                       "the Lagna sub lord stays the same around the middle of each best range (+- 60 minutes scanned)."),
            "caution": ("Several ranges can tie, and event dates must be certain; rectification narrows the time, it does not prove it. "
                        "Add more events to separate candidates. The 'Rule of Origin' and the 2-minute method named in KP e-zines are "
                        "not implemented (rules not verified).")}
