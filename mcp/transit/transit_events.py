"""Transit engine, part 2: positions, gochara, ingress / station / aspect searches, Sade Sati.

Builds on `transit_engine` (backend, scanner, time helpers). All public functions take a `BirthState`
and return JSON-serialisable dicts. Dates in and out are LOCAL at the birth place.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Callable, Optional

import transit_engine as E
from state import BirthState
from transit_engine import (
    ASPECT_HOUSES, GOCHARA_GOOD, MAX_EVENTS, NAK_LEN, PLANETS, SIGNS, TransitError,
)

ALL_PLANETS = list(range(9))
BOUNDARY_WIDTH = {"sign": 30.0, "nakshatra": NAK_LEN, "pada": NAK_LEN / 4.0}
ASPECT_ANGLES = {"conjunction": 0.0, "opposition": 180.0, "trine_forward": 120.0, "trine_back": 240.0,
                 "square_forward": 90.0, "square_back": 270.0}
MAX_DAYS_FAST = 3650        # Sun/Mercury/Venus/Mars/Moon-type scans
MAX_DAYS_SLOW = 365250      # one-planet sign scans are cheap enough for centuries only for slow planets


def _planets_arg(planets: Optional[list[Any]], default: Optional[list[int]] = None) -> list[int]:
    if planets is None:
        return list(default if default is not None else ALL_PLANETS)
    if not planets:
        raise TransitError("`planets` must not be empty.")
    seen: list[int] = []
    for item in planets:
        pid = E.planet_id(item)
        if pid not in seen:
            seen.append(pid)
    return seen


# --------------------------------------------------------------------------------------
# Boundary changes (sign / nakshatra / pada ingress) with ONE scan per planet
# --------------------------------------------------------------------------------------
def find_boundary_changes(b: Any, p: int, t0: float, t1: float, width: float, limit: int = MAX_EVENTS) -> list[dict[str, Any]]:
    """Every moment planet `p` crosses a multiple of `width` degrees between UTC jds t0 and t1."""
    speed = E.SPEED_BOUND[p]
    min_step = 0.01 if p == 1 else 0.02
    out: list[dict[str, Any]] = []
    t = t0
    lon = E._lon(b, t, p)
    idx = int(lon // width)
    guard = 0
    while t < t1 and len(out) < limit:
        guard += 1
        if guard > 300000:
            break
        off = lon % width
        dist = min(off, width - off)
        step = min(E.MAX_STEP[p], max(min_step, 0.7 * dist / speed))
        nt = min(t + step, t1)
        nlon = E._lon(b, nt, p)
        nidx = int(nlon // width)
        if nidx != idx and abs(E._wrap(nlon - lon)) < 90.0:
            lo, hi, lo_idx = t, nt, idx
            for _ in range(60):
                if hi - lo <= 2e-6:
                    break
                mid = 0.5 * (lo + hi)
                if int(E._lon(b, mid, p) // width) == lo_idx:
                    lo = mid
                else:
                    hi = mid
            root = 0.5 * (lo + hi)
            forward = E._wrap(nlon - lon) > 0
            out.append({"jd": root, "from_index": idx % int(round(360 / width)), "to_index": nidx % int(round(360 / width)),
                        "direction": "forward" if forward else "retrograde"})
        t, lon, idx = nt, nlon, nidx
    return out


# --------------------------------------------------------------------------------------
# Current positions + gochara
# --------------------------------------------------------------------------------------
def _overlay(nat: Any, row: dict[str, Any], p: int) -> dict[str, Any]:
    sign = row["sign_index"]
    h_moon = E._house(nat.moon_sign, sign)
    h_lagna = E._house(nat.lagna_sign, sign)
    row["house_from_natal_moon"] = h_moon
    row["house_from_natal_lagna"] = h_lagna
    row["gochara_from_moon"] = "favourable" if h_moon in GOCHARA_GOOD[p] else "unfavourable"
    return row


def get_positions(state: BirthState, on_date: Optional[str] = None, planets: Optional[list[Any]] = None) -> dict[str, Any]:
    """Transiting planet positions at a date, with house counted from natal Moon and natal lagna."""
    chosen = _planets_arg(planets)
    with E._calc(state) as (b, ctx):
        when = E.parse_when(on_date, state)
        jd = E.jd_utc_from_local(state, ctx, when)
        nat = E._natal(state, b, ctx)
        rows = [_overlay(nat, E._planet_state(b, jd, p), p) for p in chosen]
        return {
            "state": state.summary(),
            "ayanamsa": state.options.ayanamsa,
            "datetime_local": E.fmt_dt(when),
            "natal_reference": {"lagna": E.SIGNS[nat.lagna_sign], "moon_sign": E.SIGNS[nat.moon_sign],
                                "divisional_chart_factor": ctx.dcf},
            "planets": rows,
        }


def get_gochara(state: BirthState, on_date: Optional[str] = None) -> dict[str, Any]:
    """Classical Parashari gochara table: each transiting planet's house from natal Moon and whether that
    house is in its traditionally favourable set. Facts only; no predictions."""
    data = get_positions(state, on_date)
    table = []
    for row in data["planets"]:
        pid = PLANETS.index(row["planet"])
        table.append({
            "planet": row["planet"], "sign": row["sign"], "degree_in_sign": row["degree_in_sign"],
            "house_from_natal_moon": row["house_from_natal_moon"],
            "classical_favourable_houses": sorted(GOCHARA_GOOD[pid]),
            "gochara": row["gochara_from_moon"], "retrograde": row["retrograde"],
        })
    return {"state": data["state"], "datetime_local": data["datetime_local"],
            "natal_moon_sign": data["natal_reference"]["moon_sign"], "gochara": table,
            "note": "Classical rule of thumb only (Parashari gochara from the natal Moon); vedha and ashtakavarga are not applied."}


def get_transit_vs_natal(state: BirthState, on_date: Optional[str] = None, orb: float = 5.0) -> dict[str, Any]:
    """Transiting planets against natal planets: same-sign conjunctions, rasi aspects (7th for all; Mars 4/8,
    Jupiter 5/9, Saturn 3/10, nodes 5/9), and degree conjunctions/oppositions within `orb` degrees."""
    if not 0 < orb <= 15:
        raise TransitError("`orb` must be between 0 and 15 degrees.")
    with E._calc(state) as (b, ctx):
        when = E.parse_when(on_date, state)
        jd = E.jd_utc_from_local(state, ctx, when)
        nat = E._natal(state, b, ctx)
        natal_sign = {p: int(l // 30) for p, l in nat.lon.items()}
        rows = []
        for tp in ALL_PLANETS:
            tlon = E._lon(b, jd, tp)
            tsign = int(tlon // 30)
            for npl in ALL_PLANETS:
                rel = []
                if tsign == natal_sign[npl]:
                    rel.append("same sign (conjunction by sign)")
                dist = E._house(tsign, natal_sign[npl])
                if dist in ASPECT_HOUSES[tp] and dist != 1:
                    rel.append(f"aspects natal {PLANETS[npl]}'s sign (house {dist} from transit planet)")
                diff = E._wrap(tlon - nat.lon[npl])
                if abs(diff) <= orb:
                    rel.append(f"within {abs(diff):.2f} deg of exact conjunction")
                elif abs(abs(diff) - 180.0) <= orb:
                    rel.append(f"within {abs(abs(diff) - 180.0):.2f} deg of exact opposition")
                if rel:
                    rows.append({"transit_planet": PLANETS[tp], "natal_planet": PLANETS[npl], "relations": rel})
        return {"state": state.summary(), "datetime_local": E.fmt_dt(when), "orb_degrees": orb, "relations": rows}


# --------------------------------------------------------------------------------------
# Ingress (sign / nakshatra / pada)
# --------------------------------------------------------------------------------------
def get_ingresses(state: BirthState, planet: Any = "Saturn", kind: str = "sign", start: Optional[str] = None,
                  end: Optional[str] = None, limit: int = 20) -> dict[str, Any]:
    """Moments a planet enters a new sign / nakshatra / pada in a window (default: now .. +2 years;
    Moon: now .. +60 days). Includes retrograde re-entries."""
    kind = kind.strip().lower()
    if kind not in BOUNDARY_WIDTH:
        raise TransitError(f"`kind` must be one of {list(BOUNDARY_WIDTH)}.")
    if not 1 <= limit <= MAX_EVENTS:
        raise TransitError(f"`limit` must be 1..{MAX_EVENTS}.")
    p = E.planet_id(planet)
    default_days = 60 if p == 1 else 730
    max_days = 800 if p == 1 else (MAX_DAYS_SLOW if p in (4, 6, 7, 8) else MAX_DAYS_FAST)
    with E._calc(state) as (b, ctx):
        a, z = E._window(state, start, end, default_days, max_days)
        width = BOUNDARY_WIDTH[kind]
        raw = find_boundary_changes(b, p, E.jd_utc_from_local(state, ctx, a), E.jd_utc_from_local(state, ctx, z), width, limit)
        names = SIGNS if kind == "sign" else (E.NAKSHATRAS if kind == "nakshatra" else None)
        events = []
        for ev in raw:
            row = E._event_row(state, ctx, ev["jd"])
            row["direction"] = ev["direction"]
            if names is not None:
                row["from"], row["to"] = names[ev["from_index"]], names[ev["to_index"]]
            else:
                nak = int(E._lon(b, ev["jd"] + 1e-5, p) // NAK_LEN) % 27
                row["nakshatra"] = E.NAKSHATRAS[nak]
                row["pada"] = ev["to_index"] % 4 + 1
            events.append(row)
        return {"state": state.summary(), "planet": PLANETS[p], "kind": kind,
                "window_local": [E.fmt_dt(a), E.fmt_dt(z)], "count": len(events),
                "truncated": len(raw) >= limit, "events": events}


def get_next_sign_change(state: BirthState, planet: Any = "Jupiter", on_date: Optional[str] = None) -> dict[str, Any]:
    """The current sign of a planet and when it next changes sign (searches up to ~30 years for slow planets)."""
    p = E.planet_id(planet)
    horizon = 60 if p == 1 else (1100 if p not in (4, 6, 7, 8) else 10950)
    with E._calc(state) as (b, ctx):
        when = E.parse_when(on_date, state)
        jd = E.jd_utc_from_local(state, ctx, when)
        cur = E._planet_state(b, jd, p)
        raw = find_boundary_changes(b, p, jd, jd + horizon, 30.0, 1)
        nxt = None
        if raw:
            ev = raw[0]
            nxt = {**E._event_row(state, ctx, ev["jd"]), "to": SIGNS[ev["to_index"]], "direction": ev["direction"],
                   "days_from_now": round(ev["jd"] - jd, 2)}
        return {"state": state.summary(), "datetime_local": E.fmt_dt(when), "current": cur, "next_sign_change": nxt}


# --------------------------------------------------------------------------------------
# Stations / retrograde periods
# --------------------------------------------------------------------------------------
def get_stations(state: BirthState, planet: Any = "Jupiter", start: Optional[str] = None, end: Optional[str] = None) -> dict[str, Any]:
    """Stationary points (retrograde / direct) of Mars..Saturn in a window (default: now .. +3 years) and the
    retrograde periods they bound."""
    p = E.planet_id(planet)
    if p not in E.STATION_STEP:
        raise TransitError("Stations exist only for Mars, Mercury, Jupiter, Venus and Saturn (not Sun, Moon or the nodes).")
    with E._calc(state) as (b, ctx):
        a, z = E._window(state, start, end, 1095, 36525)
        t0, t1 = E.jd_utc_from_local(state, ctx, a), E.jd_utc_from_local(state, ctx, z)
        # speed changes sign twice per retrograde loop; a plain sign-change scan is enough
        stations: list[dict[str, Any]] = []
        t, gt = t0, E._speed(b, t0, p)
        step = E.STATION_STEP[p]
        while t < t1 and len(stations) < MAX_EVENTS:
            nt = min(t + step, t1)
            gn = E._speed(b, nt, p)
            if (gt < 0) != (gn < 0):
                root = E._bisect(lambda x: E._speed(b, x, p), t, nt, gt)
                kind = "retrograde (station retrograde)" if gn < 0 else "direct (station direct)"
                row = E._event_row(state, ctx, root)
                row.update({"station": kind, **{k: v for k, v in E.describe_longitude(E._lon(b, root, p)).items()
                                                 if k in ("sign", "degree_in_sign", "nakshatra")}})
                stations.append(row)
            t, gt = nt, gn
        now_state = E._planet_state(b, t0, p)
        periods = []
        i = 0
        while i < len(stations):
            if stations[i]["station"].startswith("retrograde"):
                end_row = stations[i + 1] if i + 1 < len(stations) else None
                periods.append({"retrograde_from": stations[i]["datetime_local"],
                                "retrograde_until": end_row["datetime_local"] if end_row else "after window end",
                                "from_sign": stations[i]["sign"], "to_sign": end_row["sign"] if end_row else None})
            i += 1
        return {"state": state.summary(), "planet": PLANETS[p], "window_local": [E.fmt_dt(a), E.fmt_dt(z)],
                "retrograde_at_window_start": now_state["retrograde"], "stations": stations,
                "retrograde_periods": periods}


# --------------------------------------------------------------------------------------
# Exact aspects: transit planet to a natal point, or planet to planet
# --------------------------------------------------------------------------------------
def _angles(aspects: Optional[list[str]]) -> dict[str, float]:
    if aspects is None:
        return {"conjunction": 0.0, "opposition": 180.0}
    out = {}
    for name in aspects:
        key = str(name).strip().lower().replace(" ", "_")
        if key in ("trine", "square"):
            fwd, back = (f"{key}_forward", f"{key}_back")
            out[fwd], out[back] = ASPECT_ANGLES[fwd], ASPECT_ANGLES[back]
        elif key in ASPECT_ANGLES:
            out[key] = ASPECT_ANGLES[key]
        else:
            raise TransitError(f"Unknown aspect '{name}'. Use conjunction, opposition, trine, square.")
    return out


def search_transit_to_natal(state: BirthState, transit_planet: Any = "Saturn", natal_point: Any = "Moon",
                            aspects: Optional[list[str]] = None, start: Optional[str] = None,
                            end: Optional[str] = None) -> dict[str, Any]:
    """Exact dates a transiting planet makes a conjunction/opposition (optionally trine/square) to a natal
    planet or the natal 'Lagna' (default window: now .. +5 years). Retrograde re-passes are reported separately."""
    tp = E.planet_id(transit_planet)
    angles = _angles(aspects)
    with E._calc(state) as (b, ctx):
        nat = E._natal(state, b, ctx)
        if str(natal_point).strip().lower() in ("lagna", "ascendant", "asc", "l"):
            target, label = nat.lagna, "Lagna"
        else:
            npl = E.planet_id(natal_point)
            target, label = nat.lon[npl], PLANETS[npl]
        slow = tp in (4, 6, 7, 8)
        a, z = E._window(state, start, end, 1826 if slow else 365, MAX_DAYS_SLOW if slow else MAX_DAYS_FAST)
        t0, t1 = E.jd_utc_from_local(state, ctx, a), E.jd_utc_from_local(state, ctx, z)
        events = []
        for name, ang in angles.items():
            tgt = (target + ang) % 360.0
            for jd, direction in E.find_crossings(E._longitude_target_fn(b, tp, tgt), t0, t1, E.SPEED_BOUND[tp],
                                                  0.01 if tp == 1 else 0.02, E.MAX_STEP[tp]):
                row = E._event_row(state, ctx, jd)
                sp = E._speed(b, jd, tp)
                row.update({"aspect": name.split("_")[0], "transit_planet": PLANETS[tp], "natal_point": label,
                            "transit_longitude": round(E._lon(b, jd, tp), 4), "retrograde_at_event": bool(sp < 0)})
                events.append(row)
        events.sort(key=lambda r: r["jd_utc"])
        return {"state": state.summary(), "window_local": [E.fmt_dt(a), E.fmt_dt(z)], "natal_point_longitude": round(target, 4),
                "count": len(events), "events": events}


def search_planet_pair(state: BirthState, planet_a: Any = "Jupiter", planet_b: Any = "Saturn", aspects: Optional[list[str]] = None,
                       start: Optional[str] = None, end: Optional[str] = None) -> dict[str, Any]:
    """Exact dates two TRANSITING planets form a conjunction/opposition (optionally trine/square)
    (default window: now .. +3 years; up to 100 years for pairs of slow planets)."""
    pa, pb = E.planet_id(planet_a), E.planet_id(planet_b)
    if pa == pb:
        raise TransitError("Choose two different planets.")
    angles = _angles(aspects)
    with E._calc(state) as (b, ctx):
        fast = max(E.SPEED_BOUND[pa], E.SPEED_BOUND[pb])
        both_slow = pa in (4, 6, 7, 8) and pb in (4, 6, 7, 8)
        a, z = E._window(state, start, end, 1095, 36525 if both_slow else MAX_DAYS_FAST)
        t0, t1 = E.jd_utc_from_local(state, ctx, a), E.jd_utc_from_local(state, ctx, z)
        bound = E.SPEED_BOUND[pa] + E.SPEED_BOUND[pb]
        events = []
        for name, ang in angles.items():
            g = lambda jd, ang=ang: E._wrap(E._lon(b, jd, pa) - E._lon(b, jd, pb) - ang)
            for jd, direction in E.find_crossings(g, t0, t1, bound, 0.02, max(E.MAX_STEP[pa], E.MAX_STEP[pb])):
                row = E._event_row(state, ctx, jd)
                lon_a = E._lon(b, jd, pa)
                row.update({"aspect": name.split("_")[0], "planet_a": PLANETS[pa], "planet_b": PLANETS[pb],
                            **{k: v for k, v in E.describe_longitude(lon_a).items() if k in ("sign", "degree_in_sign")}})
                events.append(row)
        events.sort(key=lambda r: r["jd_utc"])
        return {"state": state.summary(), "window_local": [E.fmt_dt(a), E.fmt_dt(z)], "count": len(events), "events": events}



def _merge_dips(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join consecutive rows in the same sign separated by a short retrograde dip into the previous sign."""
    merged: list[dict[str, Any]] = []
    for row in rows:
        prev = merged[-1] if merged else None
        if prev is not None and prev["saturn_sign"] == row["saturn_sign"] and prev["_to"] < row["_from"] \
                and (row["_from"] - prev["_to"]) < 400:
            prev.setdefault("retrograde_dips_out_of_sign", []).append({"from_local": prev["to_local"], "to_local": row["from_local"]})
            prev["to_local"], prev["_to"] = row["to_local"], row["_to"]
        else:
            merged.append(row)
    for row in merged:
        row.pop("_from", None)
        row.pop("_to", None)
    return merged

# --------------------------------------------------------------------------------------
# Sade Sati / Dhaiya
# --------------------------------------------------------------------------------------
def get_sade_sati(state: BirthState, on_date: Optional[str] = None, years_back: int = 15, years_forward: int = 35) -> dict[str, Any]:
    """Saturn's position relative to the natal Moon sign: whether Sade Sati (12th/1st/2nd) or a Dhaiya (4th/8th)
    is running at `on_date`, plus every Saturn sign-stay in the surrounding window with its phase."""
    if not (0 <= years_back <= 60 and 0 <= years_forward <= 60):
        raise TransitError("`years_back` and `years_forward` must be between 0 and 60.")
    with E._calc(state) as (b, ctx):
        when = E.parse_when(on_date, state)
        jd = E.jd_utc_from_local(state, ctx, when)
        nat = E._natal(state, b, ctx)
        moon = nat.moon_sign

        def phase(sign: int) -> Optional[str]:
            h = E._house(moon, sign)
            if h in E.SADE_SATI_HOUSES:
                return "Sade Sati: " + E.SADE_SATI_HOUSES[h]
            return E.DHAIYA_HOUSES.get(h)

        t0, t1 = jd - years_back * 365.2425, jd + years_forward * 365.2425
        changes = find_boundary_changes(b, 6, t0, t1, 30.0, MAX_EVENTS)
        cur_sign = int(E._lon(b, t0, 6) // 30)
        segments = []
        seg_start, seg_sign = t0, cur_sign
        for ev in changes:
            segments.append((seg_start, ev["jd"], seg_sign))
            seg_start, seg_sign = ev["jd"], ev["to_index"]
        segments.append((seg_start, t1, seg_sign))
        rows = []
        for s, e_, sign in segments:
            ph = phase(sign)
            if ph is None:
                continue
            rows.append({
                "saturn_sign": SIGNS[sign], "house_from_moon": E._house(moon, sign), "phase": ph,
                "from_local": E.fmt_dt(E.local_dt_from_utc(state, ctx, s)) + (" (or earlier: window start)" if s == t0 else ""),
                "to_local": E.fmt_dt(E.local_dt_from_utc(state, ctx, e_)) + (" (or later: window end)" if e_ == t1 else ""),
                "_from": s, "_to": e_,
            })
        rows = _merge_dips(rows)
        now_sign = int(E._lon(b, jd, 6) // 30)
        h = E._house(moon, now_sign)
        return {
            "state": state.summary(), "datetime_local": E.fmt_dt(when),
            "natal_moon_sign": SIGNS[moon], "saturn_now": {"sign": SIGNS[now_sign], "house_from_moon": h},
            "sade_sati_running": h in E.SADE_SATI_HOUSES, "dhaiya_running": h in E.DHAIYA_HOUSES,
            "current_phase": phase(now_sign), "periods_in_window": rows,
            "note": "One row per stay in a sign; short retrograde dips into the previous sign are listed inside the row.",
        }
