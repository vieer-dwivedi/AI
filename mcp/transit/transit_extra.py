"""Transit engine, part 6: gap-filling tools.

Transit of dasha lords supplied by the Dasha MCP (no dasha is computed here), KP sub-lords of the transit, transit nakshatras (Tara from the
birth star, Janma/Anujanma/Trijanma, natal-star conjunctions), Sudarshana (houses from lagna, Moon and Sun), transit
planets in divisional charts, lunar phases and tithi dates, combustion (asta/udaya) periods, daily timings
(Rahu Kalam, choghadiya, hora ...) and one-call transit summary.

Dates in and out are LOCAL at the birth place. Facts and classical rules only; no predictions.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

import transit_calendar as C
import transit_classical as K
import transit_engine as E
import transit_events as V
from state import BirthState
from transit_engine import GOCHARA_GOOD, NAK_LEN, NAKSHATRAS, PLANETS, SIGNS, TransitError

# ------------------------------------------------------------------------------ tables
DASHA_ORDER = [8, 5, 0, 1, 2, 7, 4, 6, 3]  # Ketu, Venus, Sun, Moon, Mars, Rahu, Jupiter, Saturn, Mercury
# Vimshottari proportions are only used for KP sub-lords and nakshatra lords
DASHA_YEARS = {8: 7, 5: 20, 0: 6, 1: 10, 2: 7, 7: 18, 4: 16, 6: 19, 3: 17}
SIGN_LORDS = [2, 5, 3, 1, 0, 3, 5, 2, 4, 6, 6, 4]
CHALDEAN = [6, 4, 2, 0, 5, 3, 1]  # hora order: Saturn, Jupiter, Mars, Sun, Venus, Mercury, Moon
WEEKDAY_LORD = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 0}  # python weekday() -> day lord (Monday = Moon)
CHOGHADIYA = ["Udvega", "Chara", "Labha", "Amrita", "Kala", "Shubha", "Roga"]
CHOGHADIYA_GOOD = {"Labha", "Amrita", "Shubha", "Chara"}
VARGAS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 24, 27, 30, 40, 45, 60]
TITHI_NAMES_15 = ["Pratipada", "Dwitiya", "Tritiya", "Chaturthi", "Panchami", "Shashthi", "Saptami", "Ashtami", "Navami",
                  "Dashami", "Ekadashi", "Dwadashi", "Trayodashi", "Chaturdashi"]
PHASES = {0.0: "New Moon (Amavasya)", 90.0: "First quarter (Shukla Ashtami)", 180.0: "Full Moon (Purnima)",
          270.0: "Last quarter (Krishna Ashtami)"}


def _tithi_label(n: int) -> str:
    if n == 15:
        return "Shukla Purnima"
    if n == 30:
        return "Krishna Amavasya"
    return ("Shukla " if n < 15 else "Krishna ") + TITHI_NAMES_15[(n - 1) % 15]


def _nak_lord(lon: float) -> int:
    return DASHA_ORDER[int((lon % 360.0) // NAK_LEN) % 9]


# ------------------------------------------------------------------------------ dasha lords (lords come from the Dasha MCP)
def _lord_transit(b: Any, nat: Any, jd: float, p: int) -> dict[str, Any]:
    st = E._planet_state(b, jd, p)
    sign = st["sign_index"]
    nat_sign = int(nat.lon[p] // 30)
    row: dict[str, Any] = {
        "planet": PLANETS[p], "transit_sign": st["sign"], "degree_in_sign": st["degree_in_sign"], "nakshatra": st["nakshatra"],
        "retrograde": st["retrograde"], "house_from_natal_moon": E._house(nat.moon_sign, sign),
        "house_from_natal_lagna": E._house(nat.lagna_sign, sign), "natal_sign": SIGNS[nat_sign],
        "natal_house_from_lagna": E._house(nat.lagna_sign, nat_sign), "in_natal_sign": sign == nat_sign,
        "natal_planets_in_transit_sign": [PLANETS[q] for q in range(9) if q != p and int(nat.lon[q] // 30) == sign],
    }
    if p in GOCHARA_GOOD:
        row["gochara_favourable"] = row["house_from_natal_moon"] in GOCHARA_GOOD[p]
    return row


def get_lords_transit(state: BirthState, planets: list[Any], on_date: Optional[str] = None) -> dict[str, Any]:
    """Where chosen planets are transiting now, with their natal placement. Meant for dasha lords: take the running
    Mahadasha / Antardasha / Pratyantara lords from the Dasha MCP and pass them here. No dasha is computed in this server."""
    if not planets or len(planets) > 9:
        raise TransitError("`planets` must list 1 to 9 planets, e.g. the running dasha lords from the Dasha MCP.")
    ids = []
    for p in planets:
        i = E.planet_id(p)
        if i not in ids:
            ids.append(i)
    with E._calc(state) as (b, ctx):
        when = E.parse_when(on_date, state)
        jd = E.jd_utc_from_local(state, ctx, when)
        nat = E._natal(state, b, ctx)
        return {"state": state.summary(), "datetime_local": E.fmt_dt(when),
                "planets": [_lord_transit(b, nat, jd, p) for p in ids],
                "note": "Facts only: transit sign, houses from natal Moon/lagna, gochara verdict, natal sign/house, natal planets in the transit sign."}


# ------------------------------------------------------------------------------ KP
def _kp_lords(lon: float) -> dict[str, Any]:
    lon %= 360.0
    n = int(lon // NAK_LEN)
    star = DASHA_ORDER[n % 9]
    off = lon - n * NAK_LEN

    def split(start_lord: int, off_in: float, span: float) -> tuple[int, float, float]:
        i = DASHA_ORDER.index(start_lord)
        acc = 0.0
        for k in range(9):
            l = DASHA_ORDER[(i + k) % 9]
            ln = span * DASHA_YEARS[l] / 120.0
            if off_in < acc + ln or k == 8:
                return l, off_in - acc, ln
            acc += ln
        raise AssertionError

    sub, off2, span2 = split(star, off, NAK_LEN)
    subsub, _, _ = split(sub, off2, span2)
    return {"sign_lord": PLANETS[SIGN_LORDS[int(lon // 30)]], "star_lord": PLANETS[star], "sub_lord": PLANETS[sub],
            "sub_sub_lord": PLANETS[subsub]}


def get_kp_transit(state: BirthState, on_date: Optional[str] = None) -> dict[str, Any]:
    """KP (Krishnamurti) lords of every transiting planet at a date: sign lord, star (nakshatra) lord, sub-lord and
    sub-sub-lord, next to the same lords for the natal planets. Set the ayanamsa to KP in `options` for classic KP work."""
    with E._calc(state) as (b, ctx):
        when = E.parse_when(on_date, state)
        jd = E.jd_utc_from_local(state, ctx, when)
        nat = E._natal(state, b, ctx)
        rows = []
        for p in range(9):
            st = E._planet_state(b, jd, p)
            rows.append({"planet": PLANETS[p], "sign": st["sign"], "degree_in_sign": st["degree_in_sign"],
                         "nakshatra": st["nakshatra"], "retrograde": st["retrograde"], **_kp_lords(st["longitude"])})
        natal = [{"planet": PLANETS[p], "sign": SIGNS[int(nat.lon[p] // 30)], **_kp_lords(nat.lon[p])} for p in range(9)]
        natal.append({"planet": "Lagna", "sign": SIGNS[nat.lagna_sign], **_kp_lords(nat.lagna)})
        return {"state": state.summary(), "datetime_local": E.fmt_dt(when), "ayanamsa": state.options.ayanamsa,
                "transit": rows, "natal": natal,
                "note": "Sub-lord = nakshatra split in Vimshottari proportions (249 subs by sign-nakshatra). Natal lords use the state's "
                        "reference chart. House cusps / significators are not computed here."}


# ------------------------------------------------------------------------------ transit nakshatras
def get_transit_nakshatras(state: BirthState, on_date: Optional[str] = None) -> dict[str, Any]:
    """Every transiting planet's nakshatra with its lord, Tara from the natal Moon star (Janma ... Parama Mitra,
    favourable or not), the Janma / Anujanma / Trijanma group, and the natal planets sharing its nakshatra or sign."""
    with E._calc(state) as (b, ctx):
        when = E.parse_when(on_date, state)
        jd = E.jd_utc_from_local(state, ctx, when)
        nat = E._natal(state, b, ctx)
        birth_nak = int(nat.lon[1] // NAK_LEN)
        rows = []
        for p in range(9):
            st = E._planet_state(b, jd, p)
            nak0 = st["nakshatra_index"] - 1
            tara = (nak0 - birth_nak) % 9 + 1
            group = ["Janma (1st)", "Anujanma (10th)", "Trijanma (19th)"][((nak0 - birth_nak) % 27) // 9] if tara == 1 else None
            rows.append({
                "planet": PLANETS[p], "sign": st["sign"], "degree_in_sign": st["degree_in_sign"], "nakshatra": st["nakshatra"],
                "pada": st["pada"], "nakshatra_lord": PLANETS[_nak_lord(st["longitude"])],
                "tara_from_birth_star": {"number": tara, "name": C.TARA_NAMES[tara - 1], "favourable": tara in C.TARA_GOOD},
                "janma_group": group,
                "natal_planets_in_same_nakshatra": [PLANETS[q] for q in range(9) if q != p and int(nat.lon[q] // NAK_LEN) == nak0],
                "natal_planets_in_same_sign": [PLANETS[q] for q in range(9) if q != p and int(nat.lon[q] // 30) == st["sign_index"]],
                "over_natal_lagna_nakshatra": int(nat.lagna // NAK_LEN) == nak0,
            })
        return {"state": state.summary(), "datetime_local": E.fmt_dt(when), "janma_nakshatra": NAKSHATRAS[birth_nak],
                "planets": rows, "note": "Tara = (transit star - birth star) mod 9 + 1; favourable: Sampat, Kshema, Sadhana, Mitra, Parama Mitra."}


# ------------------------------------------------------------------------------ sudarshana
def get_sudarshana_transit(state: BirthState, on_date: Optional[str] = None) -> dict[str, Any]:
    """Sudarshana view: each transiting planet's house counted from the natal lagna, natal Moon and natal Sun at once,
    plus the classical gochara verdict (which is defined from the Moon only)."""
    with E._calc(state) as (b, ctx):
        when = E.parse_when(on_date, state)
        jd = E.jd_utc_from_local(state, ctx, when)
        nat = E._natal(state, b, ctx)
        sun_sign = int(nat.lon[0] // 30)
        rows = []
        for p in range(9):
            st = E._planet_state(b, jd, p)
            s = st["sign_index"]
            h = {"lagna": E._house(nat.lagna_sign, s), "moon": E._house(nat.moon_sign, s), "sun": E._house(sun_sign, s)}
            row = {"planet": PLANETS[p], "sign": st["sign"], "house_from_natal_lagna": h["lagna"],
                   "house_from_natal_moon": h["moon"], "house_from_natal_sun": h["sun"]}
            if p in GOCHARA_GOOD:
                row["gochara_favourable_from_moon"] = h["moon"] in GOCHARA_GOOD[p]
            rows.append(row)
        return {"state": state.summary(), "datetime_local": E.fmt_dt(when),
                "natal": {"lagna": SIGNS[nat.lagna_sign], "moon": SIGNS[nat.moon_sign], "sun": SIGNS[sun_sign]}, "planets": rows}


# ------------------------------------------------------------------------------ divisional charts
def get_transit_divisional(state: BirthState, on_date: Optional[str] = None, factors: Optional[list[int]] = None) -> dict[str, Any]:
    """Transit planets and the transit lagna placed in divisional charts (default D9 Navamsa and D10 Dasamsa; any of
    D2..D60), with each planet's house counted from the natal lagna of the same divisional chart."""
    fs = factors or [9, 10]
    bad = [f for f in fs if f not in VARGAS]
    if bad or len(fs) > 6:
        raise TransitError(f"`factors` must be at most 6 values from {VARGAS}.")
    with E._calc(state) as (b, ctx):
        when = E.parse_when(on_date, state)
        jd_local = E.jd_from_local(when)
        out = []
        for f in fs:
            tr = E._quiet(b.charts.divisional_chart, jd_local, ctx.place, divisional_chart_factor=f)
            na = E._quiet(b.charts.divisional_chart, ctx.jd, ctx.place, divisional_chart_factor=f)
            na_l = next(int(r[1][0]) for r in na if r[0] == "L")
            na_m = next(int(r[1][0]) for r in na if r[0] == 1)
            planets = []
            for r in tr:
                if r[0] == "L":
                    continue
                sign = int(r[1][0])
                planets.append({"planet": PLANETS[int(r[0])], "sign": SIGNS[sign], "degree_in_sign": round(float(r[1][1]), 3),
                                "house_from_natal_lagna_of_chart": E._house(na_l, sign), "house_from_natal_moon_of_chart": E._house(na_m, sign)})
            tl = next(int(r[1][0]) for r in tr if r[0] == "L")
            out.append({"divisional_chart": f"D{f}", "natal_lagna": SIGNS[na_l], "natal_moon": SIGNS[na_m],
                        "transit_lagna": SIGNS[tl], "planets": planets})
        return {"state": state.summary(), "datetime_local": E.fmt_dt(when), "charts": out}


# ------------------------------------------------------------------------------ lunar phases / tithis
def _elong(b: Any, jd: float) -> float:
    return (E._lon(b, jd, 1) - E._lon(b, jd, 0)) % 360.0


def get_lunar_phases(state: BirthState, start: Optional[str] = None, end: Optional[str] = None) -> dict[str, Any]:
    """New Moon, first quarter, Full Moon and last quarter moments in a window (default next 60 days, max 3 years) with the
    Moon's sign/nakshatra and its house from the natal Moon and lagna."""
    with E._calc(state) as (b, ctx):
        a, z = E._window(state, start, end, 60, 1096)
        nat = E._natal(state, b, ctx)
        t0, t1 = E.jd_utc_from_local(state, ctx, a), E.jd_utc_from_local(state, ctx, z)
        events = []
        for target, label in PHASES.items():
            for jd, _ in E.find_crossings(lambda j, t=target: E._wrap(_elong(b, j) - t), t0, t1, 14.5, 0.01, 1.0):
                m = E._planet_state(b, jd, 1)
                row = E._event_row(state, ctx, jd)
                row.update({"phase": label, "moon_sign": m["sign"], "degree_in_sign": m["degree_in_sign"], "nakshatra": m["nakshatra"],
                            "house_from_natal_moon": E._house(nat.moon_sign, m["sign_index"]),
                            "house_from_natal_lagna": E._house(nat.lagna_sign, m["sign_index"])})
                events.append(row)
        events.sort(key=lambda r: r["jd_utc"])
        return {"state": state.summary(), "window_local": [E.fmt_dt(a), E.fmt_dt(z)], "count": len(events), "events": events}


def get_tithi_dates(state: BirthState, tithis: Optional[list[int]] = None, start: Optional[str] = None,
                    end: Optional[str] = None) -> dict[str, Any]:
    """Start and end of chosen tithis (1..15 Shukla, 16..30 Krishna; 11 and 26 = Ekadashi, 13 and 28 = Pradosh/Trayodashi,
    4 and 19 = Chaturthi, 15 = Purnima, 30 = Amavasya) in a window (default next 45 days, max 1 year)."""
    ts = tithis or [11, 26]
    if any(not 1 <= t <= 30 for t in ts) or len(ts) > 30:
        raise TransitError("`tithis` must be numbers from 1 to 30.")
    with E._calc(state) as (b, ctx):
        a, z = E._window(state, start, end, 45, 366)
        t0, t1 = E.jd_utc_from_local(state, ctx, a), E.jd_utc_from_local(state, ctx, z)
        events = []
        for n in sorted(set(ts)):
            target = 12.0 * (n - 1)
            nxt = (target + 12.0) % 360.0
            # start a day early so a tithi already running at the window start is caught
            for jd, _ in E.find_crossings(lambda j, t=target: E._wrap(_elong(b, j) - t), t0 - 1.0, t1, 14.5, 0.01, 1.0):
                ends = E.find_crossings(lambda j, t=nxt: E._wrap(_elong(b, j) - t), jd + 0.2, jd + 1.7, 14.5, 0.01, 0.5, limit=1)
                end_jd = ends[0][0] if ends else None
                if end_jd is not None and end_jd < t0:
                    continue
                m = E._planet_state(b, jd, 1)
                row = E._event_row(state, ctx, jd)
                row.update({"tithi": n, "name": _tithi_label(n), "starts_local": row.pop("datetime_local"),
                            "ends_local": E.fmt_dt(E.local_dt_from_utc(state, ctx, end_jd)) if end_jd else None,
                            "moon_sign_at_start": m["sign"], "nakshatra_at_start": m["nakshatra"]})
                events.append(row)
        events.sort(key=lambda r: r["jd_utc"])
        return {"state": state.summary(), "window_local": [E.fmt_dt(a), E.fmt_dt(z)], "count": len(events), "events": events,
                "note": "Tithis begin/end at any hour; use `get_panchanga` for the tithi prevailing at sunrise."}


# ------------------------------------------------------------------------------ combustion periods
def get_combustion_periods(state: BirthState, planet: Any = "Venus", start: Optional[str] = None, end: Optional[str] = None) -> dict[str, Any]:
    """When Mars, Mercury, Jupiter, Venus or Saturn goes combust (asta) and rises again (udaya) in a window (default now to
    +2 years, max 20 years), with the classical orb (Mars 17, Mercury 14/12 retrograde, Jupiter 11, Venus 10/8 retrograde, Saturn 15)."""
    p = E.planet_id(planet)
    if p not in (2, 3, 4, 5, 6):
        raise TransitError("Combustion periods are computed for Mars, Mercury, Jupiter, Venus and Saturn.")
    with E._calc(state) as (b, ctx):
        a, z = E._window(state, start, end, 730, 7305)
        t0, t1 = E.jd_utc_from_local(state, ctx, a), E.jd_utc_from_local(state, ctx, z)
        direct, retro = K.COMBUST_ORB[p]

        def g(jd: float) -> float:
            sep = abs(E._wrap(E._lon(b, jd, p) - E._lon(b, jd, 0)))
            orb = retro if E._speed(b, jd, p) < 0 else direct
            return sep - orb

        events = []
        for jd, d in E.find_crossings(g, t0, t1, E.SPEED_BOUND[p] + 1.1, 0.05, 20.0):
            st = E._planet_state(b, jd, p)
            row = E._event_row(state, ctx, jd)
            row.update({"event": "goes combust (asta)" if d < 0 else "emerges from combustion (udaya)", "sign": st["sign"],
                        "degree_in_sign": st["degree_in_sign"], "retrograde": st["retrograde"]})
            events.append(row)
        return {"state": state.summary(), "planet": PLANETS[p], "window_local": [E.fmt_dt(a), E.fmt_dt(z)], "count": len(events),
                "events": events,
                "note": "Orb switches between the direct and retrograde value as the planet turns; a window may start or end inside a combustion."}


# ------------------------------------------------------------------------------ daily timings
def _hm_to_dt(day: datetime, text: str) -> datetime:
    hh, mm, ss = (int(float(x)) for x in (text.split(":") + ["0", "0"])[:3])
    return day.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(hours=hh, minutes=mm, seconds=ss)


def _span(vals: Any) -> Optional[list[str]]:
    try:
        return [str(vals[0]), str(vals[1])]
    except Exception:
        return None


def _hours_to_hms(h: float) -> str:
    h %= 24.0
    s = int(round(h * 3600))
    return f"{s // 3600 % 24:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def get_day_timings(state: BirthState, on_date: Optional[str] = None) -> dict[str, Any]:
    """Daily muhurta timings at the birth place for a date: sunrise/sunset/moonrise/moonset, Rahu Kalam, Yamaganda, Gulika,
    Abhijit, Durmuhurtam, Brahma muhurta, Godhuli, Vijaya, Nishita, Amrit Kalam, the 16 Gauri choghadiyas and the 24 horas
    (planetary hours) with the hora running at the requested moment."""
    with E._calc(state) as (b, ctx):
        when = E.parse_when(on_date, state)
        jd_local = E.jd_from_local(when)
        d = b.drik
        errors: dict[str, str] = {}

        def call(name: str, fn: Any, *args: Any) -> Any:
            try:
                return E._quiet(fn, *args)
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
                return None

        sr = call("sunrise", d.sunrise, jd_local, ctx.place)
        ss = call("sunset", d.sunset, jd_local, ctx.place)
        mr = call("moonrise", d.moonrise, jd_local, ctx.place)
        ms = call("moonset", d.moonset, jd_local, ctx.place)
        out: dict[str, Any] = {"state": state.summary(), "datetime_local": E.fmt_dt(when)}
        out["sunrise"] = sr[1] if sr else None
        out["sunset"] = ss[1] if ss else None
        out["moonrise"] = mr[1] if mr else None
        out["moonset"] = ms[1] if ms else None
        for key, opt in (("rahu_kalam", "raahu kaalam"), ("yamaganda", "yamagandam"), ("gulika_kalam", "gulikai")):
            v = call(key, d.trikalam, jd_local, ctx.place, opt)
            out[key] = _span(v) if v else None
        v = call("abhijit_muhurta", d.abhijit_muhurta, jd_local, ctx.place)
        out["abhijit_muhurta"] = _span(v) if v else None
        v = call("durmuhurtam", d.durmuhurtam, jd_local, ctx.place)
        out["durmuhurtam"] = [[str(v[i]), str(v[i + 1])] for i in range(0, len(v) - 1, 2)] if v else None
        v = call("amrit_kaalam", d.amrit_kaalam, jd_local, ctx.place)
        out["amrit_kaalam"] = [[str(x[0]), str(x[1])] for x in v] if v else None
        for key, fn in (("brahma_muhurta", d.brahma_muhurtha), ("godhuli_muhurta", d.godhuli_muhurtha), ("nishita_kaala", d.nishita_kaala)):
            v = call(key, fn, jd_local, ctx.place)
            out[key] = [_hours_to_hms(v[0]), _hours_to_hms(v[1])] if v else None
        v = call("vijaya_muhurta", d.vijaya_muhurtha, jd_local, ctx.place)
        out["vijaya_muhurta"] = [_hours_to_hms(v[0][0]), _hours_to_hms(v[0][1])] if v else None
        v = call("choghadiya", d.gauri_choghadiya, jd_local, ctx.place)
        if v:
            out["choghadiya"] = [{"name": CHOGHADIYA[int(r[0])], "from": r[1], "to": r[2], "favourable": CHOGHADIYA[int(r[0])] in CHOGHADIYA_GOOD}
                                 for r in v]
        # horas: sunrise -> sunset in 12 parts, sunset -> next sunrise in 12 parts; day lord by weekday of the Vedic day
        if sr and ss:
            sr_jd, ss_jd = float(sr[2]), float(ss[2])
            nsr = call("next_sunrise", d.sunrise, jd_local + 1.0, ctx.place)
            psr = call("prev_sunrise", d.sunrise, jd_local - 1.0, ctx.place)
            if jd_local < sr_jd and psr:  # before sunrise: still the previous Vedic day
                day = when.date() - timedelta(days=1)
                start, mid, endj = float(psr[2]), float(call("prev_sunset", d.sunset, jd_local - 1.0, ctx.place)[2]), sr_jd
            elif nsr:
                day = when.date()
                start, mid, endj = sr_jd, ss_jd, float(nsr[2])
            else:
                start = None
            if start is not None:
                lord0 = WEEKDAY_LORD[day.weekday()]
                idx = CHALDEAN.index(lord0)
                base = E.local_from_jd(start)
                horas = []
                for k in range(24):
                    if k < 12:
                        s_jd, e_jd = start + (mid - start) * k / 12.0, start + (mid - start) * (k + 1) / 12.0
                    else:
                        s_jd, e_jd = mid + (endj - mid) * (k - 12) / 12.0, mid + (endj - mid) * (k - 11) / 12.0
                    lord = CHALDEAN[(idx + k) % 7]
                    horas.append({"hora_lord": PLANETS[lord], "from": E.fmt_dt(E.local_from_jd(s_jd)), "to": E.fmt_dt(E.local_from_jd(e_jd)),
                                  "running": s_jd <= jd_local < e_jd})
                out["vara_lord"] = PLANETS[lord0]
                out["horas"] = horas
                out["current_hora"] = next((h["hora_lord"] for h in horas if h["running"]), None)
        out["note"] = ("Times are local at the birth place; windows after midnight belong to the same Vedic day. Hora = one-twelfth of the "
                       "day or night; the day lord's hora is first at sunrise.")
        if errors:
            out["errors"] = errors
        return out


# ------------------------------------------------------------------------------ jaimini
KARAKA_NAMES_7 = ["Atmakaraka (AK)", "Amatyakaraka (AmK)", "Bhratrikaraka (BK)", "Matrikaraka (MK)", "Putrakaraka (PiK)",
                  "Gnatikaraka (GK)", "Darakaraka (DK)"]
KARAKA_NAMES_8 = ["Atmakaraka (AK)", "Amatyakaraka (AmK)", "Bhratrikaraka (BK)", "Matrikaraka (MK)", "Pitrikaraka (PiK)",
                  "Putrakaraka (PuK)", "Gnatikaraka (GK)", "Darakaraka (DK)"]


def jaimini_aspect_signs(sign: int) -> list[int]:
    """Signs that `sign` aspects by Jaimini rasi drishti: movable -> fixed (not the adjacent one), fixed -> movable (not the
    adjacent one), dual -> the other duals."""
    kind = sign % 3  # 0 movable, 1 fixed, 2 dual
    if kind == 0:
        return [t for t in range(12) if t % 3 == 1 and t != (sign + 1) % 12]
    if kind == 1:
        return [t for t in range(12) if t % 3 == 0 and t != (sign - 1) % 12]
    return [t for t in range(12) if t % 3 == 2 and t != sign]


def get_jaimini_transit(state: BirthState, on_date: Optional[str] = None, scheme: int = 7) -> dict[str, Any]:
    """Jaimini view of the transit: natal Chara Karakas (7- or 8-karaka scheme) and Karakamsa, then for each transiting
    planet its Jaimini rasi drishti (movable-fixed, dual-dual, adjacent excluded) onto the natal lagna, karaka planets and
    Karakamsa, and its conjunctions with them."""
    if scheme not in (7, 8):
        raise TransitError("`scheme` must be 7 or 8 (8 includes Rahu as a karaka).")
    with E._calc(state) as (b, ctx):
        when = E.parse_when(on_date, state)
        jd = E.jd_utc_from_local(state, ctx, when)
        nat = E._natal(state, b, ctx)
        cands = list(range(7)) + ([7] if scheme == 8 else [])
        deg = {p: (nat.lon[p] % 30.0) if p != 7 else 30.0 - (nat.lon[p] % 30.0) for p in cands}
        order = sorted(cands, key=lambda p: -deg[p])
        names = KARAKA_NAMES_8 if scheme == 8 else KARAKA_NAMES_7
        karaka = {p: names[i] for i, p in enumerate(order)}
        ak = order[0]
        karakamsa = int(nat.lon[ak] // (30.0 / 9.0)) % 12
        targets: dict[int, list[str]] = {nat.lagna_sign: ["Lagna"], karakamsa: []}
        targets.setdefault(karakamsa, []).append("Karakamsa")
        for p in order:
            targets.setdefault(int(nat.lon[p] // 30), []).append(f"{PLANETS[p]} ({karaka[p].split(' ')[0]})")
        rows = []
        for p in range(9):
            st = E._planet_state(b, jd, p)
            s = st["sign_index"]
            asp = jaimini_aspect_signs(s)
            rows.append({
                "planet": PLANETS[p], "sign": st["sign"], "degree_in_sign": st["degree_in_sign"],
                "conjunct_natal": targets.get(s, []),
                "aspects_signs": [SIGNS[t] for t in asp],
                "aspects_natal": {SIGNS[t]: targets[t] for t in asp if t in targets},
            })
        return {"state": state.summary(), "datetime_local": E.fmt_dt(when), "scheme": f"{scheme}-karaka",
                "chara_karakas": [{"karaka": karaka[p], "planet": PLANETS[p], "sign": SIGNS[int(nat.lon[p] // 30)],
                                   "degree_in_sign": round(deg[p] if p != 7 else nat.lon[p] % 30.0, 3)} for p in order],
                "karakamsa": {"sign": SIGNS[karakamsa], "atmakaraka": PLANETS[ak],
                              "rule": "Navamsa sign of the Atmakaraka"},
                "natal_lagna": SIGNS[nat.lagna_sign], "transit": rows,
                "note": "Karakas rank Sun..Saturn by degree within the sign (Rahu counts 30 - degree in the 8-karaka scheme). "
                        "Arudha lagnas are not included."}


# ------------------------------------------------------------------------------ summary
def get_transit_summary(state: BirthState, on_date: Optional[str] = None, dasha_lords: Optional[list[Any]] = None) -> dict[str, Any]:
    """One-call overview at a date (default now): the planets with houses from Moon and lagna, gochara verdicts,
    Sade Sati / Dhaiya status, double transit, vedha-obstructed planets, panchanga, the transit of `dasha_lords` if given (take them from the Dasha MCP), and the
    next sign change of Jupiter, Saturn, Rahu and Ketu. Each section is independent; a failing section reports an error."""
    sections: dict[str, Any] = {}
    errors: dict[str, str] = {}

    def add(name: str, fn: Any, *args: Any) -> None:
        try:
            r = fn(state, *args)
            r.pop("state", None)
            sections[name] = r
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"

    add("positions", V.get_positions, on_date)
    add("gochara", V.get_gochara, on_date)
    add("sade_sati", V.get_sade_sati, on_date, 3, 6)
    add("double_transit", K.get_double_transit, on_date, "lagna")
    add("vedha", K.get_vedha, on_date)
    add("ashtakavarga", K.get_ashtakavarga_transit, on_date)
    if dasha_lords:
        add("dasha_lords_transit", get_lords_transit, dasha_lords, on_date)
    add("panchanga", K.get_panchanga, on_date)
    for p in ("Jupiter", "Saturn", "Rahu", "Ketu"):
        add(f"next_{p.lower()}_sign_change", V.get_next_sign_change, p, on_date)
    sades = sections.get("sade_sati")
    if sades:
        sades.pop("periods_in_window", None)  # keep the overview short; use get_sade_sati for the full list
    out = {"state": state.summary(), "sections": sections}
    if errors:
        out["errors"] = errors
    return out
