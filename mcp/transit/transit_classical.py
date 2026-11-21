"""Transit engine, part 5: Ashtakavarga transit scoring (with Kakshya), Vedha, Murthi Nirnaya, double transit,
planet conditions (retrograde / combust / dignity / graha yuddha / gandanta), panchanga at a moment, solstices and
equinoxes, planet returns, and a combined major-transits calendar.

Dates in and out are LOCAL at the birth place. Facts and classical rules only; no predictions.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Optional

import transit_calendar as C
import transit_engine as E
import transit_events as V
from state import BirthState
from transit_engine import GOCHARA_GOOD, NAK_LEN, NAKSHATRAS, PLANETS, SIGNS, TransitError

# ------------------------------------------------------------------------------ tables
KAKSHYA_LORDS = [6, 4, 2, 0, 5, 3, 1, 7]  # Saturn, Jupiter, Mars, Sun, Venus, Mercury, Moon, Lagna (3d45' each)
KAKSHYA_NAMES = {0: "Sun", 1: "Moon", 2: "Mars", 3: "Mercury", 4: "Jupiter", 5: "Venus", 6: "Saturn", 7: "Lagna"}

# Vedha (obstruction): {planet: {good house from natal Moon: house whose occupant obstructs it}}
VEDHA = {
    0: {3: 9, 6: 12, 10: 4, 11: 5},
    1: {1: 5, 3: 9, 6: 12, 7: 2, 10: 4, 11: 8},
    2: {3: 12, 6: 9, 11: 5},
    3: {2: 5, 4: 3, 6: 9, 8: 1, 10: 8, 11: 12},
    4: {2: 12, 5: 4, 7: 3, 9: 10, 11: 8},
    5: {1: 8, 2: 7, 3: 1, 4: 10, 5: 9, 8: 5, 9: 11, 11: 6, 12: 3},
    6: {3: 12, 6: 9, 11: 5},
}
VEDHA_EXEMPT = {frozenset((0, 6)), frozenset((1, 3))}  # father/son pairs do not obstruct each other

MURTHI = {1: "Swarna (gold)", 6: "Swarna (gold)", 11: "Swarna (gold)",
          2: "Rajata (silver)", 5: "Rajata (silver)", 9: "Rajata (silver)",
          3: "Tamra (copper)", 7: "Tamra (copper)", 10: "Tamra (copper)",
          4: "Loha (iron)", 8: "Loha (iron)", 12: "Loha (iron)"}
MURTHI_QUALITY = {"Swarna (gold)": "most favourable", "Rajata (silver)": "favourable",
                  "Tamra (copper)": "average", "Loha (iron)": "unfavourable"}

SIGN_LORDS = [2, 5, 3, 1, 0, 3, 5, 2, 4, 6, 6, 4]
EXALT = {0: (0, 10.0), 1: (1, 3.0), 2: (9, 28.0), 3: (5, 15.0), 4: (3, 5.0), 5: (11, 27.0), 6: (6, 20.0)}
DEBIL = {p: ((s + 6) % 12, d) for p, (s, d) in EXALT.items()}
OWN = {0: {4}, 1: {3}, 2: {0, 7}, 3: {2, 5}, 4: {8, 11}, 5: {1, 6}, 6: {9, 10}}
# Combustion orbs from the Sun (degrees): direct / retrograde
COMBUST_ORB = {1: (12.0, 12.0), 2: (17.0, 17.0), 3: (14.0, 12.0), 4: (11.0, 11.0), 5: (10.0, 8.0), 6: (15.0, 15.0)}

TITHI_NAMES = ["Pratipada", "Dwitiya", "Tritiya", "Chaturthi", "Panchami", "Shashthi", "Saptami", "Ashtami",
               "Navami", "Dashami", "Ekadashi", "Dwadashi", "Trayodashi", "Chaturdashi"]
YOGA_NAMES = ["Vishkambha", "Priti", "Ayushman", "Saubhagya", "Shobhana", "Atiganda", "Sukarma", "Dhriti", "Shula",
              "Ganda", "Vriddhi", "Dhruva", "Vyaghata", "Harshana", "Vajra", "Siddhi", "Vyatipata", "Variyana",
              "Parigha", "Shiva", "Siddha", "Sadhya", "Shubha", "Shukla", "Brahma", "Indra", "Vaidhriti"]
KARANA_REPEATING = ["Bava", "Balava", "Kaulava", "Taitila", "Garaja", "Vanija", "Vishti (Bhadra)"]
VARA_NAMES = ["Monday (Soma)", "Tuesday (Mangala)", "Wednesday (Budha)", "Thursday (Guru)", "Friday (Shukra)",
              "Saturday (Shani)", "Sunday (Ravi)"]  # python weekday(): Monday = 0
RETURN_DAYS = {0: 365.2564, 1: 27.3217, 2: 686.98, 3: 87.97, 4: 4332.6, 5: 224.7, 6: 10759.0, 7: 6798.4, 8: 6798.4}


def _tithi_name(n: int) -> str:
    paksha = "Shukla" if n <= 15 else "Krishna"
    if n == 15:
        return "Shukla Purnima"
    if n == 30:
        return "Krishna Amavasya"
    return f"{paksha} {TITHI_NAMES[(n - 1) % 15]}"


def _karana_name(k: int) -> str:
    """k = 1..60 (half-tithi count from the new moon)."""
    if k == 1:
        return "Kimstughna"
    if k >= 58:
        return ["Shakuni", "Chatushpada", "Naga"][k - 58]
    return KARANA_REPEATING[(k - 2) % 7]


# ------------------------------------------------------------------------------ ashtakavarga transit
def _natal_av(state: BirthState, b: Any, ctx: Any) -> tuple[list, list, list]:
    chart = E._quiet(b.charts.divisional_chart, ctx.jd, ctx.place, divisional_chart_factor=ctx.dcf)
    h2p = b.utils.get_house_planet_list_from_planet_positions(chart)
    return E._quiet(b.ashtakavarga.get_ashtaka_varga, h2p)


def get_ashtakavarga_transit(state: BirthState, on_date: Optional[str] = None) -> dict[str, Any]:
    """Ashtakavarga view of the transit: for each planet Sun..Saturn, the Bhinnashtakavarga bindus (0-8) of the sign it
    occupies in the natal chart's ashtakavarga, the Sarvashtakavarga bindus of that sign, and the Kakshya (3d45' strip)
    with whether that Kakshya's lord contributed a bindu. Rule of thumb: 4+ bindus favourable; SAV 28+ strong."""
    with E._calc(state) as (b, ctx):
        when = E.parse_when(on_date, state)
        jd = E.jd_utc_from_local(state, ctx, when)
        bav, sav, prastara = _natal_av(state, b, ctx)
        nat = E._natal(state, b, ctx)
        rows = []
        for p in range(7):
            lon = E._lon(b, jd, p)
            sign = int(lon // 30)
            deg = lon - sign * 30
            kidx = min(int(deg // 3.75), 7)
            klord = KAKSHYA_LORDS[kidx]
            bindus = int(bav[p][sign])
            gave = bool(prastara[p][klord][sign])
            rows.append({
                "planet": PLANETS[p], "sign": SIGNS[sign], "degree_in_sign": round(deg, 3),
                "house_from_natal_moon": E._house(nat.moon_sign, sign),
                "bhinna_bindus_in_sign": bindus, "bhinna_verdict": "favourable" if bindus >= 4 else "weak",
                "sarva_bindus_in_sign": int(sav[sign]), "sarva_verdict": "strong" if sav[sign] >= 28 else "below 28",
                "kakshya": {"number": kidx + 1, "lord": KAKSHYA_NAMES[klord], "lord_gave_bindu": gave,
                            "range_deg": [round(kidx * 3.75, 2), round((kidx + 1) * 3.75, 2)]},
            })
        return {"state": state.summary(), "datetime_local": E.fmt_dt(when),
                "natal_reference": {"divisional_chart_factor": ctx.dcf, "sav_total": int(sum(sav))},
                "transit": rows,
                "note": "Bindus come from PyJHora's ashtakavarga of the natal chart (default options). Rahu/Ketu have no ashtakavarga."}


def get_ashtakavarga_tables(state: BirthState) -> dict[str, Any]:
    """The natal Bhinnashtakavarga (Sun..Saturn + Lagna, bindus per sign) and Sarvashtakavarga used for transit scoring."""
    with E._calc(state) as (b, ctx):
        bav, sav, _ = _natal_av(state, b, ctx)
        names = [PLANETS[p] for p in range(7)] + ["Lagna"]
        return {"state": state.summary(), "signs": SIGNS,
                "bhinnashtakavarga": {names[i]: [int(x) for x in bav[i]] for i in range(8)},
                "sarvashtakavarga": [int(x) for x in sav], "sav_total": int(sum(sav))}


# ------------------------------------------------------------------------------ vedha
def get_vedha(state: BirthState, on_date: Optional[str] = None) -> dict[str, Any]:
    """Classical Vedha (obstruction): a planet in a favourable house from the natal Moon is obstructed when another
    planet sits in its paired vedha house (Sun-Saturn and Moon-Mercury never obstruct each other)."""
    with E._calc(state) as (b, ctx):
        when = E.parse_when(on_date, state)
        jd = E.jd_utc_from_local(state, ctx, when)
        nat = E._natal(state, b, ctx)
        house = {p: E._house(nat.moon_sign, int(E._lon(b, jd, p) // 30)) for p in range(9)}
        rows = []
        for p in range(7):
            h = house[p]
            if h not in GOCHARA_GOOD[p]:
                rows.append({"planet": PLANETS[p], "house_from_natal_moon": h, "in_favourable_house": False})
                continue
            vh = VEDHA[p][h]
            blockers = [PLANETS[q] for q in range(9) if q != p and house[q] == vh
                        and frozenset((p, q)) not in VEDHA_EXEMPT]
            rows.append({"planet": PLANETS[p], "house_from_natal_moon": h, "in_favourable_house": True,
                         "vedha_house": vh, "obstructed_by": blockers, "effective": not blockers})
        return {"state": state.summary(), "datetime_local": E.fmt_dt(when), "natal_moon_sign": SIGNS[nat.moon_sign],
                "vedha": rows,
                "note": "Rahu/Ketu are counted as obstructing planets but have no vedha table of their own."}


# ------------------------------------------------------------------------------ murthi nirnaya
def get_murthi_nirnaya(state: BirthState, planet: Any = "Saturn", start: Optional[str] = None,
                       end: Optional[str] = None) -> dict[str, Any]:
    """Murthi (gold/silver/copper/iron form) of each forward sign entry of a slow planet: decided by the Moon's house
    from the natal Moon at the moment of entry. Default: Saturn, next 10 years."""
    p = E.planet_id(planet)
    if p not in (2, 4, 6, 7, 8):
        raise TransitError("Murthi Nirnaya is classically used for Mars, Jupiter, Saturn, Rahu and Ketu.")
    with E._calc(state) as (b, ctx):
        a, z = E._window(state, start, end, 3652, 36525)
        nat = E._natal(state, b, ctx)
        raw = V.find_boundary_changes(b, p, E.jd_utc_from_local(state, ctx, a), E.jd_utc_from_local(state, ctx, z), 30.0)
        events = []
        for r in raw:
            if r["direction"] != "forward" and p not in (7, 8):
                continue
            moon_house = E._house(nat.moon_sign, int(E._lon(b, r["jd"], 1) // 30))
            m = MURTHI[moon_house]
            row = E._event_row(state, ctx, r["jd"])
            row.update({"enters": SIGNS[r["to_index"]], "direction": r["direction"],
                        "house_of_new_sign_from_natal_moon": E._house(nat.moon_sign, r["to_index"]),
                        "moon_house_from_natal_moon": moon_house, "murthi": m, "quality": MURTHI_QUALITY[m]})
            events.append(row)
        return {"state": state.summary(), "planet": PLANETS[p], "window_local": [E.fmt_dt(a), E.fmt_dt(z)],
                "count": len(events), "events": events,
                "note": "Retrograde re-entries are skipped (except for the nodes, which always move backwards)."}


# ------------------------------------------------------------------------------ double transit
def get_double_transit(state: BirthState, on_date: Optional[str] = None, reference: str = "lagna") -> dict[str, Any]:
    """Houses influenced (by occupation or rasi aspect: Jupiter 5/7/9, Saturn 3/7/10) by Jupiter, by Saturn, and by both
    at once ('double transit'), counted from the natal lagna or natal Moon (`reference` = lagna | moon)."""
    ref = reference.strip().lower()
    if ref not in ("lagna", "moon"):
        raise TransitError("`reference` must be 'lagna' or 'moon'.")
    with E._calc(state) as (b, ctx):
        when = E.parse_when(on_date, state)
        jd = E.jd_utc_from_local(state, ctx, when)
        nat = E._natal(state, b, ctx)
        base = nat.lagna_sign if ref == "lagna" else nat.moon_sign
        infl: dict[int, set[int]] = {}
        where = {}
        for p, hs in ((4, (1, 5, 7, 9)), (6, (1, 3, 7, 10))):
            sign = int(E._lon(b, jd, p) // 30)
            where[PLANETS[p]] = {"sign": SIGNS[sign], "house": E._house(base, sign)}
            infl[p] = {E._house(base, (sign + h - 1) % 12) for h in hs}
        both = sorted(infl[4] & infl[6])
        return {"state": state.summary(), "datetime_local": E.fmt_dt(when), "reference": ref,
                "reference_sign": SIGNS[base], "planets": where,
                "houses_influenced_by_jupiter": sorted(infl[4]), "houses_influenced_by_saturn": sorted(infl[6]),
                "double_transit_houses": both}


# ------------------------------------------------------------------------------ planet conditions
def get_planet_conditions(state: BirthState, on_date: Optional[str] = None) -> dict[str, Any]:
    """Per planet at a date: retrograde / stationary, dignity (exalted, debilitated, own sign), combustion (orb from Sun),
    gandanta (within 3d20' of Pisces-Aries, Cancer-Leo, Scorpio-Sagittarius junctions) and graha yuddha (planetary war)."""
    with E._calc(state) as (b, ctx):
        when = E.parse_when(on_date, state)
        jd = E.jd_utc_from_local(state, ctx, when)
        jd_local = E.jd_from_local(when)
        sun = E._lon(b, jd, 0)
        yuddha_ids: list[int] = []
        try:
            raw = E._quiet(b.drik.planets_in_graha_yudh, jd_local, ctx.place)
            flat: list[int] = []
            for item in raw or []:
                flat.extend(item if isinstance(item, (list, tuple)) else [item])
            yuddha_ids = sorted({int(x) for x in flat if isinstance(x, (int, float)) and 0 <= int(x) <= 8})
        except Exception:
            yuddha_ids = []
        rows = []
        for p in range(9):
            st = E._planet_state(b, jd, p)
            lon = st["longitude"]
            sign, deg = st["sign_index"], st["degree_in_sign"]
            cond: dict[str, Any] = {"planet": PLANETS[p], "sign": st["sign"], "degree_in_sign": deg,
                                    "speed_deg_per_day": st["speed_deg_per_day"]}
            if p in (2, 3, 4, 5, 6):
                cond["retrograde"] = st["speed_deg_per_day"] < 0
                typical = {2: 0.5, 3: 1.0, 4: 0.12, 5: 0.9, 6: 0.07}[p]
                cond["stationary"] = abs(st["speed_deg_per_day"]) < 0.1 * typical
            if p in EXALT:
                if sign == EXALT[p][0]:
                    cond["dignity"] = f"exalted (deepest at {EXALT[p][1]:g} deg)"
                elif sign == DEBIL[p][0]:
                    cond["dignity"] = f"debilitated (deepest at {DEBIL[p][1]:g} deg)"
                elif sign in OWN[p]:
                    cond["dignity"] = "own sign"
                else:
                    cond["dignity"] = "neutral/other (friend/enemy not assessed)"
                lord = SIGN_LORDS[sign]
                cond["sign_lord"] = PLANETS[lord]
            if p in COMBUST_ORB:
                sep = abs(E._wrap(lon - sun))
                orb = COMBUST_ORB[p][1 if cond.get("retrograde") else 0]
                cond["separation_from_sun_deg"] = round(sep, 2)
                cond["combust"] = sep <= orb
            junction = min(abs(E._wrap(lon - j)) for j in (0.0, 120.0, 240.0))
            cond["gandanta"] = junction <= 10.0 / 3.0
            cond["in_graha_yuddha"] = p in yuddha_ids
            rows.append(cond)
        return {"state": state.summary(), "datetime_local": E.fmt_dt(when), "planets": rows,
                "note": "Combustion uses classical orbs (Moon 12, Mars 17, Mercury 14/12 retrograde, Jupiter 11, Venus 10/8 "
                        "retrograde, Saturn 15). Graha yuddha comes from PyJHora (empty list if it could not be computed)."}


# ------------------------------------------------------------------------------ panchanga
def get_panchanga(state: BirthState, on_date: Optional[str] = None) -> dict[str, Any]:
    """Panchanga at a moment (the Sun-Moon transit): vara, tithi, nakshatra, yoga, karana, each with its end time."""
    with E._calc(state) as (b, ctx):
        when = E.parse_when(on_date, state)
        jd = E.jd_utc_from_local(state, ctx, when)
        sun, moon = E._lon(b, jd, 0), E._lon(b, jd, 1)
        elong = (moon - sun) % 360.0
        tithi = int(elong // 12.0) + 1
        yoga_sum = (sun + moon) % 360.0
        yoga = int(yoga_sum // NAK_LEN) + 1
        karana = int(elong // 6.0) + 1
        nak = int(moon // NAK_LEN)
        # weekday: the Vedic day runs sunrise to sunrise
        jd_local = E.jd_from_local(when)
        try:
            sr = E._quiet(b.drik.sunrise, jd_local, ctx.place)
            sunrise_jd, sunrise_txt = float(sr[2]), sr[1]
        except Exception:
            sunrise_jd, sunrise_txt = None, None
        day = when.date()
        if sunrise_jd is not None and jd_local < sunrise_jd:
            day = day - timedelta(days=1)
        end_t = jd + 2.5

        def first(rows: list[tuple[float, int]]) -> Optional[str]:
            return E.fmt_dt(E.local_dt_from_utc(state, ctx, rows[0][0])) if rows else None

        tithi_end = first(E.find_crossings(lambda j: E._wrap(((E._lon(b, j, 1) - E._lon(b, j, 0)) % 360.0) - 12.0 * tithi),
                                           jd, end_t, 14.5, 0.01, 1.0, limit=1))
        karana_end = first(E.find_crossings(lambda j: E._wrap(((E._lon(b, j, 1) - E._lon(b, j, 0)) % 360.0) - 6.0 * karana),
                                            jd, end_t, 14.5, 0.01, 1.0, limit=1))
        yoga_end = first(E.find_crossings(lambda j: E._wrap(((E._lon(b, j, 0) + E._lon(b, j, 1)) % 360.0) - NAK_LEN * yoga),
                                          jd, end_t, 16.5, 0.01, 1.0, limit=1))
        nk = V.find_boundary_changes(b, 1, jd, end_t, NAK_LEN, 1)
        nak_end = E.fmt_dt(E.local_dt_from_utc(state, ctx, nk[0]["jd"])) if nk else None
        return {
            "state": state.summary(), "datetime_local": E.fmt_dt(when), "ayanamsa": state.options.ayanamsa,
            "vara": VARA_NAMES[day.weekday()], "sunrise_local": sunrise_txt,
            "tithi": {"number": tithi, "name": _tithi_name(tithi), "ends_local": tithi_end},
            "nakshatra": {"name": NAKSHATRAS[nak], "number": nak + 1, "ends_local": nak_end},
            "yoga": {"number": yoga, "name": YOGA_NAMES[yoga - 1], "ends_local": yoga_end},
            "karana": {"number": karana, "name": _karana_name(karana), "ends_local": karana_end},
            "sun_sign": SIGNS[int(sun // 30)], "moon_sign": SIGNS[int(moon // 30)],
        }


# ------------------------------------------------------------------------------ solstices / equinoxes
def get_ayana_points(state: BirthState, start: Optional[str] = None, end: Optional[str] = None) -> dict[str, Any]:
    """Equinoxes and solstices (Sun at tropical 0/90/180/270 degrees) in a window (default next 12 months). The Vedic
    Uttarayana/Dakshinayana starts are the Makara/Karka Sankrantis; see get_sankrantis."""
    swe = C._swe()
    names = {0.0: "March (vernal) equinox", 90.0: "June solstice (Dakshinayana begins in the tropical sense)",
             180.0: "September (autumnal) equinox", 270.0: "December solstice (Uttarayana begins in the tropical sense)"}
    with E._calc(state) as (b, ctx):
        a, z = E._window(state, start, end, 366, 7305)
        t0, t1 = E.jd_utc_from_local(state, ctx, a), E.jd_utc_from_local(state, ctx, z)

        def tropical_sun(jd: float) -> float:
            return swe.calc_ut(jd, swe.SUN)[0][0] % 360.0

        events = []
        for target, label in names.items():
            for jd, _ in E.find_crossings(lambda j, t=target: E._wrap(tropical_sun(j) - t), t0, t1, 1.1, 0.02, 15.0):
                row = E._event_row(state, ctx, jd)
                row["event"] = label
                events.append(row)
        events.sort(key=lambda r: r["jd_utc"])
        return {"state": state.summary(), "window_local": [E.fmt_dt(a), E.fmt_dt(z)], "count": len(events), "events": events}


# ------------------------------------------------------------------------------ returns
def get_planet_returns(state: BirthState, planet: Any = "Saturn", count: int = 3, start: Optional[str] = None) -> dict[str, Any]:
    """The next `count` times a transiting planet returns to its natal longitude (Saturn return, Jupiter return, lunar
    return ...). Retrograde re-passes are listed with a flag. Start defaults to now."""
    p = E.planet_id(planet)
    if not 1 <= count <= 12:
        raise TransitError("`count` must be 1..12.")
    with E._calc(state) as (b, ctx):
        a = E.parse_when(start, state)
        z = a + timedelta(days=RETURN_DAYS[p] * (count + 0.3))
    res = V.search_transit_to_natal(state, p, PLANETS[p], ["conjunction"], E.fmt_dt(a), E.fmt_dt(z))
    events = res["events"]
    # group passes: a return is a cluster of passes closer than 40% of the planet's period
    gap = RETURN_DAYS[p] * 0.4
    groups: list[list[dict[str, Any]]] = []
    for ev in events:
        if groups and ev["jd_utc"] - groups[-1][-1]["jd_utc"] < gap:
            groups[-1].append(ev)
        else:
            groups.append([ev])
    returns = []
    for i, g in enumerate(groups[:count], 1):
        returns.append({"return_number": i, "first_exact_local": g[0]["datetime_local"],
                        "passes": [{"datetime_local": e["datetime_local"], "retrograde": e["retrograde_at_event"]} for e in g]})
    return {"state": state.summary(), "planet": PLANETS[p], "natal_longitude": res["natal_point_longitude"],
            "returns": returns, "note": "Planets that turn retrograde make 1 or 3 exact passes per return; all are listed."}


# ------------------------------------------------------------------------------ combined calendar
def get_major_transits(state: BirthState, start: Optional[str] = None, end: Optional[str] = None,
                       planets: Optional[list[Any]] = None, include_eclipses: bool = True) -> dict[str, Any]:
    """One chronological list for a window (default now .. +2 years): sign ingresses and stations of the chosen slow planets
    (default Jupiter, Saturn, Rahu, Ketu) plus eclipses, each tagged with the house from the natal Moon."""
    chosen = V._planets_arg(planets, [4, 6, 7, 8])
    with E._calc(state) as (b, ctx):
        a, z = E._window(state, start, end, 730, 36525)
        nat = E._natal(state, b, ctx)
        moon_sign = nat.moon_sign
    events: list[dict[str, Any]] = []
    for p in chosen:
        ing = V.get_ingresses(state, p, "sign", E.fmt_dt(a), E.fmt_dt(z), E.MAX_EVENTS)
        for e in ing["events"]:
            to_idx = SIGNS.index(e["to"])
            events.append({"datetime_local": e["datetime_local"], "jd_utc": e["jd_utc"], "planet": PLANETS[p],
                           "event": f"enters {e['to']} ({e['direction']})",
                           "house_from_natal_moon": E._house(moon_sign, to_idx)})
        if p in E.STATION_STEP:
            st = V.get_stations(state, p, E.fmt_dt(a), E.fmt_dt(z))
            for s in st["stations"]:
                events.append({"datetime_local": s["datetime_local"], "jd_utc": s["jd_utc"], "planet": PLANETS[p],
                               "event": f"stationary {s['station'].split(' ')[0]} in {s['sign']} {s['degree_in_sign']:.1f} deg",
                               "house_from_natal_moon": E._house(moon_sign, SIGNS.index(s["sign"]))})
    if include_eclipses:
        for e in C.get_eclipses(state, E.fmt_dt(a), E.fmt_dt(z), "both")["events"]:
            events.append({"datetime_local": e["datetime_local"], "jd_utc": e["jd_utc"],
                           "planet": "Sun" if e["kind"] == "solar" else "Moon",
                           "event": f"{e['type']} {e['kind']} eclipse in {e['sign']} {e['degree_in_sign']:.1f} deg ({e['nakshatra']})",
                           "house_from_natal_moon": e["house_from_natal_moon"]})
    events.sort(key=lambda r: r["jd_utc"])
    return {"state": state.summary(), "window_local": [E.fmt_dt(a), E.fmt_dt(z)], "count": len(events), "events": events}
