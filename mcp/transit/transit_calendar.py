"""Transit engine, part 3: daily Moon strength, eclipses, sankranti, lagna timeline.

Builds on `transit_engine` and `transit_events`. Dates in and out are LOCAL at the birth place.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Optional

import transit_engine as E
from state import BirthState
from transit_engine import NAK_LEN, NAKSHATRAS, PLANETS, SIGNS, TransitError
from transit_events import find_boundary_changes

SANKRANTI_NAMES = ["Mesha", "Vrishabha", "Mithuna", "Karka", "Simha", "Kanya", "Tula", "Vrischika",
                   "Dhanu", "Makara", "Kumbha", "Meena"]
TARA_NAMES = ["Janma", "Sampat", "Vipat", "Kshema", "Pratyak", "Sadhana", "Naidhana", "Mitra", "Parama Mitra"]
TARA_GOOD = {2, 4, 6, 8, 9}  # Sampat, Kshema, Sadhana, Mitra, Parama Mitra
TARA_MEANING = {
    1: "Janma (birth star): mixed, usually avoided for new starts",
    2: "Sampat: wealth, good", 3: "Vipat: danger, avoid", 4: "Kshema: well-being, good",
    5: "Pratyak: obstacles, avoid", 6: "Sadhana: achievement, good", 7: "Naidhana: death-like, avoid",
    8: "Mitra: friendly, good", 9: "Parama Mitra: very friendly, good",
}
ECL_TOTAL, ECL_ANNULAR, ECL_PARTIAL, ECL_ANNULAR_TOTAL, ECL_PENUMBRAL = 4, 8, 16, 32, 64


def _swe() -> Any:
    b = E._backend()
    swe = getattr(b, "swe", None)
    if swe is not None:
        return swe
    try:
        import swisseph as swe  # PyJHora depends on pyswisseph
    except ImportError as exc:
        raise TransitError("pyswisseph is not installed. Run: pip install -r requirements.txt") from exc
    return swe


# --------------------------------------------------------------------------------------
# Daily Moon: chandrabalam + tarabalam against the natal chart
# --------------------------------------------------------------------------------------
def get_daily_moon(state: BirthState, on_date: Optional[str] = None) -> dict[str, Any]:
    """Moon's sign and nakshatra at a moment, Chandrabalam (Moon's house from the natal Moon), Chandrashtama,
    Tarabalam (count from the birth nakshatra), and when the Moon next changes sign / nakshatra."""
    with E._calc(state) as (b, ctx):
        when = E.parse_when(on_date, state)
        jd = E.jd_utc_from_local(state, ctx, when)
        nat = E._natal(state, b, ctx)
        moon = E._planet_state(b, jd, 1)
        h = E._house(nat.moon_sign, moon["sign_index"])
        birth_nak = int(nat.lon[1] // NAK_LEN)  # 0-based
        tara_no = (moon["nakshatra_index"] - 1 - birth_nak) % 9 + 1
        sign_chg = find_boundary_changes(b, 1, jd, jd + 4.0, 30.0, 1)
        nak_chg = find_boundary_changes(b, 1, jd, jd + 2.0, NAK_LEN, 1)

        def nxt(rows: list[dict[str, Any]], names: list[str]) -> Optional[dict[str, Any]]:
            if not rows:
                return None
            r = rows[0]
            return {**E._event_row(state, ctx, r["jd"]), "to": names[r["to_index"]],
                    "hours_from_now": round((r["jd"] - jd) * 24, 2)}

        return {
            "state": state.summary(), "datetime_local": E.fmt_dt(when),
            "natal": {"moon_sign": SIGNS[nat.moon_sign], "janma_nakshatra": NAKSHATRAS[birth_nak]},
            "transit_moon": {k: moon[k] for k in ("sign", "degree_in_sign", "nakshatra", "pada")},
            "chandrabalam": {
                "house_from_natal_moon": h, "favourable": h in E.GOCHARA_GOOD[1],
                "favourable_houses": sorted(E.GOCHARA_GOOD[1]),
                "chandrashtama": h == 8,
            },
            "tarabalam": {"tara_number": tara_no, "tara": TARA_NAMES[tara_no - 1],
                          "favourable": tara_no in TARA_GOOD, "meaning": TARA_MEANING[tara_no]},
            "next_moon_sign_change": nxt(sign_chg, SIGNS),
            "next_moon_nakshatra_change": nxt(nak_chg, NAKSHATRAS),
            "note": "Rule-of-thumb muhurta inputs only; the verdict is the classical Chandrabala/Tarabala rule, not a prediction.",
        }


# --------------------------------------------------------------------------------------
# Eclipses
# --------------------------------------------------------------------------------------
def _ecl_type(flag: int, solar: bool) -> str:
    if flag & ECL_ANNULAR_TOTAL:
        return "hybrid (annular-total)"
    if flag & ECL_TOTAL:
        return "total"
    if solar and flag & ECL_ANNULAR:
        return "annular"
    if flag & ECL_PARTIAL:
        return "partial"
    if flag & ECL_PENUMBRAL:
        return "penumbral"
    return "unknown"


def get_eclipses(state: BirthState, start: Optional[str] = None, end: Optional[str] = None, kind: str = "both") -> dict[str, Any]:
    """Solar and/or lunar eclipses in a window (default now .. +2 years): global type and maximum time, the
    zodiac sign / nakshatra / house from natal Moon at maximum, and whether it is visible at the birth place."""
    kind = kind.strip().lower()
    if kind not in ("both", "solar", "lunar"):
        raise TransitError("`kind` must be 'solar', 'lunar' or 'both'.")
    swe = _swe()
    with E._calc(state) as (b, ctx):
        a, z = E._window(state, start, end, 730, 36525)
        t0, t1 = E.jd_utc_from_local(state, ctx, a), E.jd_utc_from_local(state, ctx, z)
        nat = E._natal(state, b, ctx)
        geo = (state.longitude, state.latitude, state.elevation_m or 0.0)
        events: list[dict[str, Any]] = []

        def add(solar: bool) -> None:
            jd = t0
            while len(events) < E.MAX_EVENTS:
                flag, tret = (swe.sol_eclipse_when_glob(jd) if solar else swe.lun_eclipse_when(jd))
                tmax = tret[0]
                if tmax > t1:
                    break
                row = E._event_row(state, ctx, tmax)
                sun_lon, moon_lon = E._lon(b, tmax, 0), E._lon(b, tmax, 1)
                focus = sun_lon if solar else moon_lon
                d = E.describe_longitude(focus)
                row.update({
                    "kind": "solar" if solar else "lunar", "type": _ecl_type(flag, solar),
                    "sign": d["sign"], "degree_in_sign": d["degree_in_sign"], "nakshatra": d["nakshatra"],
                    "house_from_natal_moon": E._house(nat.moon_sign, d["sign_index"]),
                    "house_from_natal_lagna": E._house(nat.lagna_sign, d["sign_index"]),
                })
                try:
                    if solar:
                        rl, tl, at = swe.sol_eclipse_when_loc(tmax - 2.0, geo)
                        vis = abs(tl[0] - tmax) < 1.0
                        if vis:
                            row["visible_at_birth_place"] = {
                                "maximum_local": E.fmt_dt(E.local_dt_from_utc(state, ctx, tl[0])),
                                "magnitude": round(at[0], 3), "obscuration": round(at[2], 3)}
                    else:
                        rl, tl, at = swe.lun_eclipse_when_loc(tmax - 2.0, geo)
                        vis = abs(tl[0] - tmax) < 1.0
                        if vis:
                            row["visible_at_birth_place"] = {
                                "maximum_local": E.fmt_dt(E.local_dt_from_utc(state, ctx, tl[0])),
                                "umbral_magnitude": round(at[0], 3), "penumbral_magnitude": round(at[1], 3)}
                    if not vis:
                        row["visible_at_birth_place"] = False
                except Exception:
                    row["visible_at_birth_place"] = None  # local circumstances unavailable for this one
                events.append(row)
                jd = tmax + 10.0

        if kind in ("both", "solar"):
            add(True)
        if kind in ("both", "lunar"):
            add(False)
        events.sort(key=lambda r: r["jd_utc"])
        return {"state": state.summary(), "window_local": [E.fmt_dt(a), E.fmt_dt(z)], "count": len(events),
                "events": events,
                "note": "Times of maximum come from the Swiss Ephemeris. 'visible_at_birth_place' is false when the "
                        "eclipse does not occur above the horizon there; null means that could not be computed."}


# --------------------------------------------------------------------------------------
# Sankranti (Sun's sign ingress)
# --------------------------------------------------------------------------------------
def get_sankrantis(state: BirthState, start: Optional[str] = None, end: Optional[str] = None) -> dict[str, Any]:
    """Every Sankranti (Sun entering a sidereal sign) in a window (default: next 12 months, max 20 years),
    with the Vedic name, e.g. Makara Sankranti."""
    with E._calc(state) as (b, ctx):
        a, z = E._window(state, start, end, 366, 7305)
        raw = find_boundary_changes(b, 0, E.jd_utc_from_local(state, ctx, a), E.jd_utc_from_local(state, ctx, z), 30.0)
        events = []
        for r in raw:
            row = E._event_row(state, ctx, r["jd"])
            row.update({"sankranti": f"{SANKRANTI_NAMES[r['to_index']]} Sankranti", "sun_enters": SIGNS[r["to_index"]]})
            events.append(row)
        return {"state": state.summary(), "ayanamsa": state.options.ayanamsa,
                "window_local": [E.fmt_dt(a), E.fmt_dt(z)], "count": len(events), "events": events}


# --------------------------------------------------------------------------------------
# Lagna (ascendant) through the day
# --------------------------------------------------------------------------------------
def get_lagna_timeline(state: BirthState, on_date: Optional[str] = None, hours: float = 24.0) -> dict[str, Any]:
    """The rising sign at the birth place from `on_date` (default now) for `hours` (max 72): when each lagna
    starts and ends, with the lagna's nakshatra/pada at the start of each stretch."""
    if not 0 < hours <= 72:
        raise TransitError("`hours` must be between 0 and 72.")
    with E._calc(state) as (b, ctx):
        start = E.parse_when(on_date, state)
        end = start + timedelta(hours=hours)

        def asc(dt: Any) -> list[Any]:
            return E._quiet(b.drik.ascendant, E.jd_from_local(dt), ctx.place)

        def sign_at(dt: Any) -> int:
            return int(asc(dt)[0])

        step = timedelta(minutes=10)
        edges = [start]
        t = start
        cur = sign_at(t)
        while t < end:
            nt = min(t + step, end)
            ns = sign_at(nt)
            if ns != cur:
                lo, hi = t, nt
                while (hi - lo).total_seconds() > 1:
                    mid = lo + (hi - lo) / 2
                    if sign_at(mid) == cur:
                        lo = mid
                    else:
                        hi = mid
                edges.append(hi.replace(microsecond=0))
            t, cur = nt, ns
        edges.append(end)
        rows = []
        for i in range(len(edges) - 1):
            probe = edges[i] + timedelta(seconds=2) if i else edges[i]
            res = asc(probe)
            nak, pada = int(res[2]), int(res[3])
            rows.append({"lagna": SIGNS[int(res[0])], "from_local": E.fmt_dt(edges[i]), "to_local": E.fmt_dt(edges[i + 1]),
                         "minutes": round((edges[i + 1] - edges[i]).total_seconds() / 60, 1),
                         "lagna_nakshatra_at_start": NAKSHATRAS[(nak - 1) % 27], "pada_at_start": pada})
        return {"state": state.summary(), "window_local": [E.fmt_dt(start), E.fmt_dt(end)], "lagna_periods": rows,
                "note": "First and last rows are cut at the window edges; the sign may have started earlier / end later."}
