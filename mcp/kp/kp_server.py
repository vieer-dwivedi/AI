"""KP (Krishnamurti Paddhati) MCP server. Free and local: Swiss Ephemeris + our own KP tables, no web APIs.

Every natal tool takes the birth `state` (KPState); moment tools take `moment` (MomentState). The server keeps no session data.
Tools state facts and classical KP rules only; they never predict outcomes. Every judgment returns the houses and levels it used.

Run as a script (NOT `python -m`) so the folder name `mcp` does not shadow the official `mcp` SDK:

    python mcp/kp/main.py            # stdio
    python mcp/kp/main.py --http     # streamable-http on 127.0.0.1:8002/mcp
"""

from __future__ import annotations

import functools
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from mcp.server.fastmcp import FastMCP

import kp_annual as an
import kp_chart as ch
import kp_cil as cil
import kp_core as core
import kp_dasha as da
import kp_double as dbl
import kp_fruitful as ff
import kp_horary as hy
import kp_longevity as lg
import kp_medical as med
import kp_methods as km
import kp_moment as mo
import kp_place as pl
import kp_muhurta as mh
import kp_report as rep
import kp_workflow as wf
import kp_retro as rt
import kp_transit as tr
import kp_twins as twins
from kp_tables import EVENTS, HOUSE_THEMES, division_row, dms, kp_lords, planet_name, sign_name_index
from state import KPState, MomentState, now_at

mcp = FastMCP(
    "kp",
    instructions=(
        "Krishnamurti Paddhati (KP) astrology: sign / star / sub / sub-sub lords, Placidus cusps, significators (A-D), cuspal "
        "sub-lord promise tests, Vimshottari timing with the KP year (365.25 days), ruling planets, horary 1-249, transit "
        "triggers and birth-time rectification. Default ayanamsa is KP New (swisseph VP291). Pass the same birth `state` to every "
        "natal tool. Results are rule-based facts, not predictions: always show the houses and levels used. KP is very sensitive "
        "to birth time; when the time is uncertain use get_lagna_changes and rectify_birth_time first."
    ),
)

DATE_HELP = "Local date/time at the place: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS. Omit for 'now'."


def tool(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Register `fn` as an MCP tool; engine errors become clean tool errors."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except (core.KPError, da.DashaError) as exc:
            raise ValueError(str(exc)) from exc

    return mcp.tool()(wrapper)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _natal(ctx: core.Ctx, state: KPState):
    dt = state.local_datetime
    return ch.chart_at(ctx, ctx.jd_from_local(dt), state.latitude, state.longitude, dt), dt


def _natal_chart(state: KPState):
    with core.calc(state) as ctx:
        return _natal(ctx, state)[0]


def _align(moment: MomentState, birth: Optional[KPState]):
    """Moment state using the birth's options (one ayanamsa for both), plus the natal chart (computed before, never nested)."""
    if birth is None:
        return moment, None
    return moment.model_copy(update={"options": birth.options}), _natal_chart(birth)


def _event_houses(chart, event: Optional[str], custom_favourable: Optional[list[int]], custom_denial: Optional[list[int]],
                  primary_cusp: Optional[int] = None) -> dict[str, Any]:
    j = ch.judge_event(chart, event, custom_favourable, custom_denial, primary_cusp)
    return {"label": j["label"], "event": j["event"], "favourable": j["favourable_houses"], "denial": j["denial_houses"],
            "primary_cusp": j["primary_cusp"], "promise": j}


def _window(state: KPState, start: Optional[str], end: Optional[str], default_years: float) -> tuple[datetime, datetime]:
    s = core.parse_dt(start, now_at(state))
    e = core.parse_dt(end, s + timedelta(days=365.25 * default_years))
    if e <= s:
        raise core.KPError("`end` must be after `start`.")
    return s, e


def _moment_rps(moment: Optional[MomentState], state: KPState, extended: bool = True) -> Optional[list[str]]:
    """Ruling planets of `moment` (with the birth's options). Computed outside any other calc context."""
    if moment is None:
        return None
    m = moment.model_copy(update={"options": state.options})
    with core.calc(m) as ctx:
        r = mo.ruling_planets(ctx, m.local_datetime, m.latitude, m.longitude, extended)
    return [x["planet"] for x in r["ruling_planets"]]


# --------------------------------------------------------------------------------------
# Reference tools (no birth data)
# --------------------------------------------------------------------------------------
@tool
def get_kp_sub_table(number: Optional[int] = None, sign: Optional[str] = None) -> dict[str, Any]:
    """The 249 numbered KP divisions (243 subs; 6 are split by a sign boundary). Each row: number, start/end longitude
    (sidereal, degrees and DMS), sign, nakshatra, star lord, sub lord. Give `number` (1-249) for one row, or `sign`
    (e.g. 'Aries') for that sign's rows; no argument returns the whole table."""
    if number is not None:
        try:
            return {"rows": [division_row(number)]}
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
    rows = [division_row(i) for i in range(1, 250)]
    if sign is not None:
        idx = sign_name_index(sign)
        rows = [r for r in rows if r["sign_index"] == idx]
    return {"count": len(rows), "rows": rows,
            "note": "Horary numbers 1-249 index these rows. Sub span = lord's Vimshottari years / 120 x 13d20m."}


@tool
def lookup_longitude(longitude: Optional[float] = None, sign: Optional[str] = None, degree_in_sign: Optional[float] = None) -> dict[str, Any]:
    """Sign, nakshatra, star / sub / sub-sub / sookshma lords and KP number (1-249) of a SIDEREAL longitude. Give `longitude`
    (0-360 degrees) or `sign` + `degree_in_sign` (0-30). Pure table lookup: no ephemeris."""
    if longitude is None:
        if sign is None or degree_in_sign is None:
            raise ValueError("Give `longitude`, or both `sign` and `degree_in_sign`.")
        if not 0 <= degree_in_sign < 30:
            raise ValueError("degree_in_sign must be 0 <= x < 30.")
        longitude = sign_name_index(sign) * 30.0 + degree_in_sign
    elif not 0 <= longitude < 360:
        raise ValueError("longitude must be 0 <= x < 360.")
    return kp_lords(longitude)


@tool
def get_ayanamsa(on_date: Optional[str] = None) -> dict[str, Any]:
    """Ayanamsa value of the supported modes (KP_NEW default, KP_OLD, LAHIRI, RAMAN) at a date (default today), with
    differences from Lahiri. Do not mix ayanamsas between charts."""
    d = core.parse_dt(on_date, datetime.now().replace(microsecond=0))
    jd = core.julday_utc(d.year, d.month, d.day, d.hour + d.minute / 60.0 + d.second / 3600.0)
    vals = core.ayanamsa_values(jd)
    lah = vals["LAHIRI"]
    return {"date_utc": d.strftime("%Y-%m-%dT%H:%M:%S"), "jd_ut": round(jd, 6),
            "values": {k: {"deg": round(v, 6), "dms": dms(v), "minus_lahiri_arcsec": round((v - lah) * 3600.0, 1)} for k, v in vals.items()},
            "note": ("KP_NEW = Swiss Ephemeris Krishnamurti VP291 (matches the published KP New ayanamsa, about Lahiri minus 4m36s). "
                     "KP_OLD = Swiss Ephemeris Krishnamurti (about Lahiri minus 5m49s).")}


@tool
def list_event_types() -> dict[str, Any]:
    """Every matter the server can judge (marriage, child_birth, service, foreign_travel ...) with the cusp whose sub lord is
    tested, the favourable houses and the denial houses. Textbook KP convention; schools differ, so custom houses are accepted
    by judge_event / find_event_periods."""
    rows = [{"event": k, "label": v["label"], "primary_cusp": v["primary_cusp"], "favourable": v["favourable"],
             "denial": v["denial"], "note": v["note"]} for k, v in EVENTS.items()]
    return {"count": len(rows), "events": rows}


@tool
def get_house_themes() -> dict[str, Any]:
    """Signification of each house (1-12) and the badhaka rule used for the 'longevity' denial group."""
    return {"houses": [{"house": h, "themes": t} for h, t in HOUSE_THEMES.items()],
            "badhaka": {"movable_lagna": 11, "fixed_lagna": 9, "dual_lagna": 7},
            "significator_levels": ch.LEVEL_NAMES}


# --------------------------------------------------------------------------------------
# Natal chart tools
# --------------------------------------------------------------------------------------
@tool
def get_kp_chart(state: KPState, detail: str = "standard") -> dict[str, Any]:
    """Full KP chart: ascendant, 12 Placidus cusps and 9 planets (sign / nakshatra / star / sub / sub-sub lords, KP number,
    house, retrograde), Fortuna. detail='basic' stops there; 'standard' (default) adds planet significations (levels A-D)
    and house significators; 'full' adds `node_check` (Rahu / Ketu with the mean and the true node side by side, flagging a changed star /
    sub lord)."""
    if detail not in ("basic", "standard", "full"):
        raise ValueError("detail must be 'basic', 'standard' or 'full'.")
    with core.calc(state) as ctx:
        chart, _ = _natal(ctx, state)
        return ch.chart_dict(chart, ch.meta_for(state, chart), detail)


@tool
def get_cusps(state: KPState) -> dict[str, Any]:
    """The 12 cusps with sign, star lord, sub lord, sub-sub lord, KP number and occupants."""
    with core.calc(state) as ctx:
        chart, _ = _natal(ctx, state)
        out = {"birth": state.summary(), "house_system_used": chart.houses["system"],
               "ascendant": kp_lords(chart.asc), "cusps": ch.cusp_rows(chart)}
        if chart.houses.get("warning"):
            out["warnings"] = [chart.houses["warning"]]
        return out


@tool
def get_planets(state: KPState) -> dict[str, Any]:
    """Planet table: longitude, sign, nakshatra and pada, star / sub / sub-sub lord, KP number, house, houses owned,
    retrograde and speed."""
    with core.calc(state) as ctx:
        chart, _ = _natal(ctx, state)
        return {"birth": state.summary(), "planets": ch.planet_rows(chart)}


@tool
def get_significators(state: KPState, method: Optional[str] = None, include_type_e: Optional[bool] = None) -> dict[str, Any]:
    """Significators in both views. House view: for each house, A (planets in the star of an occupant), B (occupants),
    C (planets in the star of the owner), D (owner). Planet view: houses each planet signifies by level A-D. Rahu/Ketu are
    agents and also carry the houses of the planets they conjoin, aspect, and of their star and sign lords.
    `method` (default = state.options.significator_method, else 'kp4level'): 'kp4level' = Krishnamurti A-D; 'four_step' =
    Gondhalekar / Shri Bhatt 4-step theory (planet, star lord, sub lord, sub's star lord; every house tagged primary or
    secondary); 'six_fold' = A-D plus planets in the sub of the occupant and of the owner. The A-D table is always returned
    too, and judgment tools keep using it.
    `include_type_e` (default = state.options.include_type_e, off): adds a separate `type_e` block with the E-grade significators
    (planets conjunct / aspecting the significators; basis and orb in the options). Never mixed into A-D.
    Option `cusp_orb_deg` (off by default, e.g. 3.3333) adds a `near_cusp` block (planets close behind / just past a cusp) and
    `signifies_with_near_cusp` per planet; A-D change only when `cusp_proximity_apply` is also set."""
    m = km.normalize_method(method or state.options.significator_method)
    with core.calc(state) as ctx:
        chart, _ = _natal(ctx, state)
        out = {"birth": state.summary(), "method": m, "levels": ch.LEVEL_NAMES, "house_significators": ch.house_significators(chart),
               "planet_significations": ch.planet_significations(chart),
               "node_agents": chart.agents,
               "note": "A is strongest, D weakest. A planet with no planet in its star and in its own star is 'self-strength'."}
        if chart.near_cusp_orb:
            out["near_cusp"] = ch.near_cusp_block(chart)
        if m == "four_step":
            out["four_step"] = km.four_step(chart)
        elif m == "six_fold":
            out["six_fold"] = km.six_fold(chart)
        o = state.options
        if include_type_e if include_type_e is not None else o.include_type_e:
            out["type_e"] = km.type_e(chart, o.type_e_basis, o.type_e_orb_deg)
        return out


@tool
def get_planet_significations(state: KPState, planet: Optional[str] = None, include_sub_level: bool = False) -> dict[str, Any]:
    """Houses signified by one planet (or all) by level A-D, with planets in its star / sub, retrograde flag and, for
    Rahu/Ketu, their agents. `planet` accepts English or Sanskrit names (Shani, Guru ...). With `include_sub_level=True`
    each planet also gets the 6-fold sub levels (houses occupied / owned by its sub lord) and `signifies_6fold`."""
    with core.calc(state) as ctx:
        chart, _ = _natal(ctx, state)
        rows = ch.planet_significations(chart)
        if planet is not None:
            p = planet_name(planet)
            rows = [r for r in rows if r["planet"] == p]
        if include_sub_level:
            for r in rows:
                sf = km.six_fold_planet(chart, r["planet"])
                r["sub_levels"] = {"A_sub": sf["A_sub"], "C_sub": sf["C_sub"], "sub_lord": sf["sub_lord"]}
                r["signifies_6fold"] = sorted(set(r["signifies"]) | set(sf["A_sub"]) | set(sf["C_sub"]))
        return {"birth": state.summary(), "planets": rows, "levels": ch.LEVEL_NAMES}


@tool
def get_cuspal_sublord_analysis(state: KPState, cusps: Optional[list[int]] = None) -> dict[str, Any]:
    """Cuspal sub lord (CSL) report for the given cusps (default all 12): the CSL, its star lord, the houses the CSL
    signifies by level A-D, and the matters (events) whose promise is tested at that cusp with the resulting verdict.
    Option `csl_extension` (off by default; own_bhava | sub_chain | both; not pure KP) adds a separate `csl_extension` block
    per cusp with the widened houses; judge_event then also returns an `extended_reading` beside the unchanged verdict."""
    cs = cusps or list(range(1, 13))
    if any((not isinstance(c, int)) or not 1 <= c <= 12 for c in cs):
        raise ValueError("cusps must be integers 1..12.")
    with core.calc(state) as ctx:
        chart, _ = _natal(ctx, state)
        table = ch.promise_table(chart)
        out = []
        for c in cs:
            rep = ch.csl_report(chart, c)
            rep["theme"] = HOUSE_THEMES[c]
            rep["matters_tested_here"] = [t for t in table if t["primary_cusp"] == c]
            out.append(rep)
        return {"birth": state.summary(), "cusps": out}


@tool
def get_cuspal_interlinks(state: KPState, cusps: Optional[list[int]] = None, event: Optional[str] = None,
                          custom_favourable: Optional[list[int]] = None, custom_denial: Optional[list[int]] = None,
                          primary_cusp: Optional[int] = None, levels: Optional[str] = None) -> dict[str, Any]:
    """Cuspal interlinks (CIL; S.P. Khullar / K. Bhaskaran school, NOT pure KP). For each cusp (default all 12): sign / star / sub /
    sub-sub lord, the houses the cuspal sub lord (CSL) and the sub-sub lord (SSL) signify, and the links between cusps: which cusps
    the CSL, its star lord and its sub lord are themselves CSL of. `cusp_rulership` lists, per planet, the cusps it rules at each
    level. With `event` (or `custom_favourable` / `custom_denial` / `primary_cusp`) it also applies the same favourable / denial
    house test to the SSL next to the CSL's standard verdict. Facts only, no prediction."""
    with core.calc(state) as ctx:
        chart, _ = _natal(ctx, state)
        judged = ch.judge_event(chart, event, custom_favourable, custom_denial, primary_cusp, levels) if (event or custom_favourable) else None
        out = cil.interlinks(chart, cusps, levels, judged)
        out["birth"] = state.summary()
        return out


@tool
def judge_event(state: KPState, event: Optional[str] = None, custom_favourable: Optional[list[int]] = None,
                custom_denial: Optional[list[int]] = None, primary_cusp: Optional[int] = None,
                levels: Optional[str] = None) -> dict[str, Any]:
    """KP promise test for a matter. Takes the cuspal sub lord of the matter's cusp and checks which favourable and denial
    houses it signifies (levels default 'ABCD'). `event` is a name from list_event_types; or give `custom_favourable`
    (+ `custom_denial`, `primary_cusp`). Returns the verdict, the houses used, and the planets that signify the matter.
    Rule-based: a verdict is not a prediction."""
    with core.calc(state) as ctx:
        chart, _ = _natal(ctx, state)
        out = ch.judge_event(chart, event, custom_favourable, custom_denial, primary_cusp, levels)
        out["birth"] = state.summary()
        return out


@tool
def get_promise_table(state: KPState, levels: Optional[str] = None) -> dict[str, Any]:
    """Promise verdict for every known matter at once (cuspal sub lord, favourable and denial houses it signifies)."""
    with core.calc(state) as ctx:
        chart, _ = _natal(ctx, state)
        return {"birth": state.summary(), "matters": ch.promise_table(chart, levels),
                "note": "Textbook house groups; use judge_event for the full reasoning of one matter."}


@tool
def get_fortuna(state: KPState) -> dict[str, Any]:
    """Part of Fortune (Fortuna) as used in KP: Ascendant + Moon - Sun, with its sign, star, sub and house."""
    with core.calc(state) as ctx:
        chart, _ = _natal(ctx, state)
        return {"birth": state.summary(), **ch.fortuna(chart)}


# --------------------------------------------------------------------------------------
# Dasha tools (KP Vimshottari)
# --------------------------------------------------------------------------------------
@tool
def get_kp_dasha(state: KPState, level: int = 1, start: Optional[str] = None, end: Optional[str] = None,
                 parent: Optional[list[str]] = None, with_significations: bool = True) -> dict[str, Any]:
    """Vimshottari periods with the KP year (default 365.25 days). level 1 = Dasha, 2 = Bhukti, 3 = Antara, 4 = Sookshma,
    5 = Prana. `parent` is a lord path such as ['Rahu', 'Jupiter'] to list the periods inside it. Window defaults: level 1 =
    whole life; deeper levels = birth + 10 years (or the parent's span). Each row can include the houses its lord signifies."""
    with core.calc(state) as ctx:
        chart, bdt = _natal(ctx, state)
        d = da.make_dasha(chart, bdt)
        if parent:
            par = d.find(parent)
            s = core.parse_dt(start, par["start"])
            e = core.parse_dt(end, par["end"])
        elif level == 1:
            s = core.parse_dt(start, bdt)
            e = core.parse_dt(end, bdt + timedelta(days=365.25 * 125))
        else:
            s = core.parse_dt(start, bdt)
            e = core.parse_dt(end, s + timedelta(days=365.25 * 10))
        rows = da.periods_table(d, chart, level, s, e, parent, with_significations)
        return {"birth": state.summary(), "balance_at_birth": d.balance_at_birth(), "dasha_year_days": chart.opts.dasha_year_days,
                "level": da.LEVEL_LABELS[level], "window": [core.fmt_dt(s), core.fmt_dt(e)], "count": len(rows), "periods": rows}


@tool
def get_current_kp_dasha(state: KPState, on_date: Optional[str] = None, depth: int = 4) -> dict[str, Any]:
    """Running Dasha / Bhukti / Antara / Sookshma at a date (default today) with start, end, percent elapsed, days
    remaining, and the houses each lord signifies. depth 1-5. """
    if not 1 <= depth <= 5:
        raise ValueError("depth must be 1..5")
    with core.calc(state) as ctx:
        chart, bdt = _natal(ctx, state)
        d = da.make_dasha(chart, bdt)
        at = core.parse_dt(on_date, now_at(state))
        return {"birth": state.summary(), "on_date": core.fmt_dt(at), "dasha_year_days": chart.opts.dasha_year_days,
                "running": da.running_ladder(d, chart, at, depth)}


@tool
def get_dasha_significations(state: KPState, on_date: Optional[str] = None, depth: int = 3, event: Optional[str] = None,
                             custom_favourable: Optional[list[int]] = None, custom_denial: Optional[list[int]] = None) -> dict[str, Any]:
    """For the running Dasha / Bhukti / Antara (...) lords at a date: each lord's houses by level A-D, its sub lord's houses,
    and, if an `event` (or custom houses) is given, which favourable and denial houses each lord and its sub lord signify.
    KP: an event is possible in the joint period of lords that signify its houses."""
    if not 1 <= depth <= 5:
        raise ValueError("depth must be 1..5")
    with core.calc(state) as ctx:
        chart, bdt = _natal(ctx, state)
        d = da.make_dasha(chart, bdt)
        at = core.parse_dt(on_date, now_at(state))
        ladder = da.running_ladder(d, chart, at, depth)
        lords = [r["lords"][-1] for r in ladder]
        rows = []
        for r in ladder:
            p = r["lords"][-1]
            rows.append({**r, "levels": {L: chart.sig[p][L] for L in "ABCD"}, "star_lord": chart.planets[p]["star_lord"],
                         "sub_lord": chart.planets[p]["sub_lord"], "house": chart.house_of[p], "owns": chart.owned[p]})
        out: dict[str, Any] = {"birth": state.summary(), "on_date": core.fmt_dt(at), "running": rows}
        if event or custom_favourable:
            ev = _event_houses(chart, event, custom_favourable, custom_denial)
            out["event"] = {k: ev[k] for k in ("label", "event", "favourable", "denial", "primary_cusp")}
            out["event"]["promise_verdict"] = ev["promise"]["verdict"]
            out["lord_vs_event"] = ch.lord_significations(chart, lords, ev["favourable"], ev["denial"])
        return out


@tool
def find_event_periods(state: KPState, event: Optional[str] = None, custom_favourable: Optional[list[int]] = None,
                       custom_denial: Optional[list[int]] = None, start: Optional[str] = None, end: Optional[str] = None,
                       depth: int = 3, exclude_denial: bool = False, min_coverage: Optional[int] = None,
                       ruling_planets: Optional[list[str]] = None, limit: int = 300,
                       fruitful_method: Optional[str] = None, moment: Optional[MomentState] = None) -> dict[str, Any]:
    """KP timing: periods in a window (default today .. +5 years) where every lord down to `depth` (2 = Dasha+Bhukti,
    3 = +Antara, 4 = +Sookshma) signifies at least one favourable house of the matter, and together they cover at least
    `min_coverage` favourable houses (default 2). `exclude_denial=True` also drops lords that touch denial houses. Pass
    `ruling_planets` to flag the lords that are ruling planets. The result starts with the promise verdict of the cuspal sub
    lord: periods only matter if the matter is promised.
    `fruitful_method` (default none) keeps only lords that are fruitful significators: 'detrimental', 'sub_lord',
    'ruling_planets' (needs `moment`), 'four_step_primary'; comma-separate to combine (see get_fruitful_significators).
    Every lord of a returned period must pass. `moment` = the moment of judgment (place, optional date/time) for the ruling
    planets; they are also used for the `ruling_planet_lords` flag when `ruling_planets` is not given.
    Retrograde: periods whose lords are retrograde (or in a retrograde star) carry `retrograde_flags`; with
    options.retrograde_rule='strict' they are also marked `delayed` and `retrograde_transit_info` gives the next direct station."""
    methods = ff.parse_methods(fruitful_method)
    moment_rp = _moment_rps(moment, state) if moment is not None else None
    with core.calc(state) as ctx:
        chart, bdt = _natal(ctx, state)
        d = da.make_dasha(chart, bdt)
        ev = _event_houses(chart, event, custom_favourable, custom_denial)
        s, e = _window(state, start, end, 5)
        rp = [planet_name(x) for x in ruling_planets] if ruling_planets else moment_rp
        allowed = None
        fruit = None
        if methods:
            fruit = ff.analyze(chart, ev["favourable"], ev["denial"], methods, rp)
            allowed = set(fruit["fruitful"])
        rule = state.options.retrograde_rule
        res = da.find_joint_periods(d, chart, ev["favourable"], ev["denial"], s, e, depth, exclude_denial, min_coverage, rp, None, limit,
                                    allowed, rule)
        flagged = sorted({f["retrograde_planet"] for p_ in res["periods"] for f in p_.get("retrograde_flags", [])})
        pr = ev["promise"]
        if fruit is not None:
            res["fruitful_selection"] = {k2: fruit[k2] for k2 in ("methods", "fruitful", "dropped", "ruling_planets_used")}
        res_out = {"birth": state.summary(), "event": ev["event"], "label": ev["label"], "primary_cusp": ev["primary_cusp"],
                "promise": {"cuspal_sub_lord": pr["csl"]["sub_lord"], "verdict": pr["verdict"], "reason": pr["verdict_reason"],
                            "favourable_signified_by_csl": pr["favourable_signified_by_csl"],
                            "denial_signified_by_csl": pr["denial_signified_by_csl"]},
                "window": [core.fmt_dt(s), core.fmt_dt(e)], **res}
    if rule == "strict" and flagged:
        with core.calc(state) as ctx2:
            res_out["retrograde_transit_info"] = {
                p_: {k3: v3 for k3, v3 in rt.find_stations(ctx2, p_, s, e).items() if k3 in
                     ("retrograde_at_start", "next_direct_station", "next_retrograde_station", "searched_beyond_window")}
                for p_ in flagged if p_ in rt.STATION_PLANETS}
    return res_out


@tool
def get_fruitful_significators(state: KPState, event: Optional[str] = None, custom_favourable: Optional[list[int]] = None,
                               custom_denial: Optional[list[int]] = None, methods: str = "detrimental",
                               moment: Optional[MomentState] = None, levels: Optional[str] = None) -> dict[str, Any]:
    """Which significators of a matter are 'fruitful' (KP selection). Candidates = planets signifying a favourable house
    (levels A-D). `methods` (comma-separate to combine, a planet must pass all): 'detrimental' = drop planets that also
    signify a denial house; 'sub_lord' = keep planets whose sub lord signifies a favourable house; 'ruling_planets' = keep
    planets that are ruling planets at `moment` (required for this method; place plus optional date/time, default now);
    'four_step_primary' = keep primary significators of the 4-step school. Returns every planet with the reason it was kept or
    dropped. Schools differ; this states the rule used and is not a prediction."""
    methods_l = ff.parse_methods(methods)
    rp = _moment_rps(moment, state) if (moment is not None and "ruling_planets" in methods_l) else None
    with core.calc(state) as ctx:
        chart, _ = _natal(ctx, state)
        ev = _event_houses(chart, event, custom_favourable, custom_denial)
        res = ff.analyze(chart, ev["favourable"], ev["denial"], methods_l, rp, levels)
        pr = ev["promise"]
        return {"birth": state.summary(), "event": ev["event"], "label": ev["label"], "primary_cusp": ev["primary_cusp"],
                "promise": {"cuspal_sub_lord": pr["csl"]["sub_lord"], "verdict": pr["verdict"]}, **res}


@tool
def analyze_joint_period(state: KPState, path: list[str], event: Optional[str] = None,
                         custom_favourable: Optional[list[int]] = None, custom_denial: Optional[list[int]] = None) -> dict[str, Any]:
    """Audit one period. `path` = lords from Dasha down, e.g. ['Venus', 'Sun', 'Moon']. For each lord: houses signified, which
    favourable / denial houses of the matter they touch, and the same for the lord's sub lord (KP 'permission' check)."""
    if not path:
        raise ValueError("`path` needs at least the Dasha lord.")
    with core.calc(state) as ctx:
        chart, bdt = _natal(ctx, state)
        d = da.make_dasha(chart, bdt)
        per = d.find(path)
        ev = _event_houses(chart, event, custom_favourable, custom_denial)
        fav, den = set(ev["favourable"]), set(ev["denial"])
        stat = [da.lord_status(chart, lord, fav, den) for lord in per["lords"]]
        cover = sorted(set().union(*[set(x["favourable"]) for x in stat]))
        return {"birth": state.summary(), "period": da.row(per), "event": ev["event"], "label": ev["label"],
                "favourable_houses": sorted(fav), "denial_houses": sorted(den - fav),
                "promise_verdict": ev["promise"]["verdict"], "lords": stat,
                "every_lord_signifies_favourable": all(x["favourable"] for x in stat),
                "covered_favourable_houses": cover, "denial_touched_by": [x["lord"] for x in stat if x["denial"]],
                "sub_lords_blocking": [x["lord"] for x in stat if x["sub_lord_denial"] and not x["sub_lord_favourable"]],
                "retrograde": {"rule": state.options.retrograde_rule, **rt.period_flags(chart, per["lords"], state.options.retrograde_rule)},
                "rule": "KP: the lords must signify the matter's houses; a lord whose sub lord only signifies denial houses may withhold."}


# --------------------------------------------------------------------------------------
# Moment tools: ruling planets, horary, transit, boundary searches
# --------------------------------------------------------------------------------------
@tool
def get_ruling_planets(moment: MomentState, extended: bool = True, birth: Optional[KPState] = None) -> dict[str, Any]:
    """Ruling planets at a moment and place (default now): day lord (sunrise to sunrise), Moon sign / star (/ sub) lords and
    Lagna sign / star (/ sub) lords, with counts, strength hints (planet in star / sub of another ruling planet) and the planets
    that Rahu/Ketu represent. `extended=False` drops the two sub lords. Give `birth` to add the houses each ruling planet
    signifies in the natal chart (and use the birth's ayanamsa)."""
    m, natal = _align(moment, birth)
    with core.calc(m) as ctx:
        return mo.ruling_planets(ctx, m.local_datetime, m.latitude, m.longitude, extended, natal)


@tool
def get_ruling_planets_timeline(moment: MomentState, hours: float = 6.0, step_minutes: float = 5.0, extended: bool = True) -> dict[str, Any]:
    """How the ruling planets change from a moment over the next `hours` (default 6), sampled every `step_minutes`: which
    role changed and when (about). Useful to pick a moment when wanted planets align."""
    with core.calc(moment) as ctx:
        return mo.ruling_planets_timeline(ctx, moment.local_datetime, moment.latitude, moment.longitude, hours, step_minutes, extended)


@tool
def cast_horary_chart(moment: MomentState, number: int, lagna_point: str = "start") -> dict[str, Any]:
    """KP horary (Prashna) chart. `number` 1-249 fixes the Lagna to that numbered division; the cusps come from the moment (same
    place, within a day of the question) when the Ascendant is at the chosen point; planets are those of the question moment.
    lagna_point: 'start' (default), 'middle', 'end' of the division, or 'actual_if_inside' (the real Ascendant if it falls in the
    division). Textbooks differ on this point; the result says which was used."""
    with core.calc(moment) as ctx:
        dt = moment.local_datetime
        chart, info = mo.cast_horary(ctx, number, dt, moment.latitude, moment.longitude, lagna_point)
        meta = {"horary": info, "question": moment.summary()}
        return ch.chart_dict(chart, meta, "standard")


@tool
def judge_horary(moment: MomentState, number: int, event: Optional[str] = None, custom_favourable: Optional[list[int]] = None,
                 custom_denial: Optional[list[int]] = None, primary_cusp: Optional[int] = None, lagna_point: str = "start",
                 extended_rp: bool = True, asked_before: bool = False) -> dict[str, Any]:
    """Cast the horary chart for `number` and judge the matter: cuspal sub lord of the matter's cusp, favourable / denial houses
    it signifies, whether it (or its star lord) is a ruling planet, and the Moon sub lord check. `horary_extras` adds the
    genuineness checks (Lagna star lord linked to the Moon sub lord; Moon star lord signifying the matter), a structured
    `horary_checklist` (CSL not retrograde / not in a retrograde star / linked to supporting houses / ruling-planet links), an
    explanation of the `lagna_point` used, and the first-asking-only reminders (`asked_before=true` flags a repeated question).
    Every line is a fact marked True / False; no overall verdict."""
    with core.calc(moment) as ctx:
        dt = moment.local_datetime
        chart, info = mo.cast_horary(ctx, number, dt, moment.latitude, moment.longitude, lagna_point)
        rp = mo.ruling_planets(ctx, dt, moment.latitude, moment.longitude, extended_rp, None, chart)
        j = mo.judge_horary(chart, rp, event, custom_favourable, custom_denial, primary_cusp)
        return {"horary": info, "question": moment.summary(), "judgment": j, "ruling_planets": rp,
                "horary_extras": hy.horary_extras(chart, info, rp, j, None, asked_before),
                "planets": ch.planet_rows(chart), "cusps": ch.cusp_rows(chart)}


@tool
def get_transit_snapshot(moment: MomentState, birth: Optional[KPState] = None) -> dict[str, Any]:
    """Planet positions at a moment (default now) with sign / star / sub lords and retrograde flag. With `birth`, each planet also
    gets its natal house and the natal houses signified by its star lord and sub lord."""
    m, natal = _align(moment, birth)
    with core.calc(m) as ctx:
        return mo.transit_snapshot(ctx, m.local_datetime, natal)


@tool
def find_boundary_changes(moment: MomentState, planet: str, start: str, end: str, level: str = "sub", limit: int = 500) -> dict[str, Any]:
    """When a planet changes sign / star / sub / sub-sub lord between `start` and `end` (local times; the moment's place gives the
    time zone). level: 'sign', 'star', 'sub' (default) or 'sub_sub'. Exact to about a second. Works for Sun..Ketu."""
    p = planet_name(planet)
    with core.calc(moment) as ctx:
        return mo.find_boundary_changes(ctx, p, core.parse_dt(start), core.parse_dt(end), level, limit)


@tool
def find_stations(moment: MomentState, planet: str, start: Optional[str] = None, end: Optional[str] = None) -> dict[str, Any]:
    """Transit station dates of Mercury, Venus, Mars, Jupiter or Saturn: when it turns retrograde and when it turns direct between
    `start` and `end` (local times at the moment's place; default today .. +400 days), exact to about a second, with the sign / star /
    sub lord at each station. Also says whether the planet is retrograde at `start` and the next direct / retrograde station (searching
    up to about two years beyond the window if needed). Use it to see when a result held back by a retrograde planet can mature."""
    p = planet_name(planet)
    s = core.parse_dt(start, now_at(moment))
    e = core.parse_dt(end, s + timedelta(days=400))
    with core.calc(moment) as ctx:
        return rt.find_stations(ctx, p, s, e)


@tool
def find_transit_triggers(state: KPState, start: Optional[str] = None, end: Optional[str] = None, planets: Optional[list[str]] = None,
                          levels: Optional[list[str]] = None, target_lords: Optional[list[str]] = None, dasha_depth: int = 4,
                          include_ruling_planets_at_start: bool = False, limit: int = 400) -> dict[str, Any]:
    """KP transit timing: stretches when a transiting planet's sign / star / sub lord is a target lord. Targets default to the running
    Dasha / Bhukti / Antara / Sookshma lords at `start` (`dasha_depth`); give `target_lords` to use your own, and
    `include_ruling_planets_at_start` to add the ruling planets at `start`. Defaults: window today .. +60 days, planets ['Sun'],
    levels ['star']. KP rule of thumb: Jupiter / Saturn / nodes pick months, the Sun the week, the Moon the day."""
    with core.calc(state) as ctx:
        chart, bdt = _natal(ctx, state)
        s, e = _window(state, start, end, 60 / 365.25)
        pl = [planet_name(x) for x in (planets or ["Sun"])]
        lv = levels or ["star"]
        targets: dict[str, list[str]] = {}
        if target_lords:
            for x in target_lords:
                targets.setdefault(planet_name(x), []).append("given")
        else:
            d = da.make_dasha(chart, bdt)
            for r in da.running_ladder(d, chart, s, dasha_depth):
                targets.setdefault(r["lords"][-1], []).append(r["label"])
        if include_ruling_planets_at_start:
            rp = mo.ruling_planets(ctx, s, state.latitude, state.longitude, True, None)
            for item in rp["ruling_planets"]:
                targets.setdefault(item["planet"], []).append("ruling planet")
        out = mo.find_transit_triggers(ctx, pl, targets, s, e, lv, limit)
        out["birth"] = state.summary()
        return out


@tool
def find_double_transit(state: KPState, event: Optional[str] = None, houses: Optional[list[int]] = None, primary_only: bool = False,
                       start: Optional[str] = None, end: Optional[str] = None, sign_scope: str = "cusp", include_lord: bool = False,
                       limit: int = 200) -> dict[str, Any]:
    """Double transit (K.N. Rao / Parashari technique used beside KP, NOT Krishnamurti's own rule; a supporting condition, not a
    trigger). Windows in which BOTH transiting Jupiter and Saturn occupy or aspect a target sign of an event house. Give `event`
    (its favourable houses, or only the primary cusp with `primary_only`) or your own `houses`. `sign_scope`: 'cusp' (sign of the
    cusp, default) or 'house' (every sign the unequal house spans); `include_lord` adds the natal sign of the house lord. Aspects
    are sign-based (7th for both; Jupiter also 5th/9th, Saturn 3rd/10th). Retrograde-aware: stretches follow the real longitude, so
    a station that slips a planet back into a sign shows as a new window; nearby Jupiter / Saturn stations are listed per window.
    Default window: from `start` (today) for 3 years; max 30 years. Facts only, no prediction."""
    hs, source = dbl.resolve_houses(event, houses, primary_only)
    s, e = _window(state, start, end, 3.0)
    with core.calc(state) as ctx:
        chart, _ = _natal(ctx, state)
        out = dbl.find_double_transit(ctx, chart, hs, source, s, e, sign_scope.strip().lower(), include_lord, limit)
        out["birth"] = state.summary()
        return out


@tool
def verify_event_day(state: KPState, when: Optional[str] = None, event: Optional[str] = None, custom_favourable: Optional[list[int]] = None,
                     custom_denial: Optional[list[int]] = None, primary_cusp: Optional[int] = None, depth: int = 4,
                     planets: Optional[list[str]] = None, extended_rps: bool = True, levels: Optional[str] = None) -> dict[str, Any]:
    """KP transit rule set for one day / moment. Shows the running dasha chain (to `depth`: 1 Dasha .. 5 Prana), the transiting
    Jupiter, Saturn, Sun and Moon (or your `planets`) with their sign / star / sub lords, and marks each 'match' / 'sub lord only' /
    'no match' against the dasha-bhukti-antara lords and against the ruling planets of that moment. Also says whether the Sun and
    Moon both touch the dasha lords. With `event` (or custom houses) each dasha lord is checked for signifying the event houses.
    `when`: YYYY-MM-DD (taken as 12:00 local) or YYYY-MM-DDTHH:MM:SS; default now. Facts only, no prediction."""
    s = core.parse_dt(when, now_at(state))
    if when and len(when.strip()) <= 10:
        s = s.replace(hour=12, minute=0, second=0, microsecond=0)
    with core.calc(state) as ctx:
        chart, bdt = _natal(ctx, state)
        fav = den = ev = None
        if event or custom_favourable:
            eh = _event_houses(chart, event, custom_favourable, custom_denial, primary_cusp)
            fav, den, ev = eh["favourable"], eh["denial"], {"label": eh["label"], "event": eh["event"], "favourable": eh["favourable"],
                                                           "denial": eh["denial"], "primary_cusp": eh["primary_cusp"]}
        pl = [planet_name(x) for x in planets] if planets else None
        out = tr.verify_event_day(ctx, chart, bdt, s, state.latitude, state.longitude, depth, pl, extended_rps, fav, den, ev, levels)
        out["birth"] = state.summary()
        return out


@tool
def get_longevity_facts(state: KPState, start: Optional[str] = None, end: Optional[str] = None, depth: int = 3, include_periods: bool = True,
                        extra_houses: Optional[list[int]] = None, limit: int = 100, levels: Optional[str] = None) -> dict[str, Any]:
    """Classical longevity / maraka / badhaka listing, facts only (never a lifespan estimate or a prediction of death). Gives the
    cuspal sub lords of the 1st and 8th cusps with the supporting (1, 5, 9, 10) and difficult (6, 8, 12) houses they signify, the
    classical 'longevity' group hits, the badhaka house (movable 11, fixed 9, dual 7) and maraka houses (2, 7) with their lords, and
    (`include_periods`) joint periods, depth 2-5 (default 3 = Dasha / Bhukti / Antara), in which every lord signifies a maraka or
    badhaka house (plus any `extra_houses`), flagged 'critical period'. Window default today .. +5 years. The caller decides how
    to present it."""
    with core.calc(state) as ctx:
        chart, bdt = _natal(ctx, state)
        s, e = _window(state, start, end, 5)
        d = da.make_dasha(chart, bdt)
        out = lg.longevity_facts(chart, d, s, e, depth, include_periods, extra_houses, limit, levels)
        out["birth"] = state.summary()
        return out


@tool
def get_health_facts(state: KPState, start: Optional[str] = None, end: Optional[str] = None, include_periods: bool = True, limit: int = 50,
                     levels: Optional[str] = None) -> dict[str, Any]:
    """KP health listing, facts only: NOT medical advice, a diagnosis or a prognosis. The 6th cusp sub lord with the disease houses
    (6, 8, 12), the 1st (recurrence) and the recovery houses (5, 11) it signifies, nature hints (Saturn chronic, Mercury complications,
    Mars sudden), the traditional sign-to-body-part table for the 6th cusp and Ascendant signs, Mars significations, the 8th and 12th
    cuspal sub lords, and (`include_periods`) joint periods in `start`..`end` (default today +5 years) whose lords signify the disease
    houses or the recovery houses. No planet-to-body-part table (no vetted source)."""
    if limit < 1:
        raise core.KPError("limit must be at least 1.")
    with core.calc(state) as ctx:
        chart, bdt = _natal(ctx, state)
        s, e = _window(state, start, end, 5)
        d = da.make_dasha(chart, bdt)
        out = med.health_facts(chart, d, s, e, include_periods, limit, levels)
        out["birth"] = state.summary()
        return out


@tool
def compare_births(a: KPState, b: KPState) -> dict[str, Any]:
    """KP twin / close-birth comparison. Lists, for two birth records (same options are applied to both: those of `a`), which cusps
    and planets differ at the sign / star / sub / sub-sub level, how far the Lagna and each cusp moved, the dasha balance of each,
    and the events whose cuspal-sub-lord promise verdict differs. Cusp sub lords can change within a few minutes, which is why
    twins can differ. Facts only; no claim about the people."""
    b2 = b.model_copy(update={"options": a.options})
    ca = _natal_chart(a)
    cb = _natal_chart(b2)
    same = (a.latitude, a.longitude) == (b.latitude, b.longitude)
    out = twins.compare_charts(ca, cb, a.local_datetime, b2.local_datetime, same)
    out["a"], out["b"] = a.summary(), b2.summary()
    return out


@tool
def find_muhurta_windows(moment: MomentState, event: Optional[str] = None, end: Optional[str] = None, custom_favourable: Optional[list[int]] = None,
                         custom_denial: Optional[list[int]] = None, primary_cusp: Optional[int] = None, step_minutes: float = 10.0,
                         csl_min: str = "weak", require_lagna_sub: bool = True, min_rp_support: int = 2, limit: int = 50,
                         levels: Optional[str] = None) -> dict[str, Any]:
    """KP electional helper: scans from `moment` (date / time / place; the start) to `end` (default +24 h) and lists windows in which the
    chart of the moment supports a matter: the Lagna sub lord signifies favourable houses and no denial house (`require_lagna_sub`),
    the matter's cuspal sub lord has a promise verdict of at least `csl_min` (none | weak | strong) and at least `min_rp_support`
    distinct ruling planets signify a favourable house without a denial house. Matter = `event` or `custom_favourable` houses
    (+ `custom_denial`, `primary_cusp`). Sampled every `step_minutes` (edges accurate to the step; confirm with get_lagna_changes).
    Facts only: no panchang / tithi rules, not an auspiciousness verdict."""
    with core.calc(moment) as ctx:
        s = moment.local_datetime
        e = core.parse_dt(end, s + timedelta(days=1))
        try:
            out = mh.find_windows(ctx, s, e, moment.latitude, moment.longitude, event, custom_favourable, custom_denial, primary_cusp,
                                  step_minutes, csl_min, require_lagna_sub, min_rp_support, limit, levels)
        except ValueError as ex:
            raise core.KPError(str(ex))
        out["place"] = moment.summary()
        return out


@tool
def get_timezone_info(latitude: float, longitude: float, date: Optional[str] = None, time: Optional[str] = None) -> dict[str, Any]:
    """Offline: IANA timezone name for coordinates and, with `date` (YYYY-MM-DD) and optional `time` (HH:MM[:SS], default 12:00), the
    UTC offset then, whether daylight saving applied, and flags for an ambiguous or non-existent local clock time. Use it to check a
    birth record's offset (historical rules from the system tz database)."""
    when = None
    if date:
        when = core.parse_dt(f"{date}T{time}" if time else date, None)
        if not time:
            when = when.replace(hour=12, minute=0, second=0, microsecond=0)
    return pl.timezone_info(latitude, longitude, when)


@tool
def lookup_place(query: str, limit: int = 3) -> dict[str, Any]:
    """Latitude, longitude and IANA timezone for a place name (OpenStreetMap Nominatim; needs internet, otherwise a clean tool
    error). Use it instead of guessing coordinates; check that the first match is the intended place."""
    return pl.lookup_place(query, limit)


@tool
def get_kp_annual_chart(state: KPState, year: int, place: Optional[MomentState] = None, detail: str = "standard") -> dict[str, Any]:
    """KP annual chart (solar return): the moment in `year` when the sidereal Sun returns to its natal longitude, cast for `place`
    (latitude / longitude / timezone; date and time of that state are ignored; default = the birth place, KP practice uses the place
    of residence that year). Returns the return moment (local and UTC), the chart (detail as in get_kp_chart), the promise table of
    the return chart, a cusp-by-cusp and planet comparison with the natal chart, and the natal Dasha ladder at that moment.
    Tajika yogas, Muntha and Mudda dasha are not KP and are not computed. Facts only."""
    if detail not in ("basic", "standard", "full"):
        raise ValueError("detail must be 'basic', 'standard' or 'full'.")
    with core.calc(state) as ctx:
        natal, bdt = _natal(ctx, state)
        sun = natal.planets["Sun"]["longitude"]
        jd = an.solve_return(ctx, sun, year, bdt)
        at_birth_place = ctx.local_from_jd(jd)
        d = da.make_dasha(natal, bdt)
        ladder = da.running_ladder(d, natal, at_birth_place, 4)
    pst = state if place is None else place.model_copy(update={"options": state.options})
    with core.calc(pst) as ctx2:
        local = ctx2.local_from_jd(jd)
        ret = ch.chart_at(ctx2, jd, pst.latitude, pst.longitude, local)
        meta = {"place": pst.place_name, "latitude": pst.latitude, "longitude": pst.longitude, "timezone": pst.timezone,
                "datetime_local": core.fmt_dt(local)}
        out = {"birth": state.summary(), "year": year, "return_moment": {"local": core.fmt_dt(local), "place": pst.place_name,
                                                                         "julian_day_ut": round(jd, 6), "natal_sun": kp_lords(sun)["dms"]},
               "chart": ch.chart_dict(ret, meta, detail), "promise_table": ch.promise_table(ret), "comparison": an.compare(natal, ret),
               "natal_dasha_at_return": ladder, "natal_dasha_checked_at_local_birth_place_time": core.fmt_dt(at_birth_place),
               "school": "KP flavour of the annual chart: return chart read with KP cusp sub lords beside the natal chart; not a Tajika reading.",
               "note": "Place matters: the cusps change with the place, the planets do not. Facts only; no prediction."}
    return out


@tool
def get_lagna_changes(moment: MomentState, minutes: float = 30.0, level: str = "sub") -> dict[str, Any]:
    """Stretches of time around a moment (+- `minutes`) during which the Lagna keeps the same sign / star / sub (or sub-sub) lord,
    with exact change times. Shows how sensitive a birth time is: use the birth date/time and place as the moment."""
    with core.calc(moment) as ctx:
        return mo.lagna_changes(ctx, moment.local_datetime, moment.latitude, moment.longitude, minutes, level)


@tool
def rectify_birth_time(state: KPState, events: list[dict[str, Any]], minutes: float = 20.0, step_seconds: int = 60,
                       top: int = 8, moon_link: str = "report") -> dict[str, Any]:
    """Event-based birth-time scan. `events` = [{'date': 'YYYY-MM-DD', 'event': 'marriage'}, ...] (or 'custom_favourable',
    'custom_denial', 'primary_cusp'). Trial birth times within +- `minutes` of the given time are scored: cuspal sub lord promise
    of the matter plus Dasha / Bhukti / Antara lords signifying its houses on the event date. Returns ranges of equal outcome,
    best first. Needs certain event dates; more events separate the candidates better. Each event may carry a `weight` (0-10,
    default 1; anchors such as accident, marriage, child birth deserve more). Every range shows the Lagna-Moon lord link
    (Lagna star / sub lord connected with the Moon star / sub lord); `moon_link='require'` ranks only linked ranges. Each best range
    reports the Lagna sub-lord sensitivity (minutes back / forward until the Lagna sub lord changes)."""
    with core.calc(state) as ctx:
        out = mo.rectify(ctx, state.local_datetime, state.latitude, state.longitude, events, minutes, step_seconds, None, top, moon_link)
        out["birth"] = state.summary()
        return out


@tool
def get_kp_report(state: KPState, on_date: Optional[str] = None) -> dict[str, Any]:
    """One-page condensed KP report as markdown text (`markdown`) that can be saved or printed: planets and cusps with sign / star /
    sub / sub-sub lords, houses signified (A-D), the cuspal links table (each CSL and the houses it signifies), significators by house,
    balance at birth, the running dasha ladder and the ruling planets on `on_date` (default now) at the birth place, Fortuna.
    Composition of existing facts; no verdicts or predictions."""
    with core.calc(state) as ctx:
        chart, bdt = _natal(ctx, state)
        d = da.make_dasha(chart, bdt)
        at = core.parse_dt(on_date, now_at(state))
        rp = mo.ruling_planets(ctx, at, state.latitude, state.longitude, True, chart)
        meta = state.summary()
        text = rep.build(chart, {"place": state.place_name, "datetime_local": f"{meta['date_of_birth']} {meta['time_of_birth']}"}, d.balance_at_birth(),
                         da.running_ladder(d, chart, at, 4), rp, core.fmt_dt(at))
        return {"birth": meta, "on_date": core.fmt_dt(at), "format": "markdown", "markdown": text}


@tool
def kp_judgment_workflow(state: KPState, event: Optional[str] = None, custom_favourable: Optional[list[int]] = None,
                         custom_denial: Optional[list[int]] = None, primary_cusp: Optional[int] = None, start: Optional[str] = None,
                         end: Optional[str] = None, when: Optional[str] = None, depth: int = 3, limit: int = 10,
                         levels: Optional[str] = None) -> dict[str, Any]:
    """Guided KP judgment in the teaching order, one auditable result: 1 matter and cusp, 2 cuspal sub lord, 3 promise, 4 timing (joint
    Dasha / Bhukti / Antara periods in `start`..`end`, default today +5 years, first `limit` shown), 5 transit check on `when` (a day or
    moment; skipped without it), 6 ruling planets (at `when`, else at `start`) verification. Each step states its rule, the houses used
    and a status pass / partial / fail / n.a. Composition of the existing tools; no overall verdict and no prediction."""
    s, e = _window(state, start, end, 5)
    w = None
    if when:
        w = core.parse_dt(when, now_at(state))
        if len(when.strip()) <= 10:
            w = w.replace(hour=12, minute=0, second=0, microsecond=0)
    if limit < 1:
        raise core.KPError("limit must be at least 1.")
    with core.calc(state) as ctx:
        chart, bdt = _natal(ctx, state)
        try:
            out = wf.run(ctx, chart, bdt, state.latitude, state.longitude, s, e, w, event, custom_favourable, custom_denial, primary_cusp,
                         depth, limit, levels)
        except (ValueError, da.DashaError) as ex:
            raise core.KPError(str(ex))
        out["birth"] = state.summary()
        return out


@tool
def get_kp_summary(state: KPState, on_date: Optional[str] = None) -> dict[str, Any]:
    """One-call overview: ascendant and Moon with all lords, cuspal sub lords, running Dasha chain, ruling planets now at the birth
    place, and the promise verdict for every known matter."""
    with core.calc(state) as ctx:
        chart, bdt = _natal(ctx, state)
        d = da.make_dasha(chart, bdt)
        at = core.parse_dt(on_date, now_at(state))
        rp = mo.ruling_planets(ctx, at, state.latitude, state.longitude, True, chart)
        moon = chart.planets["Moon"]
        return {"birth": state.summary(), "ayanamsa": chart.opts.ayanamsa, "node_type": chart.opts.node_type,
                "house_system_used": chart.houses["system"],
                "ascendant": kp_lords(chart.asc),
                "moon": {k: moon[k] for k in ("dms", "sign", "nakshatra", "star_lord", "sub_lord", "house")},
                "cuspal_sub_lords": [{"house": r["house"], "sign": r["sign"], "star_lord": r["star_lord"], "sub_lord": r["sub_lord"]}
                                     for r in ch.cusp_rows(chart)],
                "balance_at_birth": d.balance_at_birth(), "on_date": core.fmt_dt(at),
                "running_dasha": da.running_ladder(d, chart, at, 4),
                "ruling_planets_at_date": [{"planet": p["planet"], "count": p["count"], "roles": p["roles"],
                                            "natal_signifies": p["natal_signifies"]} for p in rp["ruling_planets"]],
                "promise_table": ch.promise_table(chart),
                "warnings": [chart.houses["warning"]] if chart.houses.get("warning") else [],
                "note": "Facts and classical KP rules only. Use judge_event / find_event_periods for reasoning on one matter."}


if __name__ == "__main__":
    mcp.run()
