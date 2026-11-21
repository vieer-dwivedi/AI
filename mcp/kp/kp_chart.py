"""KP chart assembly, significators and event judgment (rule-based; never predicts outcomes).

Conventions (see progress.md, research notes R1-R4, R10-R11):
  * House owner = sign lord of the sign on the cusp; Rahu/Ketu own no sign.
  * Planet P signifies houses on four levels (strongest first):
        A  the house occupied by P's star lord
        B  the house P occupies
        C  the houses owned by P's star lord
        D  the houses owned by P
    House-wise significators are derived from the same table, so both views always agree.
  * Rahu/Ketu are agents: they also carry the houses of the planets they conjoin, aspect, and of their star and sign lords.
  * Cuspal sub lord (CSL) test: the CSL of the matter's cusp must signify the favourable houses and not the denial houses.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import kp_csl_ext
import kp_cusp
from kp_core import Ctx, fmt_dt
from kp_retro import cusp_check
from kp_tables import (EVENTS, HOUSE_THEMES, PLANETS, SIGN_LORDS, SIGNS, badhaka_house, dms, event_info,
                       kp_lords, sign_index_of)
from state import KPState

GRAHAS = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]
NODES = ["Rahu", "Ketu"]
# Sign aspects counted as houses from the planet's sign (7th for all; extras for Mars, Jupiter, Saturn).
ASPECTS = {"Mars": (4, 7, 8), "Jupiter": (5, 7, 9), "Saturn": (3, 7, 10)}
LEVEL_NAMES = {"A": "in the star of an occupant", "B": "occupant / the planet's own house",
               "C": "in the star of the owner", "D": "owner"}


def house_of_lon(lon: float, cusps: list[float]) -> int:
    for i in range(12):
        start, end = cusps[i], cusps[(i + 1) % 12]
        width = (end - start) % 360.0
        if width == 0.0:
            width = 360.0
        if (lon - start) % 360.0 < width:
            return i + 1
    return 12


def _pl(lon: float, house: Optional[int] = None) -> dict[str, Any]:
    r = kp_lords(lon)
    if house is not None:
        r["house"] = house
    return r


# --------------------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------------------
def chart_at(ctx: Ctx, jd_planets: float, lat: float, lon: float, dt_local, jd_houses: Optional[float] = None,
             houses: Optional[dict[str, Any]] = None) -> SimpleNamespace:
    """Build a chart: planets at `jd_planets`, cusps at `jd_houses` (horary uses a different time for the cusps)."""
    h = houses or ctx.cusps(jd_houses if jd_houses is not None else jd_planets, lat, lon)
    cusps: list[float] = h["cusps"]
    pos = ctx.positions(jd_planets)
    planets: dict[str, dict[str, Any]] = {}
    for name, p in pos.items():
        d = _pl(p["lon"], house_of_lon(p["lon"], cusps))
        d.update(planet=name, speed=round(p["speed"], 5), retrograde=p["retro"])
        planets[name] = d
    owner = {i + 1: SIGN_LORDS[sign_index_of(cusps[i])] for i in range(12)}
    owned: dict[str, list[int]] = {p: [] for p in PLANETS}
    for hh, lord in owner.items():
        owned[lord].append(hh)
    star_of = {p: planets[p]["star_lord"] for p in PLANETS}
    house_of = {p: planets[p]["house"] for p in PLANETS}
    # cusp proximity (item 5): facts whenever `cusp_orb_deg` is set; shifted into A/B only with `cusp_proximity_apply`
    orb = ctx.opts.cusp_orb_deg
    near = kp_cusp.near_cusp_entries(planets, cusps, orb)
    apply_shift = bool(orb and ctx.opts.cusp_proximity_apply)
    extra = kp_cusp.shift_map(near) if apply_shift else {p: [] for p in PLANETS}
    sig: dict[str, dict[str, list[int]]] = {}
    for p in PLANETS:
        sl = star_of[p]
        sig[p] = {"A": [house_of[sl]] + extra[sl], "B": [house_of[p]] + extra[p], "C": sorted(owned[sl]), "D": sorted(owned[p])}
    chart = SimpleNamespace(near_cusp=near, near_cusp_orb=orb, near_cusp_applied=apply_shift,
                            jd=jd_planets, jd_houses=jd_houses if jd_houses is not None else jd_planets, dt=dt_local, lat=lat, lon=lon, ctx=ctx, houses=h, cusps=cusps, planets=planets,
                            owner=owner, owned=owned, star_of=star_of, house_of=house_of, sig=sig,
                            asc=h["asc"], opts=ctx.opts)
    chart.agents = _node_agents(chart)
    chart.agent_houses = {n: sorted({x for a in chart.agents[n] for x in _union(chart, a["planet"], "ABCD", agents=False)})
                          for n in NODES}
    return chart


def _node_agents(chart: SimpleNamespace) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for n in NODES:
        lst: list[dict[str, Any]] = []
        ns = int(chart.planets[n]["sign_index"])
        seen: set[str] = set()

        def add(p: str, rel: str) -> None:
            if p not in seen and p != n:
                seen.add(p)
                lst.append({"planet": p, "relation": rel})

        for g in GRAHAS:
            if int(chart.planets[g]["sign_index"]) == ns:
                add(g, "conjunction (same sign)")
        for g in GRAHAS:
            gs = int(chart.planets[g]["sign_index"])
            for off in ASPECTS.get(g, (7,)):
                if (gs + off - 1) % 12 == ns:
                    add(g, f"aspect ({off}th)")
        add(chart.star_of[n], "star lord")
        add(SIGN_LORDS[ns], "sign lord")
        out[n] = lst
    return out


def _union(chart: SimpleNamespace, planet: str, levels: str, agents: bool = True) -> list[int]:
    s: set[int] = set()
    for L in levels:
        s.update(chart.sig[planet][L])
    if agents and chart.opts.nodes_as_agents and planet in NODES:
        s.update(chart.agent_houses.get(planet, []))
    return sorted(s)


def signified(chart: SimpleNamespace, planet: str, levels: Optional[str] = None) -> list[int]:
    """Houses a planet signifies under the chart's level setting (nodes include their agents' houses)."""
    return _union(chart, planet, levels or chart.opts.significator_levels)


# --------------------------------------------------------------------------------------
# Output builders
# --------------------------------------------------------------------------------------
def planet_rows(chart: SimpleNamespace) -> list[dict[str, Any]]:
    rows = []
    for name, d in chart.planets.items():
        row = {"planet": name, "longitude": d["longitude"], "dms": d["dms"], "sign": d["sign"],
               "degree_in_sign": d["degree_in_sign"], "nakshatra": d["nakshatra"], "pada": d["pada"],
               "sign_lord": d["sign_lord"], "star_lord": d["star_lord"], "sub_lord": d["sub_lord"],
               "sub_sub_lord": d["sub_sub_lord"], "kp_number": d["kp_number"], "house": d["house"],
               "retrograde": d["retrograde"], "speed_deg_per_day": d["speed"]}
        if name in PLANETS:
            row["owns_houses"] = chart.owned[name]
        rows.append(row)
    return rows


OUTER_NAMES = ("Uranus", "Neptune", "Pluto")


def outer_occupants(chart: SimpleNamespace, house: int) -> list[str]:
    """Uranus / Neptune / Pluto standing in `house` (empty unless `include_outer`). Occupants only: they own no sign or star, so they
    never enter the A-D tables."""
    return [p for p in OUTER_NAMES if p in chart.planets and chart.planets[p]["house"] == house]


def outer_block(chart: SimpleNamespace) -> Optional[dict[str, Any]]:
    present = [p for p in OUTER_NAMES if p in chart.planets]
    if not present:
        return None
    return {"planets": [{"planet": p, "house": chart.planets[p]["house"], "sign": chart.planets[p]["sign"], "dms": chart.planets[p]["dms"],
                         "star_lord": chart.planets[p]["star_lord"], "sub_lord": chart.planets[p]["sub_lord"],
                         "retrograde": chart.planets[p]["retrograde"]} for p in present],
            "rule": ("Occupants only: Uranus, Neptune and Pluto own no sign or star in KP, so they are listed beside the house but never "
                     "enter the A-D significator tables, the judgments or the dasha. Shown because `include_outer` is on.")}


def node_check(chart: SimpleNamespace) -> dict[str, Any]:
    """Mean vs true node at the chart time (fact only). Flags when Rahu / Ketu change sign, star or sub lord between the two."""
    ctx, jd = chart.ctx, chart.jd
    s = ctx.s
    other_id = s.TRUE_NODE if chart.opts.node_type == "mean" else s.MEAN_NODE
    xx, _ = s.calc_ut(jd, other_id, ctx.flags)
    rows: dict[str, Any] = {}
    for n in NODES:
        lon = xx[0] % 360.0 if n == "Rahu" else (xx[0] + 180.0) % 360.0
        o, u = kp_lords(lon), chart.planets[n]
        diff = [lv.replace("_lord", "") for lv in ("sign_lord", "star_lord", "sub_lord") if o[lv] != u[lv]]
        rows[n] = {"used": {"dms": u["dms"], "sign_lord": u["sign_lord"], "star_lord": u["star_lord"], "sub_lord": u["sub_lord"]},
                   "other": {"dms": o["dms"], "sign_lord": o["sign_lord"], "star_lord": o["star_lord"], "sub_lord": o["sub_lord"]},
                   "differs_at": diff, "sub_lord_differs": "sub" in diff,
                   "difference_deg": round(((lon - u["longitude"] + 180.0) % 360.0) - 180.0, 6)}
    used, other = chart.opts.node_type, ("true" if chart.opts.node_type == "mean" else "mean")
    return {"used": used, "compared_with": other, "nodes": rows,
            "any_sub_lord_differs": any(r["sub_lord_differs"] for r in rows.values()),
            "note": ("The mean and the true node can lie up to about 1.5 degrees apart, so the star or sub lord of Rahu / Ketu (and their "
                     "agent houses) can change with the node type. Fact only; the chart uses the type named in `used`.")}


def cusp_rows(chart: SimpleNamespace) -> list[dict[str, Any]]:
    rows = []
    for i, lon in enumerate(chart.cusps, start=1):
        r = kp_lords(lon)
        occ = [p for p in PLANETS if chart.house_of[p] == i]
        row = {"house": i, "longitude": r["longitude"], "dms": r["dms"], "sign": r["sign"], "degree_in_sign": r["degree_in_sign"],
               "sign_lord": r["sign_lord"], "star_lord": r["star_lord"], "sub_lord": r["sub_lord"],
               "sub_sub_lord": r["sub_sub_lord"], "kp_number": r["kp_number"], "occupants": occ,
               "theme": HOUSE_THEMES[i]}
        if "Uranus" in chart.planets:
            row["outer_occupants"] = outer_occupants(chart, i)
        if chart.near_cusp_orb:
            row["planets_behind_cusp"] = [e["planet"] for e in chart.near_cusp if e["relation"] == "behind_cusp" and e["cusp"] == i]
            row["planets_past_cusp"] = [e["planet"] for e in chart.near_cusp if e["relation"] == "past_cusp" and e["cusp"] == i]
        rows.append(row)
    return rows


def near_cusp_block(chart: SimpleNamespace) -> Optional[dict[str, Any]]:
    """The `near_cusp` output block (None when `cusp_orb_deg` is off)."""
    return kp_cusp.block(chart.near_cusp, chart.near_cusp_orb, chart.near_cusp_applied)


def planet_significations(chart: SimpleNamespace) -> list[dict[str, Any]]:
    out = []
    for p in PLANETS:
        d = chart.planets[p]
        in_star = [q for q in PLANETS if q != p and chart.star_of[q] == p]
        in_sub = [q for q in PLANETS if q != p and chart.planets[q]["sub_lord"] == p]
        row = {"planet": p, "house": chart.house_of[p], "owns_houses": chart.owned[p], "star_lord": d["star_lord"],
               "sub_lord": d["sub_lord"],
               "levels": {L: chart.sig[p][L] for L in "ABCD"},
               "signifies": signified(chart, p, "ABCD"),
               "planets_in_its_star": in_star, "planets_in_its_sub": in_sub,
               "no_planet_in_its_star": len(in_star) == 0, "in_own_star": d["star_lord"] == p,
               "retrograde": d["retrograde"]}
        if chart.near_cusp_orb:
            mine = [e for e in chart.near_cusp if e["planet"] == p]
            row["near_cusp"] = mine
            row["signifies_with_near_cusp"] = sorted(set(row["signifies"]) | {e["cusp"] for e in mine if e["relation"] == "behind_cusp"})
        if p in NODES:
            row["agents"] = chart.agents[p]
            row["agent_houses"] = chart.agent_houses[p]
            row["note"] = "Rahu/Ketu carry the results of the planets they represent; a node is stronger than its agent."
        out.append(row)
    return out


def house_significators(chart: SimpleNamespace) -> list[dict[str, Any]]:
    rows = []
    for h in range(1, 13):
        lv: dict[str, list[str]] = {L: [p for p in PLANETS if h in chart.sig[p][L]] for L in "ABCD"}
        # nodes also signify through their agents (listed separately so the A-D table stays pure)
        via_agents = [n for n in NODES if chart.opts.nodes_as_agents and h in chart.agent_houses[n]
                      and all(n not in lv[L] for L in "ABCD")]
        every = []
        for L in "ABCD":
            for p in lv[L]:
                if p not in every:
                    every.append(p)
        row = {"house": h, "cusp_sign": SIGNS[sign_index_of(chart.cusps[h - 1])], "owner": chart.owner[h],
               "occupants": [p for p in PLANETS if chart.house_of[p] == h],
               "A_in_star_of_occupants": lv["A"], "B_occupants": lv["B"],  # lv["B"] also lists planets shifted in by cusp_proximity_apply
               "C_in_star_of_owner": lv["C"], "D_owner": [chart.owner[h]],
               "all_in_strength_order": every, "nodes_via_agents": via_agents, "theme": HOUSE_THEMES[h]}
        if "Uranus" in chart.planets:
            row["outer_occupants"] = outer_occupants(chart, h)
        if chart.near_cusp_orb:
            row["near_cusp_planets"] = [e["planet"] for e in chart.near_cusp if e["relation"] == "behind_cusp" and e["cusp"] == h]
        rows.append(row)
    return rows


def fortuna(chart: SimpleNamespace) -> dict[str, Any]:
    lon = (chart.asc + chart.planets["Moon"]["longitude"] - chart.planets["Sun"]["longitude"]) % 360.0
    d = _pl(lon, house_of_lon(lon, chart.cusps))
    d["formula"] = "Ascendant + Moon - Sun (KP Reader 1)"
    return d


def chart_dict(chart: SimpleNamespace, meta: dict[str, Any], detail: str = "standard") -> dict[str, Any]:
    ay = chart.ctx.ayanamsa(chart.jd)
    out = {
        **meta,
        "ayanamsa": {"name": chart.opts.ayanamsa, "value_deg": round(ay, 6), "value_dms": dms(ay)},
        "node_type": chart.opts.node_type,
        "house_system_used": chart.houses["system"],
        "ascendant": {**kp_lords(chart.asc)},
        "cusps": cusp_rows(chart),
        "planets": planet_rows(chart),
        "fortuna": fortuna(chart),
    }
    if detail == "full":
        out["node_check"] = node_check(chart)
    ob = outer_block(chart)
    if ob:
        out["outer_planets"] = ob
    if chart.houses.get("warning"):
        out["warnings"] = [chart.houses["warning"]]
    if chart.near_cusp_orb:
        out["near_cusp"] = near_cusp_block(chart)
    if detail in ("standard", "full"):
        out["planet_significations"] = planet_significations(chart)
        out["house_significators"] = house_significators(chart)
    return out


# --------------------------------------------------------------------------------------
# Cuspal sub-lord and event judgment
# --------------------------------------------------------------------------------------
def csl_report(chart: SimpleNamespace, cusp: int, levels: Optional[str] = None) -> dict[str, Any]:
    if not 1 <= cusp <= 12:
        raise ValueError("cusp must be 1..12")
    lv = levels or chart.opts.significator_levels
    r = kp_lords(chart.cusps[cusp - 1])
    sub = r["sub_lord"]
    star = r["star_lord"]
    by_level = {L: chart.sig[sub][L] for L in "ABCD"}
    rep = {
        "cusp": cusp, "cusp_dms": r["dms"], "sign": r["sign"], "sign_lord": r["sign_lord"], "star_lord": star, "sub_lord": sub,
        "sub_sub_lord": r["sub_sub_lord"],
        "sub_lord_in_house": chart.house_of[sub], "sub_lord_owns": chart.owned[sub],
        "sub_lord_star_lord": chart.star_of[sub], "sub_lord_star_lord_in_house": chart.house_of[chart.star_of[sub]],
        "sub_lord_levels": by_level, "sub_lord_signifies": signified(chart, sub, lv),
        "star_lord_signifies": signified(chart, star, lv),
        "levels_used": lv,
        "sub_lord_retrograde": chart.planets[sub]["retrograde"],
        "retrograde_check": cusp_check(chart, sub, chart.opts.retrograde_rule),
    }
    ext = kp_csl_ext.extension(chart, cusp, sub, rep["sub_lord_signifies"], chart.opts.csl_extension)
    if ext:
        rep["csl_extension"] = ext  # separate block: the A-D houses above are never widened
    return rep


def _resolve_denial(chart: SimpleNamespace, denial: list[Any]) -> list[int]:
    out: list[int] = []
    for d in denial:
        if d == "badhaka":
            out.append(badhaka_house(sign_index_of(chart.asc)))
        else:
            out.append(int(d))
    return sorted(set(out))


def _verdict(fav_hit: list[int], den_hit: list[int], fav: list[int]) -> tuple[str, str]:
    need = min(2, len(fav))
    if len(fav_hit) >= need and need > 0:
        if den_hit:
            return "promised_with_obstacles", "CSL signifies the favourable houses but also denial houses."
        return "strongly_promised", "CSL signifies the favourable houses and no denial house."
    if len(fav_hit) == 1:
        if den_hit:
            return "doubtful", "CSL touches one favourable house and also denial houses."
        return "weakly_promised", "CSL touches only one favourable house."
    return "not_promised", "CSL signifies none of the favourable houses."


def judge_event(chart: SimpleNamespace, event: Optional[str] = None, custom_favourable: Optional[list[int]] = None,
                custom_denial: Optional[list[int]] = None, primary_cusp: Optional[int] = None,
                levels: Optional[str] = None) -> dict[str, Any]:
    if event:
        info = event_info(event)
    elif custom_favourable:
        info = {"key": "custom", "label": "Custom matter", "primary_cusp": primary_cusp or 1,
                "favourable": list(custom_favourable), "denial": list(custom_denial or []), "note": ""}
    else:
        raise ValueError("Give an `event` name or `custom_favourable` houses.")
    fav = sorted(set(custom_favourable)) if custom_favourable else sorted(info["favourable"])
    den = _resolve_denial(chart, custom_denial if custom_denial is not None else info["denial"])
    cusp = primary_cusp or info["primary_cusp"]
    lv = levels or chart.opts.significator_levels
    rep = csl_report(chart, cusp, lv)
    sig_set = set(rep["sub_lord_signifies"])
    fav_hit = sorted(sig_set & set(fav))
    den_hit = sorted(sig_set & set(den) - set(fav))  # a house that is both favourable and denial counts as favourable
    verdict, why = _verdict(fav_hit, den_hit, fav)
    star_sig = set(rep["star_lord_signifies"])
    planets = []
    for p in PLANETS:
        s = set(signified(chart, p, lv))
        fh = sorted(s & set(fav))
        if fh:
            strong = sorted((set(chart.sig[p]["A"]) | set(chart.sig[p]["B"])) & set(fav))
            planets.append({"planet": p, "favourable_houses": fh, "denial_houses": sorted(s & set(den) - set(fav)),
                            "via_A_or_B": strong, "all_houses": sorted(s)})
    planets.sort(key=lambda r: (-len(r["via_A_or_B"]), len(r["denial_houses"]), -len(r["favourable_houses"])))
    res = {
        "event": info["key"], "label": info["label"], "primary_cusp": cusp, "favourable_houses": fav, "denial_houses": den,
        "note": info.get("note", ""), "csl": rep, "favourable_signified_by_csl": fav_hit, "denial_signified_by_csl": den_hit,
        "star_lord_of_csl_favourable": sorted(star_sig & set(fav)), "star_lord_of_csl_denial": sorted(star_sig & set(den) - set(fav)),
        "verdict": verdict, "verdict_reason": why,
        "retrograde_note": ("; ".join(rep["retrograde_check"]["why"]) + ". " + rep["retrograde_check"]["effect"]) if rep["retrograde_check"]["triggered"] else None,
        "rule": ("KP rule: the cuspal sub lord must signify the favourable houses and not the denial houses; the dasha / bhukti / "
                 "antara lords must be significators of the same houses for the event to fall in their joint period."),
        "significator_planets": planets,
        "caution": "Rule-based reading of the chart as given; it is not a prediction. Birth-time errors change cusp sub lords.",
    }
    ext = rep.get("csl_extension")
    if ext:
        s2 = set(ext["signifies_extended"])
        fh2, dh2 = sorted(s2 & set(fav)), sorted(s2 & set(den) - set(fav))
        v2, why2 = _verdict(fh2, dh2, fav)
        res["extended_reading"] = {"mode": ext["mode"], "verdict": v2, "verdict_reason": why2, "favourable_signified_by_csl": fh2,
                                   "denial_signified_by_csl": dh2, "added_houses": ext["added_houses"], "changes_verdict": v2 != verdict,
                                   "school": ext["school"],
                                   "note": "Separate reading with the widened CSL houses; the standard verdict above is unchanged."}
    return res


def promise_table(chart: SimpleNamespace, levels: Optional[str] = None) -> list[dict[str, Any]]:
    out = []
    for key in EVENTS:
        r = judge_event(chart, event=key, levels=levels)
        out.append({"event": key, "label": r["label"], "primary_cusp": r["primary_cusp"], "csl": r["csl"]["sub_lord"],
                    "favourable_signified": r["favourable_signified_by_csl"], "denial_signified": r["denial_signified_by_csl"],
                    "verdict": r["verdict"]})
    return out


def lord_significations(chart: SimpleNamespace, lords: list[str], fav: list[int], den: list[int],
                        levels: Optional[str] = None) -> list[dict[str, Any]]:
    lv = levels or chart.opts.significator_levels
    out = []
    for p in lords:
        s = set(signified(chart, p, lv))
        out.append({"planet": p, "signifies": sorted(s), "favourable": sorted(s & set(fav)),
                    "denial": sorted(s & set(den) - set(fav)),
                    "sub_lord": chart.planets[p]["sub_lord"],
                    "sub_lord_signifies": signified(chart, chart.planets[p]["sub_lord"], lv)})
    return out


def meta_for(state: KPState, chart: SimpleNamespace) -> dict[str, Any]:
    return {"birth": state.summary(), "jd_ut": round(chart.jd, 6), "datetime_local": fmt_dt(chart.dt)}
