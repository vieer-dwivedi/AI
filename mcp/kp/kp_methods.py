"""Alternative significator methods: the 4-step theory and the 6-fold table (progress.md backlog item 1).

Both are optional views next to the default KP 4-level (A-D) table in `kp_chart`. They never change the A-D result.
Judgment tools (judge_event, find_event_periods ...) still use A-D; fruitful-significator filtering by primary significators is item 3.

4-step theory (Sunil Gondhalekar / Shri Bhatt school; NOT Krishnamurti's own method):
  step 1 the planet, step 2 its star lord, step 3 its sub lord, step 4 the star lord of that sub lord (the 'end result indicator').
  In every step a planet signifies the house it occupies and the houses it owns. Each house is PRIMARY (strong) or SECONDARY (weak):
    * Steps 1 and 3 (rule 1, applied to the planet / the sub lord):
        - planet in its own star: occupied house primary (even if others sit there);
        - no other planet in its star: occupied house primary;
        - some other planet in its star: occupied house secondary;
        - owned houses: primary only when no other planet sits in that house, else secondary.
    * Steps 2 and 4 (rule 2, applied to the star lord / the sub's star lord): the star lord's occupied house is primary even if
      others sit there; its owned houses are primary only when vacant.
    * Rahu/Ketu act through agents (conjunction, aspect, star lord, sign lord). A node that is primary makes its agents' houses
      follow rule 1 for the agent; a node that is only secondary makes all its agents' houses secondary.
    * A planet within 3d20' of a cusp is a primary significator of that cusp's house (`cusp_orb_deg`).
    * Uranus, Neptune, Pluto are ignored. Only primary significators count in this school ("secondary ones are like a mirage").
  NOT implemented yet: upgrading planets conjunct a strong planet (3d20'), type-E upgrades; they are reported, not applied.

6-fold table (KP EZine 128 / Dhanabalan Rangasamy; absent from the 1970-71 Readers): the 4-fold columns plus the planets in the
SUB of the occupant and the SUB of the owner ("planet A is in the star of B and the sub of C").
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from kp_chart import ASPECTS, NODES
from kp_tables import PLANETS

ORB_DEG = 200.0 / 60.0  # 3d20'
STEP_ROLES = {1: "planet", 2: "star_lord", 3: "sub_lord", 4: "sub_star_lord"}
METHODS = ("kp4level", "four_step", "six_fold")


def normalize_method(method: Optional[str]) -> str:
    m = (method or "kp4level").strip().lower().replace("-", "_").replace(" ", "_")
    m = {"4level": "kp4level", "kp": "kp4level", "4step": "four_step", "foursteps": "four_step", "6fold": "six_fold"}.get(m, m)
    if m not in METHODS:
        raise ValueError(f"method must be one of {list(METHODS)}")
    return m


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _tenants(chart: SimpleNamespace, house: int, exclude: Optional[str] = None) -> list[str]:
    return [q for q in PLANETS if q != exclude and chart.house_of[q] == house]


def _in_star_of(chart: SimpleNamespace, planet: str) -> list[str]:
    return [q for q in PLANETS if q != planet and chart.star_of[q] == planet]


def _cusp_distance(lon: float, cusp: float) -> float:
    d = abs(lon - cusp) % 360.0
    return min(d, 360.0 - d)


def _on_cusp_houses(chart: SimpleNamespace, planet: str, orb: float) -> list[int]:
    lon = chart.planets[planet]["longitude"]
    return [i + 1 for i, c in enumerate(chart.cusps) if _cusp_distance(lon, c) <= orb]


def _entry(house: int, kind: str, strength: str, reason: str) -> dict[str, Any]:
    return {"house": house, "kind": kind, "strength": strength, "reason": reason}


# --------------------------------------------------------------------------------------
# rule 1 / rule 2 (one planet, one step)
# --------------------------------------------------------------------------------------
def _rule1(chart: SimpleNamespace, p: str, orb: float) -> list[dict[str, Any]]:
    """Houses of planet `p` judged as the planet itself (steps 1 and 3), without node agents."""
    occ = chart.house_of[p]
    in_star = _in_star_of(chart, p)
    out: list[dict[str, Any]] = []
    own_star = chart.star_of[p] == p
    if own_star:
        out.append(_entry(occ, "occupied", "primary", "planet is in its own star"))
    elif not in_star:
        out.append(_entry(occ, "occupied", "primary", "no planet in its star"))
    else:
        out.append(_entry(occ, "occupied", "secondary", "planets in its star: " + ", ".join(in_star)))
    for h in chart.owned[p]:
        others = _tenants(chart, h, exclude=p)
        if h == occ:
            if not others and out[0]["strength"] == "secondary":
                out[0] = _entry(occ, "occupied+owned", "primary", "owned house with no other planet in it")
            else:
                out[0]["kind"] = "occupied+owned"
            continue
        if others:
            out.append(_entry(h, "owned", "secondary", "other planets in the house: " + ", ".join(others)))
        else:
            out.append(_entry(h, "owned", "primary", "owned house with no other planet in it"))
    for h in _on_cusp_houses(chart, p, orb):
        hit = next((e for e in out if e["house"] == h), None)
        if hit is None:
            out.append(_entry(h, "on_cusp", "primary", f"within {orb:.4f} deg of cusp {h}"))
        elif hit["strength"] == "secondary":
            hit.update(strength="primary", reason=hit["reason"] + f"; within {orb:.4f} deg of cusp {h}")
    return out


def _rule2(chart: SimpleNamespace, s: str) -> list[dict[str, Any]]:
    """Houses of a star lord (steps 2 and 4): occupied house always primary, owned houses primary only when vacant."""
    occ = chart.house_of[s]
    out = [_entry(occ, "occupied", "primary", "house of the star lord is strongly signified even if others sit there")]
    for h in chart.owned[s]:
        if h == occ:
            out[0]["kind"] = "occupied+owned"
            continue
        tenants = _tenants(chart, h)
        if tenants:
            out.append(_entry(h, "owned", "secondary", "house not vacant: " + ", ".join(tenants)))
        else:
            out.append(_entry(h, "owned", "primary", "owned house is vacant"))
    return out


def _step(chart: SimpleNamespace, step: int, planet: str, orb: float) -> dict[str, Any]:
    star_rule = step in (2, 4)
    houses = _rule2(chart, planet) if star_rule else _rule1(chart, planet, orb)
    agents_out: list[dict[str, Any]] = []
    if planet in NODES and chart.opts.nodes_as_agents:
        node_strong = any(e["strength"] == "primary" for e in houses if e["kind"].startswith("occupied"))
        for a in chart.agents[planet]:
            ap = a["planet"]
            if ap in NODES:
                base = [_entry(chart.house_of[ap], "occupied", "secondary", "agent is the other node")]
            else:
                base = _rule1(chart, ap, orb)
                if not node_strong:
                    base = [dict(e, strength="secondary", reason="node is only a secondary significator") for e in base]
            for e in base:
                agents_out.append(dict(e, kind=f"agent:{ap} ({a['relation']})"))
    return {"step": step, "role": STEP_ROLES[step], "planet": planet, "rule": "rule 2 (star lord)" if star_rule else "rule 1 (planet)",
            "houses": houses, "agent_houses": agents_out}


# --------------------------------------------------------------------------------------
# public: 4-step
# --------------------------------------------------------------------------------------
def four_step_planet(chart: SimpleNamespace, planet: str, orb: float = ORB_DEG) -> dict[str, Any]:
    sub = chart.planets[planet]["sub_lord"]
    sub_star = chart.star_of[sub]
    steps = [_step(chart, 1, planet, orb), _step(chart, 2, chart.star_of[planet], orb),
             _step(chart, 3, sub, orb), _step(chart, 4, sub_star, orb)]
    primary: set[int] = set()
    every: set[int] = set()
    by_step_primary: dict[int, list[int]] = {}
    for st in steps:
        sp: set[int] = set()
        for e in st["houses"] + st["agent_houses"]:
            every.add(e["house"])
            if e["strength"] == "primary":
                sp.add(e["house"])
        by_step_primary[st["step"]] = sorted(sp)
        primary |= sp
    near = [q for q in PLANETS if q != planet
            and _cusp_distance(chart.planets[q]["longitude"], chart.planets[planet]["longitude"]) <= orb]
    return {"planet": planet, "star_lord": chart.star_of[planet], "sub_lord": sub, "sub_star_lord": sub_star, "steps": steps,
            "primary_houses": sorted(primary), "secondary_houses": sorted(every - primary),
            "primary_by_step": by_step_primary, "planets_conjunct_within_orb": near,
            "retrograde": chart.planets[planet]["retrograde"]}


def four_step(chart: SimpleNamespace, orb: float = ORB_DEG) -> dict[str, Any]:
    rows = [four_step_planet(chart, p, orb) for p in PLANETS]
    houses = []
    for h in range(1, 13):
        prim = [r["planet"] for r in rows if h in r["primary_houses"]]
        sec = [r["planet"] for r in rows if h in r["secondary_houses"]]
        houses.append({"house": h, "primary_significators": prim, "secondary_significators": sec})
    return {"method": "four_step", "cusp_orb_deg": round(orb, 6), "planets": rows, "houses": houses,
            "school": "Gondhalekar / Shri Bhatt 4-step theory; not Krishnamurti's own method (there the sub lord is the end-result indicator).",
            "rules_applied": ["rule 1 (steps 1, 3)", "rule 2 (steps 2, 4)", "Rahu/Ketu agents", "planet on cusp"],
            "not_implemented": ["upgrade of planets conjunct a strong planet (3d20')", "type-E (aspect) upgrades",
                                "denial test at steps 3-4 (see fruitful-significator filtering)"],
            "note": "Only primary significators count in this school; secondary ones are listed for completeness."}


# --------------------------------------------------------------------------------------
# public: 6-fold
# --------------------------------------------------------------------------------------
def six_fold_planet(chart: SimpleNamespace, planet: str) -> dict[str, Any]:
    sub = chart.planets[planet]["sub_lord"]
    return {"A_sub": [chart.house_of[sub]], "C_sub": sorted(chart.owned[sub]), "sub_lord": sub}


def six_fold_houses(chart: SimpleNamespace) -> list[dict[str, Any]]:
    rows = []
    for h in range(1, 13):
        occ = _tenants(chart, h)
        owner = chart.owner[h]
        in_star_occ = [p for p in PLANETS if chart.star_of[p] in occ]
        in_sub_occ = [p for p in PLANETS if chart.planets[p]["sub_lord"] in occ]
        in_star_own = [p for p in PLANETS if chart.star_of[p] == owner]
        in_sub_own = [p for p in PLANETS if chart.planets[p]["sub_lord"] == owner]
        every: list[str] = []
        for grp in (in_sub_occ, in_star_occ, occ, in_star_own, in_sub_own, [owner]):
            for p in grp:
                if p not in every:
                    every.append(p)
        rows.append({"house": h, "owner": owner,
                     "planets_in_sub_of_occupants": in_sub_occ, "planets_in_star_of_occupants": in_star_occ,
                     "occupants": occ, "planets_in_star_of_owner": in_star_own, "planets_in_sub_of_owner": in_sub_own,
                     "owner_planet": [owner], "all_six_fold": every})
    return rows


def six_fold(chart: SimpleNamespace) -> dict[str, Any]:
    planets = []
    for p in PLANETS:
        sf = six_fold_planet(chart, p)
        four = sorted(set(chart.sig[p]["A"] + chart.sig[p]["B"] + chart.sig[p]["C"] + chart.sig[p]["D"]))
        planets.append({"planet": p, "levels": {L: chart.sig[p][L] for L in "ABCD"}, "sub_levels": {"A_sub": sf["A_sub"], "C_sub": sf["C_sub"]},
                        "sub_lord": sf["sub_lord"], "signifies_4fold": four,
                        "signifies_6fold": sorted(set(four) | set(sf["A_sub"]) | set(sf["C_sub"]))})
    return {"method": "six_fold", "planets": planets, "houses": six_fold_houses(chart),
            "columns": ["planets in sub of occupant", "planets in star of occupant (A)", "occupant (B)",
                        "planets in star of owner (C)", "planets in sub of owner", "owner (D)"],
            "school": ("KP EZine 128 (Dhanabalan Rangasamy): 4-fold table plus sub-level columns. Not in the 1970-71 Readers; the sub is "
                       "taken as an outright significator, so it cannot also decide, and ruling planets are then needed to select."),
            "note": "Column letters of the original table are not fully legible in the source; the columns are named, not lettered."}


# --------------------------------------------------------------------------------------
# public: type-E significators (backlog item 2)
# --------------------------------------------------------------------------------------
def _relation(chart: SimpleNamespace, a: str, b_lon: float, b_sign: int, basis: str, orb: float,
              b_name: Optional[str] = None) -> Optional[str]:
    """Does planet `a` conjoin or aspect the point (b_lon, b_sign)? Aspects: sign-based houses (7th for all grahas, plus
    Mars 4/8, Jupiter 5/9, Saturn 3/10); Rahu/Ketu only conjoin. basis 'orb' = within `orb` degrees of the exact point;
    'sign' = sign relation only."""
    a_lon = chart.planets[a]["longitude"]
    a_sign = int(chart.planets[a]["sign_index"])
    if basis == "sign":
        if a_sign == b_sign:
            return "conjunction (same sign)"
    elif _cusp_distance(a_lon, b_lon) <= orb:
        return "conjunction (within orb)"
    if a in NODES:
        return None
    for off in ASPECTS.get(a, (7,)):
        if basis == "sign":
            if (a_sign + off - 1) % 12 == b_sign:
                return f"aspect ({off}th)"
        elif _cusp_distance((a_lon + (off - 1) * 30.0) % 360.0, b_lon) <= orb:
            return f"aspect ({off}th, within orb)"
    return None


def type_e_planet(chart: SimpleNamespace, planet: str, basis: str = "orb", orb: float = ORB_DEG) -> dict[str, Any]:
    own = set(chart.sig[planet]["A"] + chart.sig[planet]["B"] + chart.sig[planet]["C"] + chart.sig[planet]["D"])
    links: list[dict[str, Any]] = []
    houses: set[int] = set()
    for q in PLANETS:
        if q == planet:
            continue
        rel = _relation(chart, planet, chart.planets[q]["longitude"], int(chart.planets[q]["sign_index"]), basis, orb)
        if rel is None:
            continue
        qh = sorted(set(chart.sig[q]["A"] + chart.sig[q]["B"] + chart.sig[q]["C"] + chart.sig[q]["D"]))
        links.append({"planet": q, "relation": rel, "houses_of_that_planet": qh})
        houses.update(qh)
    cusp_links: list[dict[str, Any]] = []
    for i, c in enumerate(chart.cusps):
        rel = _relation(chart, planet, c, int(c // 30.0) % 12, basis, orb)
        if rel is not None:
            cusp_links.append({"house": i + 1, "relation": rel})
            houses.add(i + 1)
    return {"planet": planet, "e_houses": sorted(houses - own), "already_in_A_to_D": sorted(houses & own),
            "planet_links": links, "cusp_links": cusp_links}


def type_e(chart: SimpleNamespace, basis: str = "orb", orb: float = ORB_DEG) -> dict[str, Any]:
    if basis not in ("orb", "sign"):
        raise ValueError("type_e_basis must be 'orb' or 'sign'")
    rows = [type_e_planet(chart, p, basis, orb) for p in PLANETS]
    houses = [{"house": h, "E_planets": [r["planet"] for r in rows if h in r["e_houses"]]} for h in range(1, 13)]
    return {"method": "type_e", "basis": basis, "orb_deg": round(orb, 6) if basis == "orb" else None, "planets": rows, "houses": houses,
            "rule": ("Type E (E grade, Krishnamurti's later addition): a planet is an E significator of the houses signified (A-D) "
                     "by the planets it conjoins or aspects, and of the cusps it conjoins or aspects. Kept separate from A-D; "
                     "lower than D; houses already in the planet's own A-D are listed apart."),
            "note": ("Aspects are sign-based house aspects (7th; Mars 4/8, Jupiter 5/9, Saturn 3/10); Rahu/Ketu only conjoin. "
                     "The 4-step school upgrades weak E significators by conjunction/aspect with cusps or planets (not applied here).")}
