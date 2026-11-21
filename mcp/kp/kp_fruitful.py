"""Fruitful-significator selection (progress.md backlog item 3; research note R8).

KP first lists every planet that signifies the matter's houses, then keeps only the 'fruitful' ones. The schools select differently,
so each rule is an explicit method and every planet gets the reason it was kept or dropped:

  detrimental        drop planets that also signify a denial house (the 12th-from group of the matter).
  sub_lord           keep a planet only if its sub lord signifies at least one favourable house ("permission"; a sub lord that
                     signifies only denial houses, or nothing of the matter, withholds).
  ruling_planets     keep planets that are ruling planets at the moment of judgment (EZine 128: the sub cannot select without RPs).
  four_step_primary  4-step school: keep planets that are PRIMARY significators of a favourable house, and drop a planet whose
                     steps 3-4 primary houses touch only denial houses (its rule: such a planet does not fructify).

Several methods can be combined (comma separated): a planet must pass all of them. Facts and rules only, never a prediction.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional, Union

import kp_methods as km
from kp_chart import signified
from kp_tables import PLANETS

METHODS = ("detrimental", "sub_lord", "ruling_planets", "four_step_primary")
_ALIASES = {"detrimental_house": "detrimental", "sublord": "sub_lord", "sub": "sub_lord", "rp": "ruling_planets",
            "ruling": "ruling_planets", "four_step": "four_step_primary", "4step": "four_step_primary", "primary": "four_step_primary"}


def parse_methods(value: Union[None, str, list[str]]) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else str(value).split(",")
    out: list[str] = []
    for x in items:
        k = x.strip().lower().replace("-", "_").replace(" ", "_")
        if k in ("", "none"):
            continue
        k = _ALIASES.get(k, k)
        if k not in METHODS:
            raise ValueError(f"Unknown fruitful-significator method '{x}'. Use {list(METHODS)} (comma separated to combine) or 'none'.")
        if k not in out:
            out.append(k)
    return out


def analyze(chart: SimpleNamespace, fav: list[int], den: list[int], methods: list[str], rp: Optional[list[str]] = None,
            levels: Optional[str] = None, orb: float = km.ORB_DEG) -> dict[str, Any]:
    if "ruling_planets" in methods and not rp:
        raise ValueError("Method 'ruling_planets' needs the ruling planets of the moment of judgment: give `moment` (place and, optionally, date/time).")
    fav_s = set(fav)
    den_s = set(den) - fav_s
    rp_set = set(rp or [])
    rows: list[dict[str, Any]] = []
    for p in PLANETS:
        s = set(signified(chart, p, levels))
        f, d = s & fav_s, s & den_s
        sub = chart.planets[p]["sub_lord"]
        ss = set(signified(chart, sub, levels))
        sf, sd = ss & fav_s, ss & den_s
        star = chart.star_of[p]
        row: dict[str, Any] = {"planet": p, "signifies": sorted(s), "favourable": sorted(f), "denial": sorted(d), "sub_lord": sub,
                               "sub_lord_favourable": sorted(sf), "sub_lord_denial": sorted(sd),
                               "via_A_or_B": sorted((set(chart.sig[p]["A"]) | set(chart.sig[p]["B"])) & fav_s)}
        if rp_set:
            row["is_ruling_planet"] = p in rp_set
            row["in_star_of_ruling_planet"] = star in rp_set
            row["in_sub_of_ruling_planet"] = sub in rp_set
        row["candidate"] = bool(f)
        if not f:
            row.update(kept=False, verdicts={}, reasons=["signifies no favourable house of the matter"])
            rows.append(row)
            continue
        v: dict[str, dict[str, Any]] = {}
        if "detrimental" in methods:
            v["detrimental"] = {"kept": not d, "reason": ("signifies no denial house" if not d else
                                                          f"also signifies denial house(s) {sorted(d)}")}
        if "sub_lord" in methods:
            if sf:
                msg = f"sub lord {sub} signifies favourable house(s) {sorted(sf)}"
                if sd:
                    msg += f" (and denial house(s) {sorted(sd)}: mixed permission)"
                v["sub_lord"] = {"kept": True, "reason": msg}
            else:
                v["sub_lord"] = {"kept": False, "reason": (f"sub lord {sub} signifies only denial house(s) {sorted(sd)}" if sd
                                                           else f"sub lord {sub} signifies no house of the matter")}
        if "ruling_planets" in methods:
            ok = p in rp_set
            v["ruling_planets"] = {"kept": ok, "reason": "is a ruling planet at the moment of judgment" if ok
                                   else "is not a ruling planet at the moment of judgment"}
        if "four_step_primary" in methods:
            fs = km.four_step_planet(chart, p, orb)
            prim = set(fs["primary_houses"])
            s34 = set(fs["primary_by_step"][3]) | set(fs["primary_by_step"][4])
            negated = not (s34 & fav_s) and bool(s34 & den_s)
            if not prim & fav_s:
                v["four_step_primary"] = {"kept": False, "reason": "no favourable house among its primary significations"}
            elif negated:
                v["four_step_primary"] = {"kept": False, "reason": f"steps 3-4 primary houses {sorted(s34)} touch only denial houses"}
            else:
                v["four_step_primary"] = {"kept": True, "reason": f"primary for favourable house(s) {sorted(prim & fav_s)}"}
        row["verdicts"] = v
        row["kept"] = all(x["kept"] for x in v.values())
        row["reasons"] = [f"{k}: {x['reason']}" for k, x in v.items()] or ["no selection method applied"]
        rows.append(row)
    kept = [r for r in rows if r["candidate"] and r["kept"]]
    kept.sort(key=lambda r: (-len(r["via_A_or_B"]), len(r["denial"]), -len(r["favourable"])))
    dropped = [{"planet": r["planet"], "reasons": r["reasons"]} for r in rows if r["candidate"] and not r["kept"]]
    return {"methods": methods or ["none"], "favourable_houses": sorted(fav_s), "denial_houses": sorted(den_s),
            "ruling_planets_used": sorted(rp_set) if rp_set else None, "fruitful": [r["planet"] for r in kept],
            "dropped": dropped, "not_candidates": [r["planet"] for r in rows if not r["candidate"]], "planets": rows,
            "rule": ("Candidates signify a favourable house (levels A-D). Methods narrow them; planets are ordered by strength "
                     "(favourable house via level A/B first, fewer denial houses, more favourable houses)."),
            "caution": "Schools differ on which method selects fruitful significators; the result lists the rule used, it is not a prediction."}
