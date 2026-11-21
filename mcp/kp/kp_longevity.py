"""Longevity / maraka / badhaka facts (progress.md backlog item 10). Classical rule listing only.

This module NEVER estimates a lifespan and never says that a death will or will not happen. It lists, for the caller to present:
  * the cuspal sub lords of the 1st and 8th cusps and which 'supporting' (1, 5, 9, 10) and 'difficult' (6, 8, 12) houses they signify;
  * the classical group of the 'longevity' event (1, 5, 8, 11 against 2, 7 and the badhaka house), as house hits only;
  * the badhaka house (movable lagna 11, fixed 9, dual 7) and the maraka houses (2, 7) with their lords and what those lords signify;
  * joint periods in which the Dasha / Bhukti / Antara lords all signify a maraka or badhaka house, flagged 'critical period'
    (a classical timing flag, not a prediction).
Schools differ on the house groups; the sets used are returned in the output.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any, Optional

import kp_dasha as da
from kp_chart import csl_report, judge_event, signified
from kp_core import KPError
from kp_tables import SIGNS, badhaka_house, sign_index_of

SUPPORTING = (1, 5, 9, 10)
DIFFICULT = (6, 8, 12)
MARAKA = (2, 7)
LAGNA_KIND = ("movable", "fixed", "dual")
NOTE = ("Classical rule listing, not a prediction and not a lifespan estimate. Schools differ on the house groups; the groups used are "
        "stated in the output. How (and whether) to present any of this is for the caller to decide with care.")


def _cusp_block(chart: SimpleNamespace, cusp: int, levels: Optional[str]) -> dict[str, Any]:
    rep = csl_report(chart, cusp, levels)
    sig = set(rep["sub_lord_signifies"])
    return {"cusp": cusp, "sign": rep["sign"], "cuspal_sub_lord": rep["sub_lord"], "star_lord": rep["star_lord"],
            "sub_lord_signifies": sorted(sig), "supporting_houses_signified": sorted(sig & set(SUPPORTING)),
            "difficult_houses_signified": sorted(sig & set(DIFFICULT)), "sub_lord_retrograde": rep["sub_lord_retrograde"],
            "levels_used": rep["levels_used"]}


def _house_lord_block(chart: SimpleNamespace, house: int, role: str, levels: Optional[str]) -> dict[str, Any]:
    lord = chart.owner[house]
    return {"role": role, "house": house, "sign": SIGNS[int(chart.cusps[house - 1] // 30.0) % 12], "lord": lord,
            "lord_in_house": chart.house_of[lord], "lord_owns": list(chart.owned[lord]), "lord_signifies": signified(chart, lord, levels)}


def critical_houses(chart: SimpleNamespace, extra: Optional[list[int]] = None) -> list[int]:
    bad = badhaka_house(sign_index_of(chart.asc))
    hs = set(MARAKA) | {bad} | set(extra or [])
    if any((not isinstance(h, int)) or not 1 <= h <= 12 for h in hs):
        raise KPError("extra_houses must be integers 1..12.")
    return sorted(hs)


def longevity_facts(chart: SimpleNamespace, d: Optional[da.Dasha], start: datetime, end: datetime, depth: int = 3,
                    include_periods: bool = True, extra_houses: Optional[list[int]] = None, limit: int = 100,
                    levels: Optional[str] = None) -> dict[str, Any]:
    bad = badhaka_house(sign_index_of(chart.asc))
    kind = LAGNA_KIND[sign_index_of(chart.asc) % 3]
    crit = critical_houses(chart, extra_houses)
    lg = judge_event(chart, "longevity", levels=levels)
    out: dict[str, Any] = {
        "groups_used": {"supporting": list(SUPPORTING), "difficult": list(DIFFICULT), "maraka": list(MARAKA), "badhaka": bad,
                        "critical_period_houses": crit, "classical_longevity_group": {"favourable": lg["favourable_houses"],
                                                                                   "denial": lg["denial_houses"]}},
        "lagna": {"sign": SIGNS[sign_index_of(chart.asc)], "kind": kind},
        "ascendant_csl": _cusp_block(chart, 1, levels),
        "eighth_csl": _cusp_block(chart, 8, levels),
        "classical_group_hits": {"cusp": lg["primary_cusp"], "cuspal_sub_lord": lg["csl"]["sub_lord"],
                                 "favourable_signified_by_csl": lg["favourable_signified_by_csl"],
                                 "denial_signified_by_csl": lg["denial_signified_by_csl"]},
        "badhaka": _house_lord_block(chart, bad, f"badhaka house ({kind} lagna)", levels),
        "maraka": [_house_lord_block(chart, h, "maraka house", levels) for h in MARAKA],
        "note": NOTE,
    }
    if include_periods and d is not None:
        res = da.find_joint_periods(d, chart, crit, [], start, end, depth, False, 1, None, levels, limit)
        for p in res["periods"]:
            p["critical_period"] = True
            p["flag_reason"] = ("every lord of the period signifies a maraka / badhaka house; classical timing flag, "
                                "not a prediction")
        out["critical_periods"] = {"window": [start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")], "depth": depth,
                                   "houses": crit, "count": res["count"], "truncated": res["truncated"], "periods": res["periods"],
                                   "lord_status": res["lord_status"]}
    return out
