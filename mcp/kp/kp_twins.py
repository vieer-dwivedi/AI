"""Twin / close-birth comparison (progress.md backlog item 13). Facts only.

KP shows that two births a few minutes apart (twins) can differ: the cusp sub lords change within minutes, so houses, promise tests
and dasha balances can diverge. `compare_charts` lists, for two natal charts, which cusps and planets differ at the sign / star / sub /
sub-sub level, the Lagna difference, the dasha balance and the events whose promise verdict differs. It makes no claim about people.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any

import kp_dasha as da
from kp_chart import promise_table
from kp_tables import PLANETS, kp_lords

LEVELS = ("sign_lord", "star_lord", "sub_lord", "sub_sub_lord")


def _diff_levels(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    return [lv.replace("_lord", "") for lv in LEVELS if a[lv] != b[lv]]


def _lag_diff(a: float, b: float) -> float:
    return ((b - a + 180.0) % 360.0) - 180.0


def compare_charts(ca: SimpleNamespace, cb: SimpleNamespace, dt_a: datetime, dt_b: datetime, same_place: bool) -> dict[str, Any]:
    cusps = []
    for i in range(12):
        ra, rb = kp_lords(ca.cusps[i]), kp_lords(cb.cusps[i])
        diff = _diff_levels(ra, rb)
        cusps.append({"house": i + 1, "a": {k: ra[k] for k in ("dms", "sign", *LEVELS)}, "b": {k: rb[k] for k in ("dms", "sign", *LEVELS)},
                      "differs_at": diff, "sub_lord_differs": "sub" in diff, "moved_degrees": round(_lag_diff(ca.cusps[i], cb.cusps[i]), 6)})
    planets = []
    for p in PLANETS:
        pa, pb = ca.planets[p], cb.planets[p]
        d = [lv.replace("_lord", "") for lv in LEVELS if pa[lv] != pb[lv]]
        hd = ca.house_of[p] != cb.house_of[p]
        if d or hd:
            planets.append({"planet": p, "differs_at": d, "house_a": ca.house_of[p], "house_b": cb.house_of[p], "house_differs": hd,
                            "a": {lv: pa[lv] for lv in LEVELS}, "b": {lv: pb[lv] for lv in LEVELS}})
    pt_a, pt_b = promise_table(ca), promise_table(cb)
    events = [{"event": x["event"], "label": x["label"], "primary_cusp": x["primary_cusp"], "csl_a": x["csl"], "csl_b": y["csl"],
               "verdict_a": x["verdict"], "verdict_b": y["verdict"]} for x, y in zip(pt_a, pt_b) if x["verdict"] != y["verdict"] or x["csl"] != y["csl"]]
    da_, db_ = da.make_dasha(ca, dt_a), da.make_dasha(cb, dt_b)
    ba, bb = da_.balance_at_birth(), db_.balance_at_birth()
    sub_diff = [c["house"] for c in cusps if c["sub_lord_differs"]]
    return {
        "minutes_apart": round((dt_b - dt_a).total_seconds() / 60.0, 3), "same_place": same_place,
        "lagna": {"a": cusps[0]["a"], "b": cusps[0]["b"], "differs_at": cusps[0]["differs_at"], "moved_degrees": cusps[0]["moved_degrees"]},
        "cusps": cusps,
        "summary": {"cusps_with_different_sub_lord": sub_diff, "count": len(sub_diff),
                    "cusps_with_different_star_lord": [c["house"] for c in cusps if "star" in c["differs_at"]],
                    "cusps_with_different_sign": [c["house"] for c in cusps if "sign" in c["differs_at"]],
                    "planets_that_differ": [p["planet"] for p in planets], "events_with_different_verdict_or_csl": len(events)},
        "planets": planets,
        "dasha_balance": {"a": ba, "b": bb, "same_start_lord": ba["lord"] == bb["lord"],
                          "balance_difference_days": round((bb["balance_years"] - ba["balance_years"]) * da_.year_days, 3)},
        "events": events,
        "rule": ("KP twin demonstration: the cuspal sub lords (and so the promise tests) change within minutes; the sign / star / sub of "
                 "each cusp is compared level by level, no claim is made about the people."),
        "note": ("The planets move little in minutes (the Moon is the fastest); most differences come from the cusps. Different places "
                 "or dates also change the cusps: `same_place` says whether the two inputs share the place."),
    }
