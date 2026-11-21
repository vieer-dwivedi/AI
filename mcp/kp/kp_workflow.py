"""Guided KP judgment workflow (progress.md backlog item 26): composition of existing engines in the teaching order, no new astrology.

Steps: 1 matter and cusp -> 2 cuspal sub lord -> 3 promise -> 4 dasha / bhukti / antara timing -> 5 transit check on a day ->
6 ruling planets verification. Each step returns the rule used, the houses used and a status (pass / partial / fail / n.a.).
Statuses describe whether the stated rule is met; there is no overall verdict or prediction.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import kp_chart as ch
import kp_dasha as da
import kp_moment as mo
import kp_transit as tr
from kp_core import Ctx, fmt_dt

PROMISE_STATUS = {"strongly_promised": "pass", "promised_with_obstacles": "partial", "weakly_promised": "partial",
                  "doubtful": "fail", "not_promised": "fail"}


def run(ctx: Ctx, chart, birth_dt: datetime, lat: float, lon: float, start: datetime, end: datetime, when: Optional[datetime],
        event: Optional[str], custom_favourable: Optional[list[int]], custom_denial: Optional[list[int]], primary_cusp: Optional[int],
        depth: int, limit: int, levels: Optional[str]) -> dict[str, Any]:
    j = ch.judge_event(chart, event, custom_favourable, custom_denial, primary_cusp, levels)
    fav, den = j["favourable_houses"], j["denial_houses"]
    steps: list[dict[str, Any]] = []
    steps.append({"step": 1, "name": "Matter and cusp", "status": "info",
                  "rule": "The matter is read from its primary cusp; favourable houses support it, denial houses (12th from them etc.) oppose it.",
                  "event": j["event"], "label": j["label"], "primary_cusp": j["primary_cusp"], "favourable_houses": fav, "denial_houses": den})
    rep = j["csl"]
    steps.append({"step": 2, "name": "Cuspal sub lord (CSL)", "status": "info",
                  "rule": "The sub lord of the primary cusp decides; its star lord's houses count first, then its own.",
                  "csl": rep["sub_lord"] if "sub_lord" in rep else None, "csl_signifies": rep["sub_lord_signifies"],
                  "csl_star_lord_signifies": rep["star_lord_signifies"], "retrograde_check": rep["retrograde_check"]})
    steps.append({"step": 3, "name": "Promise", "status": PROMISE_STATUS[j["verdict"]],
                  "rule": "The cuspal sub lord must signify the favourable houses and not the denial houses.", "verdict": j["verdict"], "reason": j["verdict_reason"],
                  "favourable_signified_by_csl": j["favourable_signified_by_csl"], "denial_signified_by_csl": j["denial_signified_by_csl"],
                  "houses_used": fav + den})
    d = da.make_dasha(chart, birth_dt)
    jp = da.find_joint_periods(d, chart, fav, den, start, end, depth, False, None, None, levels, limit, None, chart.opts.retrograde_rule)
    periods = jp["periods"]
    steps.append({"step": 4, "name": "Timing: Dasha / Bhukti / Antara", "status": "pass" if periods else "fail",
                  "rule": (f"Every lord down to depth {depth} must signify a favourable house, together covering at least "
                           f"{min(2, len(fav))} of them; the matter needs the promise of step 3 first."),
                  "window": [fmt_dt(start), fmt_dt(end)], "period_count": len(periods), "periods_shown": periods[:limit],
                  "promise_pass_before_timing": steps[2]["status"] != "fail"})
    if when is not None:
        v = tr.verify_event_day(ctx, chart, birth_dt, when, lat, lon, 4, None, True, fav, den,
                                {"label": j["label"], "event": j["event"], "favourable": fav, "denial": den, "primary_cusp": j["primary_cusp"]}, levels)
        both = v["summary"].get("both_luminaries_touch") if isinstance(v.get("summary"), dict) else None
        status = "pass" if both is True else ("fail" if both is False else "n.a.")
        steps.append({"step": 5, "name": "Transit check", "status": status,
                      "rule": ("A transiting planet touches a dasha lord when its sign or star lord is that lord; no event unless transiting "
                               "Sun and Moon both touch the wanted lords (Reader 5 reading)."),
                      "when": fmt_dt(when), "summary": v["summary"], "transits": v["transits"], "running_dasha": v["running_dasha"]})
    else:
        steps.append({"step": 5, "name": "Transit check", "status": "n.a.", "rule": "Give `when` (a day or moment) to run the transit check."})
    at = when or start
    rp = mo.ruling_planets(ctx, at, lat, lon, True, chart)
    rps = [p["planet"] for p in rp["ruling_planets"]]
    fl = ch.lord_significations(chart, rps, fav, den, levels)
    supp = [x["planet"] for x in fl if x["favourable"] and not x["denial"]]
    steps.append({"step": 6, "name": "Ruling planets verification", "status": "pass" if len(supp) >= 2 else ("partial" if supp else "fail"),
                  "rule": ("Ruling planets (day lord, Moon and Lagna sign / star / sub lords) at the moment of judgment should be significators "
                           "of the matter: pass = at least two signify favourable houses without denial houses."),
                  "at": fmt_dt(at), "ruling_planets": rps, "supporting": supp, "detail": fl, "retrieved_for": "when" if when else "start of window"})
    counts = {k: sum(1 for s in steps if s["status"] == k) for k in ("info", "pass", "partial", "fail", "n.a.")}
    return {"matter": {"event": j["event"], "label": j["label"]}, "steps": steps, "status_counts": counts,
            "note": ("Composition of the existing tools in the KP teaching order. Statuses say whether each stated rule is met; there is no "
                     "overall verdict and nothing here is a prediction. Schools differ on several rules (see the individual tools).")}
