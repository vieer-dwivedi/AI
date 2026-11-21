"""KP electional helper (progress.md backlog item 15). Facts only: lists moments whose chart meets stated KP tests.

For each sampled moment at a place the chart is cast and three tests are evaluated for the chosen matter:
  lagna_sub  : the sub lord of the Ascendant signifies favourable houses of the matter and no denial house;
  event_csl  : the cuspal sub lord of the matter's primary cusp has a promise verdict at least `csl_min`;
  rp_support : at least `min_rp_support` distinct ruling planets (day lord, Moon / Lagna sign, star, sub lords) signify a favourable
               house without a denial house.
Consecutive passing samples are merged into windows. It is a scan at `step_minutes`, not an exact boundary search: use
get_lagna_changes / find_boundary_changes around a window for exact times. Not a prediction, not an auspiciousness verdict.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

import kp_chart as ch
from kp_core import Ctx, KPError, fmt_dt
from kp_tables import kp_lords

MAX_STEPS = 4000
VERDICT_RANK = {"not_promised": 0, "doubtful": 1, "weakly_promised": 2, "promised_with_obstacles": 2, "strongly_promised": 3}
CSL_MIN = {"none": 0, "weak": 2, "strong": 3}


def _sig(chart, planet: str, levels: Optional[str]) -> set[int]:
    return set(ch.signified(chart, planet, levels))


def evaluate(ctx: Ctx, t: datetime, lat: float, lon: float, event: Optional[str], custom_favourable: Optional[list[int]],
             custom_denial: Optional[list[int]], primary_cusp: Optional[int], levels: Optional[str],
             day_cache: dict) -> dict[str, Any]:
    jd = ctx.jd_from_local(t)
    chart = ch.chart_at(ctx, jd, lat, lon, t)
    j = ch.judge_event(chart, event, custom_favourable, custom_denial, primary_cusp, levels)
    fav, den = set(j["favourable_houses"]), set(j["denial_houses"])
    lag = kp_lords(chart.asc)
    ls = _sig(chart, lag["sub_lord"], levels)
    lag_fav, lag_den = sorted(ls & fav), sorted(ls & den - fav)
    ck = (t.date(), t.hour)
    if ck not in day_cache:
        day_cache[ck] = ctx.day_lord(t)["day_lord"]
    moon = chart.planets["Moon"]
    rps = {day_cache[ck], moon["sign_lord"], moon["star_lord"], moon["sub_lord"], lag["sign_lord"], lag["star_lord"], lag["sub_lord"]}
    supporting = []
    for p in sorted(rps):
        s = _sig(chart, p, levels)
        if s & fav and not (s & den - fav):
            supporting.append(p)
    return {"lagna_sub_lord": lag["sub_lord"], "lagna_sub_favourable": lag_fav, "lagna_sub_denial": lag_den,
            "lagna_sub_ok": bool(lag_fav) and not lag_den, "csl": j["csl"]["sub_lord"], "csl_verdict": j["verdict"],
            "ruling_planets": sorted(rps), "rp_supporting": supporting, "event": j["event"], "label": j["label"],
            "favourable_houses": j["favourable_houses"], "denial_houses": j["denial_houses"], "primary_cusp": j["primary_cusp"],
            "lagna": lag["dms"] + " " + lag["sign"], "moon_star_lord": moon["star_lord"], "moon_sub_lord": moon["sub_lord"]}


def find_windows(ctx: Ctx, start: datetime, end: datetime, lat: float, lon: float, event: Optional[str] = None,
                 custom_favourable: Optional[list[int]] = None, custom_denial: Optional[list[int]] = None,
                 primary_cusp: Optional[int] = None, step_minutes: float = 10.0, csl_min: str = "weak",
                 require_lagna_sub: bool = True, min_rp_support: int = 2, limit: int = 50, levels: Optional[str] = None) -> dict[str, Any]:
    if end <= start:
        raise KPError("`end` must be after `start`.")
    if step_minutes < 1:
        raise KPError("step_minutes must be at least 1.")
    if csl_min not in CSL_MIN:
        raise KPError("csl_min must be one of: none, weak, strong.")
    if min_rp_support < 0 or min_rp_support > 7:
        raise KPError("min_rp_support must be 0 to 7.")
    if limit < 1:
        raise KPError("limit must be at least 1.")
    n = int((end - start).total_seconds() // (step_minutes * 60)) + 1
    if n > MAX_STEPS:
        raise KPError(f"Too many samples ({n}); shorten the window or raise step_minutes (max {MAX_STEPS} samples).")
    if not event and not custom_favourable:
        raise KPError("Give an `event` name or `custom_favourable` houses.")
    need = CSL_MIN[csl_min]
    day_cache: dict = {}
    windows: list[dict[str, Any]] = []
    cur: Optional[dict[str, Any]] = None
    first: Optional[dict[str, Any]] = None
    tested = passed = 0
    meta: dict[str, Any] = {}
    for i in range(n):
        t = start + timedelta(minutes=i * step_minutes)
        r = evaluate(ctx, t, lat, lon, event, custom_favourable, custom_denial, primary_cusp, levels, day_cache)
        meta = r
        tested += 1
        tests = {"lagna_sub": r["lagna_sub_ok"], "event_csl": VERDICT_RANK[r["csl_verdict"]] >= need,
                 "rp_support": len(r["rp_supporting"]) >= min_rp_support}
        ok = (tests["lagna_sub"] or not require_lagna_sub) and tests["event_csl"] and tests["rp_support"]
        if ok:
            passed += 1
            if cur is None:
                cur = {"from": fmt_dt(t), "last_sample": fmt_dt(t), "samples": 1, "lagna_sub_lord": r["lagna_sub_lord"], "csl": r["csl"],
                       "csl_verdict": r["csl_verdict"], "ruling_planets_at_start": r["ruling_planets"],
                       "rp_supporting_at_start": r["rp_supporting"], "lagna_at_start": r["lagna"], "_last": t,
                       "lagna_sub_lords_seen": [r["lagna_sub_lord"]], "csl_verdicts_seen": [r["csl_verdict"]]}
            else:
                cur["last_sample"], cur["samples"], cur["_last"] = fmt_dt(t), cur["samples"] + 1, t
                if r["lagna_sub_lord"] not in cur["lagna_sub_lords_seen"]:
                    cur["lagna_sub_lords_seen"].append(r["lagna_sub_lord"])
                if r["csl_verdict"] not in cur["csl_verdicts_seen"]:
                    cur["csl_verdicts_seen"].append(r["csl_verdict"])
        elif cur is not None:
            cur["until_about"] = fmt_dt(t)
            windows.append(cur)
            cur = None
    if cur is not None:
        cur["until_about"] = fmt_dt(start + timedelta(minutes=n * step_minutes))
        cur["continues_after_window_end"] = True
        windows.append(cur)
    for w in windows:
        w.pop("_last", None)
    truncated = len(windows) > limit
    return {"start": fmt_dt(start), "end": fmt_dt(end), "step_minutes": step_minutes, "samples": tested, "samples_passing": passed,
            "matter": {"event": meta.get("event"), "label": meta.get("label"), "primary_cusp": meta.get("primary_cusp"),
                       "favourable": meta.get("favourable_houses"), "denial": meta.get("denial_houses")},
            "tests": {"require_lagna_sub": require_lagna_sub, "csl_min": csl_min, "min_rp_support": min_rp_support},
            "window_count": len(windows), "truncated": truncated, "windows": windows[:limit],
            "rule": ("A sampled moment passes when the Lagna sub lord signifies a favourable house and no denial house (if required), the "
                     "matter's cuspal sub lord has a promise verdict of at least `csl_min`, and at least `min_rp_support` distinct ruling "
                     "planets signify a favourable house without a denial house. Consecutive passing samples form a window."),
            "note": ("Scan at the step size: window edges are accurate to the step. Lagna lords change within minutes, so confirm the edges "
                     "with get_lagna_changes / find_boundary_changes. Facts only: not an auspiciousness verdict (no panchang, "
                     "tithi or yoga rules are applied) and not a prediction."),
            "school": "KP electional practice (Krishnamurti Readers: the Lagna / cusp sub lord and the ruling planets must support the matter)."}
