"""KP transit rule set and event-day verification (progress.md backlog item 9). Facts only; no prediction.

Rule encoded (KP Reader 5 and the wider KP practice, schools differ and the output says which step was used):
  * A transit only matters inside a running dasha / bhukti / antara (/ sookshma) that already allows the event.
  * A transiting planet "touches" a dasha lord when the planet's SIGN lord or STAR lord is that lord (the two-step rule of Reader 5);
    a match through the SUB lord only is reported separately (some teachers use it). The planet being the lord itself is also reported.
  * Slow planets (Jupiter, Saturn) pick the months, the Sun the week, the Moon the day: "no event unless the transiting Sun and Moon
    also touch the wanted lords" is reported as `both_luminaries_touch`.
  * The ruling planets of the moment are checked the same way (a transit lord that is also a ruling planet is stronger).
Every line is marked 'match' or 'no match'; nothing here says that an event will or will not happen.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any, Optional

import kp_dasha as da
import kp_moment as mo
from kp_chart import signified
from kp_core import Ctx, KPError, fmt_dt
from kp_tables import PLANETS, kp_lords

DEFAULT_PLANETS = ("Jupiter", "Saturn", "Sun", "Moon")
ALLOWED_PLANETS = tuple(PLANETS)
SPEED_ROLE = {"Jupiter": "slow (months)", "Saturn": "slow (months)", "Rahu": "slow (months)", "Ketu": "slow (months)",
              "Sun": "weeks", "Moon": "days", "Mars": "weeks", "Venus": "weeks", "Mercury": "weeks"}
SCHOOL = ("KP transit rule (Reader 5): the transiting planet's sign lord / star lord is a dasha-bhukti-antara lord; the sub lord is "
          "reported separately. Supporting check only: the dasha lords must already signify the event houses.")


def _roles(targets: dict[str, list[str]], lord: str) -> list[str]:
    return list(targets.get(lord, []))


def _touch(planet: str, lords: dict[str, Any], targets: dict[str, list[str]]) -> dict[str, Any]:
    sign_r, star_r, sub_r = _roles(targets, lords["sign_lord"]), _roles(targets, lords["star_lord"]), _roles(targets, lords["sub_lord"])
    self_r = _roles(targets, planet)
    two_step = bool(sign_r or star_r)
    status = "match" if two_step else ("sub lord only" if sub_r else "no match")
    return {"status": status, "two_step_match": two_step,
            "sign_lord": {"lord": lords["sign_lord"], "roles": sign_r},
            "star_lord": {"lord": lords["star_lord"], "roles": star_r},
            "sub_lord": {"lord": lords["sub_lord"], "roles": sub_r},
            "planet_is_a_target_lord": {"is": bool(self_r), "roles": self_r}}


def verify_event_day(ctx: Ctx, chart: SimpleNamespace, birth_dt: datetime, at: datetime, lat: float, lon: float, depth: int = 4,
                     planets: Optional[list[str]] = None, extended_rps: bool = True, fav: Optional[list[int]] = None,
                     den: Optional[list[int]] = None, event: Optional[dict[str, Any]] = None, levels: Optional[str] = None
                     ) -> dict[str, Any]:
    if not 1 <= depth <= 5:
        raise KPError("depth must be 1 (Dasha) .. 5 (Prana).")
    pl = list(planets) if planets else list(DEFAULT_PLANETS)
    bad = [p for p in pl if p not in ALLOWED_PLANETS]
    if bad:
        raise KPError(f"planets must be among {list(ALLOWED_PLANETS)}; got {bad}")
    pl = list(dict.fromkeys(pl))

    d = da.make_dasha(chart, birth_dt)
    ladder = da.running_ladder(d, chart, at, depth)
    targets: dict[str, list[str]] = {}
    ladder_out = []
    fav_set, den_set = set(fav or []), set(den or [])
    for r in ladder:
        lord = r["lords"][-1]
        targets.setdefault(lord, []).append(r["label"])
        item = {"level": r["level"], "label": r["label"], "lord": lord, "start": r["start"], "end": r["end"],
                "lord_signifies": r["lord_signifies"]}
        if fav_set:
            st = da.lord_status(chart, lord, fav_set, den_set, levels)
            item["event_houses"] = {"favourable": st["favourable"], "denial": st["denial"],
                                    "sub_lord": st["sub_lord"], "sub_lord_favourable": st["sub_lord_favourable"],
                                    "sub_lord_denial": st["sub_lord_denial"],
                                    "status": "signifies event houses" if st["favourable"] else "does not signify event houses"}
        ladder_out.append(item)

    rp = mo.ruling_planets(ctx, at, lat, lon, extended_rps, chart)
    rp_targets: dict[str, list[str]] = {}
    for it in rp["ruling_planets"]:
        rp_targets[it["planet"]] = it["roles"]

    jd = ctx.jd_from_local(at)
    pos = ctx.positions(jd)
    rows = []
    for p in pl:
        lords = kp_lords(pos[p]["lon"])
        dasha_touch = _touch(p, lords, targets)
        rp_touch = _touch(p, lords, rp_targets)
        rows.append({
            "planet": p, "pace": SPEED_ROLE[p], "dms": lords["dms"], "sign": lords["sign"], "nakshatra": lords["nakshatra"],
            "retrograde": bool(pos[p]["retro"]), "dasha_lords": dasha_touch,
            "ruling_planets": {"status": rp_touch["status"], "two_step_match": rp_touch["two_step_match"],
                               "sign_lord": rp_touch["sign_lord"], "star_lord": rp_touch["star_lord"], "sub_lord": rp_touch["sub_lord"],
                               "planet_is_a_ruling_planet": rp_touch["planet_is_a_target_lord"]},
        })

    by = {r["planet"]: r for r in rows}

    def ok(p: str) -> Optional[bool]:
        return by[p]["dasha_lords"]["two_step_match"] if p in by else None

    both = (ok("Sun") and ok("Moon")) if "Sun" in by and "Moon" in by else None
    matches = [r["planet"] for r in rows if r["dasha_lords"]["two_step_match"]]
    rp_matches = [r["planet"] for r in rows if r["ruling_planets"]["two_step_match"]]
    dasha_ok = None
    if fav_set:
        dasha_ok = [x["lord"] for x in ladder_out if x["event_houses"]["favourable"]]
    return {
        "datetime_local": fmt_dt(at), "school": SCHOOL, "event": event,
        "running_dasha": ladder_out, "target_lords": targets,
        "ruling_planets_at_moment": {"day_lord": rp["day"]["day_lord"], "set": sorted(rp_targets), "entries": rp["entries"],
                                     "appearing_most": rp["appearing_most"]},
        "transits": rows,
        "summary": {
            "planets_touching_dasha_lords": matches, "planets_touching_ruling_planets": rp_matches,
            "jupiter_touches": ok("Jupiter"), "saturn_touches": ok("Saturn"),
            "sun_touches": ok("Sun"), "moon_touches": ok("Moon"), "both_luminaries_touch": both,
            "dasha_chain_lords_signifying_event_houses": dasha_ok,
            "every_dasha_lord_signifies_event_houses": (len(dasha_ok) == len(ladder_out)) if dasha_ok is not None else None,
        },
        "rule": ("A transiting planet 'touches' a dasha lord when its sign lord or star lord is that lord (two-step rule); a match "
                 "through the sub lord alone is shown as 'sub lord only'. Marked match / no match, never a prediction."),
        "note": ("Moon moves one sub in a few hours: give the clock time of the event. KP schools differ on whether the transiting "
                 "sub lord counts; both readings are shown. Retrograde flags are the transit planet's own state at that moment."),
    }
