"""Rectification helpers (progress.md backlog item 12): Lagna-Moon lord link and event weights. Facts only.

KP rectification rule used as a pre-filter (R20 / R12): the star lord and sub lord of the Lagna must connect with the star lord and
sub lord of the Moon. 'Connect' = the same planet, or two of these lords in the same sign (association), or one aspecting the other
(sign-based house aspects: 7th for all; Mars 4/8, Jupiter 5/9, Saturn 3/10; Rahu/Ketu only associate).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from kp_chart import ASPECTS, NODES
from kp_core import KPError
from kp_tables import kp_lords

MAX_WEIGHT = 10.0
MOON_LINK_MODES = ("report", "require")
ANCHOR_EVENTS = ("accident", "marriage", "child_birth", "second_child", "service", "promotion", "foreign_travel", "property_purchase")
ANCHOR_NOTE = ("Events usually treated as anchors in KP rectification: accident, marriage, child birth, service / promotion, foreign "
               "travel, property purchase; the death of a parent has no event key here, so use `custom_favourable` houses "
               "(father: 9 with 8 / 12 from it, mother: 4 with 3 / 5) if you need it.")


def _relation(chart: SimpleNamespace, a: str, b: str) -> str | None:
    sa, sb = int(chart.planets[a]["sign_index"]), int(chart.planets[b]["sign_index"])
    if sa == sb:
        return "association (same sign)"
    if a in NODES:
        return None
    for off in ASPECTS.get(a, (7,)):
        if (sa + off - 1) % 12 == sb:
            return f"{a} aspects ({off})"
    return None


def lagna_moon_link(chart: SimpleNamespace) -> dict[str, Any]:
    lag = kp_lords(chart.asc)
    moon = chart.planets["Moon"]
    lg = {"star": lag["star_lord"], "sub": lag["sub_lord"]}
    mn = {"star": moon["star_lord"], "sub": moon["sub_lord"]}
    shared = sorted(set(lg.values()) & set(mn.values()))
    links: list[dict[str, str]] = []
    for ln, lp in lg.items():
        for mnn, mp in mn.items():
            if lp == mp:
                links.append({"lagna": f"{ln} lord {lp}", "moon": f"{mnn} lord {mp}", "kind": "same planet"})
                continue
            rel = _relation(chart, lp, mp) or _relation(chart, mp, lp)
            if rel:
                links.append({"lagna": f"{ln} lord {lp}", "moon": f"{mnn} lord {mp}", "kind": rel})
    return {"linked": bool(links), "lagna_lords": lg, "moon_lords": mn, "shared_planets": shared, "links": links}


def event_weight(ev: dict[str, Any]) -> float:
    w = ev.get("weight", 1.0)
    try:
        w = float(w)
    except (TypeError, ValueError) as exc:
        raise KPError("event `weight` must be a number.") from exc
    if not 0 < w <= MAX_WEIGHT:
        raise KPError(f"event `weight` must be above 0 and at most {MAX_WEIGHT:g}.")
    return w
