"""Horary extras (progress.md backlog item 11): genuineness check, structured checklist, Lagna-point explanation, reminders.

Facts only. Every check is marked True / False with the planets used; no check says that the answer is yes or no, and no overall
verdict is produced. KP teachers differ on how these checks are weighed; the tool lists them, the caller decides.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from kp_chart import signified
from kp_retro import is_retro
from kp_tables import kp_lords

LAGNA_POINT_TEXT = {
    "start": "The Lagna is put at the start of the numbered division (KP textbook convention used by many software packages).",
    "middle": "The Lagna is put at the middle of the numbered division.",
    "end": "The Lagna is put at the end of the numbered division.",
    "actual_if_inside": ("The Lagna is the real Ascendant of the question moment when it falls inside the numbered division; otherwise "
                         "the start of the division is used."),
}
REMINDERS = [
    "First sincere asking only: judge a matter on the first time it is asked; repeated asking of the same question is not judged "
    "again (user-process rule, not computable; pass asked_before=true to have it flagged).",
    "The number (1-249) must be spontaneous, chosen by the querent for this one question; the tool cannot check that.",
    "One question, one chart: the matter's cusp, its favourable and denial houses are stated in the judgment, not guessed.",
    "The Lagna changes fast: the sub lord of the horary Lagna is fixed by the number, but the Moon and the ruling planets are those of "
    "the question moment, so the exact time matters.",
]


def _in_star_of(chart: SimpleNamespace, planet: str, other: str) -> bool:
    return planet != other and chart.star_of[planet] == other


def genuineness(chart: SimpleNamespace, judgment: dict[str, Any], rps: set[str], levels: Optional[str]) -> dict[str, Any]:
    lag = kp_lords(chart.asc)
    lagna_star = lag["star_lord"]
    moon = chart.planets["Moon"]
    moon_sub, moon_star = moon["sub_lord"], moon["star_lord"]
    fav = set(judgment["favourable_houses"])
    star_sig = set(signified(chart, moon_star, levels))
    same = lagna_star == moon_sub
    a = _in_star_of(chart, moon_sub, lagna_star)
    b = _in_star_of(chart, lagna_star, moon_sub)
    linked = same or a or b
    checks = [
        {"id": "lagna_star_lord_linked_to_moon_sub_lord", "value": linked,
         "detail": {"lagna_star_lord": lagna_star, "moon_sub_lord": moon_sub, "same_planet": same,
                    "moon_sub_lord_in_star_of_lagna_star_lord": a, "lagna_star_lord_in_star_of_moon_sub_lord": b}},
        {"id": "moon_star_lord_signifies_matter_houses", "value": bool(star_sig & fav),
         "detail": {"moon_star_lord": moon_star, "signifies": sorted(star_sig), "favourable_houses": sorted(fav),
                    "favourable_signified": sorted(star_sig & fav)}},
        {"id": "lagna_star_lord_is_ruling_planet", "value": lagna_star in rps, "detail": {"planet": lagna_star}},
        {"id": "moon_sub_lord_is_ruling_planet", "value": moon_sub in rps, "detail": {"planet": moon_sub}},
    ]
    return {"rule": ("Textbook genuineness checks: the Lagna star lord is linked to the Moon sub lord (same planet, or one in the star "
                     "of the other), and the Moon's star lord signifies the houses of the matter."),
            "checks": checks, "passed": sum(1 for c in checks if c["value"]), "total": len(checks)}


def checklist(chart: SimpleNamespace, judgment: dict[str, Any]) -> dict[str, Any]:
    csl = judgment["csl"]["sub_lord"]
    star = chart.star_of[csl]
    fav = judgment["favourable_houses"]
    fav_hit = judgment["favourable_signified_by_csl"]
    den_hit = judgment["denial_signified_by_csl"]
    need = min(2, len(fav))
    rps = set(judgment.get("ruling_planet_check", {}).get("ruling_planets", []))
    moon = judgment.get("moon_sub_lord_check", {})
    items = [
        ("csl_not_retrograde", not is_retro(chart, csl), {"csl": csl, "retrograde": is_retro(chart, csl)}),
        ("csl_not_in_retrograde_star", not is_retro(chart, star), {"csl": csl, "star_lord": star, "star_lord_retrograde": is_retro(chart, star)}),
        ("csl_linked_to_supporting_house", bool(fav_hit), {"favourable_houses": fav, "signified": fav_hit}),
        ("csl_signifies_enough_favourable_houses", len(fav_hit) >= need > 0, {"needed": need, "signified": fav_hit}),
        ("csl_signifies_no_denial_house", not den_hit, {"denial_signified": den_hit}),
        ("csl_is_ruling_planet", csl in rps, {"csl": csl}),
        ("csl_star_lord_is_ruling_planet", star in rps, {"csl": csl, "star_lord_of_csl": star}),
        ("moon_sub_lord_signifies_favourable_house", bool(moon.get("favourable")), {"moon_sub_lord": moon.get("moon_sub_lord"),
                                                                                    "favourable": moon.get("favourable", [])}),
    ]
    rows = [{"id": i, "value": bool(v), "detail": d} for i, v, d in items]
    return {"primary_cusp": judgment["primary_cusp"], "cuspal_sub_lord": csl, "items": rows,
            "passed": sum(1 for r in rows if r["value"]), "total": len(rows), "all_passed": all(r["value"] for r in rows),
            "note": ("Structured checklist of the horary conditions teachers name; each line is a fact. How many lines must hold, and "
                     "whether the retrograde lines apply to this matter, differs between schools; the standard verdict is unchanged.")}


def lagna_point_explanation(info: dict[str, Any]) -> dict[str, Any]:
    used = info["lagna_point"]
    div = info["division"]
    act = info["actual_ascendant_at_question"]
    return {"used": used, "meaning": LAGNA_POINT_TEXT[used], "number": info["number"],
            "division": {"start_dms": div.get("start_dms"), "end_dms": div.get("end_dms"), "sign": div.get("sign"),
                         "star_lord": div.get("star_lord"), "sub_lord": div.get("sub_lord")},
            "target_longitude": info["target_longitude"], "cusp_time_local": info["cusp_time_local"],
            "question_time_local": info["question_time_local"],
            "actual_ascendant": {"dms": act["dms"], "kp_number": act["kp_number"], "inside_the_numbered_division": act["inside_the_numbered_division"]},
            "alternatives": LAGNA_POINT_TEXT,
            "note": ("Textbooks differ on the point inside the division. The sign / star / sub of the Lagna is the same at every point, "
                     "but the other cusps move slightly with it; re-run with another `lagna_point` to see whether a cusp sub lord changes.")}


def horary_extras(chart: SimpleNamespace, info: dict[str, Any], rp: dict[str, Any], judgment: dict[str, Any], levels: Optional[str] = None,
                  asked_before: bool = False) -> dict[str, Any]:
    rps = {p["planet"] for p in rp["ruling_planets"]}
    out: dict[str, Any] = {
        "genuineness": genuineness(chart, judgment, rps, levels),
        "horary_checklist": checklist(chart, judgment),
        "lagna_point": lagna_point_explanation(info),
        "reminders": REMINDERS,
    }
    if asked_before:
        out["first_asking_only"] = {"declared_repeat": True,
                                    "note": "The caller says this matter was asked before: by the first-asking rule this chart is not the one to judge."}
    return out
