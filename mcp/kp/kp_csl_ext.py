"""Cuspal-sub-lord (CSL) significator extensions, progress.md backlog item 6. Pure helpers: no swisseph import.

Two optional readings that widen the houses a CSL is taken to signify. Both are OFF by default, are NOT pure Krishnamurti
(they are later or school-specific readings) and never change the A-D tables or the standard verdict: the widened houses are a
separate block (`csl_extension` on the CSL report) and `judge_event` shows a separate `extended_reading` next to the unchanged
`verdict`.

  own_bhava  KSK-note reading: when NO planet stands in the star of the CSL, the CSL is also taken as a significator of the
             bhava whose cusp it is the sub lord of ("the chief of that house"). Source wording could not be re-verified
             in this project; the rule encoded is exactly the sentence above.
  sub_chain  Kar / EZine 128 style "sub as significator": the sub lord of the CSL is also taken as an outright significator,
             so the house it occupies and the houses it owns are added. The sub is then both significator and decider, which
             the EZine 128 author notes breaks the usual KP division of roles.
  both       the union of the two.

Option: `KPOptions.csl_extension` = off (default) | own_bhava | sub_chain | both.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from kp_tables import PLANETS

MODES = ("off", "own_bhava", "sub_chain", "both")
SCHOOL = "Extension readings, not pure KP: schools differ and the rule wording of the original notes is thin."


def normalize(mode: Optional[str]) -> str:
    m = (mode or "off").strip().lower().replace("-", "_").replace(" ", "_")
    m = {"none": "off", "false": "off", "ksk": "own_bhava", "own": "own_bhava", "kar": "sub_chain", "sub": "sub_chain",
         "all": "both"}.get(m, m)
    if m not in MODES:
        raise ValueError(f"csl_extension must be one of {list(MODES)}")
    return m


def extension(chart: SimpleNamespace, cusp: int, csl: str, base_houses: list[int], mode: str) -> Optional[dict[str, Any]]:
    """Facts for one cusp. `base_houses` = houses the CSL signifies under the chart's level setting. None when mode is off."""
    if mode == "off":
        return None
    base = set(base_houses)
    added: set[int] = set()
    out: dict[str, Any] = {"mode": mode, "school": SCHOOL, "base_houses": sorted(base)}
    if mode in ("own_bhava", "both"):
        in_star = [q for q in PLANETS if q != csl and chart.star_of[q] == csl]
        met = not in_star
        out["own_bhava"] = {"rule": "no planet in the star of the CSL -> the CSL also signifies the bhava of its own cusp",
                            "condition_met": met, "planets_in_its_star": in_star, "house": cusp,
                            "adds": [cusp] if met and cusp not in base else []}
        if met:
            added.add(cusp)
    if mode in ("sub_chain", "both"):
        sub2 = chart.planets[csl]["sub_lord"]
        houses = sorted({chart.house_of[sub2], *chart.owned[sub2]})
        out["sub_chain"] = {"rule": "the sub lord of the CSL is an outright significator: the house it occupies and the houses it owns are added",
                            "sub_lord_of_csl": sub2, "occupies": chart.house_of[sub2], "owns": sorted(chart.owned[sub2]),
                            "houses": houses, "adds": sorted(set(houses) - base)}
        added.update(houses)
    out["added_houses"] = sorted(added - base)
    out["signifies_extended"] = sorted(base | added)
    return out
