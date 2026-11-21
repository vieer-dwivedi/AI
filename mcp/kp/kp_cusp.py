"""Cusp proximity ("bhava chalit") rule, progress.md backlog item 5. Pure helpers: no swisseph import.

Rule (K.P. practice; R17): a planet that stands close BEHIND a cusp (inside house h but within `cusp_orb_deg` of the cusp of
house h+1) also signifies house h+1; a planet that stands just PAST a cusp (within the orb after the cusp of its own house) is
a strong significator of that house. The usual orb is 3d20' (200').

Two separate switches, both off by default so existing results never change:
  * `cusp_orb_deg`          : turns the rule on and sets the orb. Adds `near_cusp` facts to the chart and to the significator
                              tables, plus `signifies_with_near_cusp` per planet. The A-D tables are NOT altered.
  * `cusp_proximity_apply`  : also feeds the shift into A-D. The planet counts as an occupant (level B) of the next house, and so
                              the planets in its star count at level A of that house; judgments, fruitful-significator
                              selection and period search then see it. Level C / D (ownership) never change.
The shift never moves the planet's own house: its natal placement stays as cast. Schools differ on this rule.
"""

from __future__ import annotations

from typing import Any, Optional

from kp_tables import PLANETS, dms

DEFAULT_ORB_DEG = 200.0 / 60.0  # 3d20'


def near_cusp_entries(planets: dict[str, dict[str, Any]], cusps: list[float], orb: Optional[float]) -> list[dict[str, Any]]:
    """Facts only. `planets[name]` needs `longitude` and `house`. Returns [] when `orb` is falsy (rule off).

    relation 'behind_cusp': planet is in house h and the cusp of house h+1 lies ahead of it within the orb.
    relation 'past_cusp'  : planet is within the orb after the cusp of its own house (0 = exactly on the cusp).
    """
    if not orb:
        return []
    out: list[dict[str, Any]] = []
    for p in PLANETS:
        lon = planets[p]["longitude"]
        h = int(planets[p]["house"])
        ahead = (cusps[h % 12] - lon) % 360.0
        if 0.0 < ahead <= orb:
            nxt = h % 12 + 1
            out.append({"planet": p, "house": h, "relation": "behind_cusp", "cusp": nxt, "distance_deg": round(ahead, 6),
                        "distance_dms": dms(ahead),
                        "effect": f"also signifies house {nxt} (stands {dms(ahead)} before its cusp)"})
        after = (lon - cusps[h - 1]) % 360.0
        if after <= orb:
            out.append({"planet": p, "house": h, "relation": "past_cusp", "cusp": h, "distance_deg": round(after, 6),
                        "distance_dms": dms(after),
                        "effect": f"strong significator of house {h} (stands {dms(after)} past its cusp)"})
    return out


def shift_map(entries: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Extra houses per planet when the shift is applied (only 'behind_cusp' entries move a planet)."""
    extra: dict[str, list[int]] = {p: [] for p in PLANETS}
    for e in entries:
        if e["relation"] == "behind_cusp" and e["cusp"] not in extra[e["planet"]]:
            extra[e["planet"]].append(e["cusp"])
    return extra


def block(entries: list[dict[str, Any]], orb: Optional[float], applied: bool) -> Optional[dict[str, Any]]:
    """The `near_cusp` output block, or None when the rule is off."""
    if not orb:
        return None
    return {"orb_deg": round(orb, 6), "orb_dms": dms(orb), "applied_to_A_to_D": applied, "count": len(entries), "entries": entries,
            "rule": ("A planet standing within the orb behind a cusp also signifies the house that cusp opens; a planet within the orb "
                     "past a cusp is a strong significator of that house (bhava chalit idea, K.P. practice)."),
            "note": ("Applied to A-D (planet counts as occupant of the next house, level B and so level A)." if applied else
                     "Reported only: the A-D tables are unchanged. Use signifies_with_near_cusp, or set cusp_proximity_apply to feed the shift into A-D."),
            "school": "Schools differ; the 4-step theory has its own 3d20' cusp rule (kp_methods) that this option does not change."}
