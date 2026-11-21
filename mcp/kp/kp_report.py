"""One-page condensed KP report as markdown text (progress.md backlog item 27). Composition of existing facts, no new astrology."""

from __future__ import annotations

from typing import Any

import kp_chart as ch
from kp_tables import PLANETS, kp_lords


def _h(x: list[int]) -> str:
    return ",".join(str(i) for i in x) if x else "-"


def _table(head: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def build(chart, meta: dict[str, Any], balance: dict[str, Any], ladder: list[dict[str, Any]], rps: dict[str, Any], on_date: str) -> str:
    L: list[str] = []
    L.append(f"# KP chart: {meta.get('place', '')} {meta.get('datetime_local', '')}".rstrip())
    L.append(f"Ayanamsa {chart.opts.ayanamsa}, nodes {chart.opts.node_type}, houses {chart.houses['system']}. Facts and classical KP rules only; not a prediction.")
    asc = kp_lords(chart.asc)
    L.append(f"Ascendant {asc['dms']} {asc['sign']} | star {asc['star_lord']} | sub {asc['sub_lord']} | sub-sub {asc['sub_sub_lord']}")
    L.append("")
    L.append("## Planets")
    rows = []
    for name, d in chart.planets.items():
        rows.append([name + (" (R)" if d["retrograde"] else ""), d["dms"], d["sign"], d["star_lord"], d["sub_lord"], d["sub_sub_lord"], d["house"],
                     _h(ch.signified(chart, name, "ABCD")) if name in PLANETS else "-"])
    L += _table(["Planet", "Longitude", "Sign", "Star", "Sub", "Sub-sub", "House", "Signifies"], rows)
    L.append("")
    L.append("## Cusps and cuspal links")
    rows = []
    for r in ch.cusp_rows(chart):
        sub = r["sub_lord"]
        rows.append([r["house"], r["dms"], r["sign"], r["sign_lord"], r["star_lord"], sub, r["sub_sub_lord"],
                     _h(ch.signified(chart, sub, "ABCD")), ",".join(r["occupants"]) or "-"])
    L += _table(["Cusp", "Longitude", "Sign", "Sign lord", "Star", "Sub (CSL)", "Sub-sub", "CSL signifies", "Occupants"], rows)
    L.append("")
    L.append("## Significators by house (A star of occupant | B occupant | C star of owner | D owner)")
    rows = [[h["house"], ",".join(h["A_in_star_of_occupants"]) or "-", ",".join(h["B_occupants"]) or "-",
             ",".join(h["C_in_star_of_owner"]) or "-", ",".join(h["D_owner"])] for h in ch.house_significators(chart)]
    L += _table(["House", "A", "B", "C", "D"], rows)
    L.append("")
    L.append(f"## Dasha (year 365.25 days). Balance at birth: {balance['lord']} {balance['balance_ymd']}")
    L.append(f"Running on {on_date}:")
    rows = [[x["label"], x["path"], x["start"][:16], x["end"][:16], _h(x["lord_signifies"])] for x in ladder]
    L += _table(["Level", "Lords", "From", "To", "Last lord signifies"], rows)
    L.append("")
    L.append(f"## Ruling planets at {on_date} (birth place)")
    L.append("; ".join(f"{p['planet']} x{p['count']} ({', '.join(p['roles'])})" for p in rps["ruling_planets"]))
    f = ch.fortuna(chart)
    L.append("")
    L.append(f"Fortuna {f['dms']} {f['sign']} (house {f['house']}), star {f['star_lord']}, sub {f['sub_lord']}.")
    return "\n".join(L) + "\n"
