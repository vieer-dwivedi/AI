"""Cuspal interlinks (CIL), progress.md backlog item 7. Facts only; no verdict of its own.

School note: the Sub-Sub / Cuspal Interlinks theory is S.P. Khullar's (KCIL), with K. Bhaskaran's "inter-link of cuspal sub lords"
beside it. It is NOT Krishnamurti's own method. What the sources say, and what this module reports:
  * Khullar: the sub-sub lord (SSL) of a cusp indicates the promise and potential of that cusp, and the sub lord qualifies it
    (a sub lord through its star lord, e.g. Jupiter through Saturn). -> each cusp row lists the SSL and the houses it signifies.
  * Bhaskaran: a planet that is the cuspal sub lord (CSL) of a bhava has a star lord and a sub lord that are themselves CSLs of other
    bhavas; these are the links between cusps. -> each row lists the cusps the CSL, its star lord and its sub lord are CSL of.
  * Generic CIL texts: the houses the CSL signifies are the houses its cusp is linked with. -> `links_to_other_houses`.
Only these readings are encoded; the finer link kinds of Bhaskaran's theory (four kinds) are not (source pages were not legible enough).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import kp_chart as ch
from kp_tables import HOUSE_THEMES, PLANETS, kp_lords

SCHOOL = ("Cuspal Interlinks (S.P. Khullar's KCIL, K. Bhaskaran's inter-links of cuspal sub lords): later schools, not "
          "pure Krishnamurti. Facts only: the tool lists links and signified houses, it gives no verdict of its own.")


def cusp_rulership(chart: SimpleNamespace) -> dict[str, dict[str, list[int]]]:
    """For each planet: the cusps it rules at each level (sign / star / sub / sub-sub lord)."""
    out = {p: {"sign_lord_of": [], "star_lord_of": [], "sub_lord_of": [], "sub_sub_lord_of": []} for p in PLANETS}
    for i, lon in enumerate(chart.cusps, start=1):
        r = kp_lords(lon)
        out[r["sign_lord"]]["sign_lord_of"].append(i)
        out[r["star_lord"]]["star_lord_of"].append(i)
        out[r["sub_lord"]]["sub_lord_of"].append(i)
        out[r["sub_sub_lord"]]["sub_sub_lord_of"].append(i)
    return out


def interlinks(chart: SimpleNamespace, cusps: Optional[list[int]] = None, levels: Optional[str] = None,
               judged: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    cs = cusps or list(range(1, 13))
    if any((not isinstance(c, int)) or not 1 <= c <= 12 for c in cs):
        raise ValueError("cusps must be integers 1..12.")
    lv = levels or chart.opts.significator_levels
    rule = cusp_rulership(chart)
    rows = []
    for i in cs:
        r = kp_lords(chart.cusps[i - 1])
        csl, ssl = r["sub_lord"], r["sub_sub_lord"]
        star, sub2 = chart.star_of[csl], chart.planets[csl]["sub_lord"]
        csl_sig, ssl_sig = ch.signified(chart, csl, lv), ch.signified(chart, ssl, lv)
        rows.append({
            "cusp": i, "theme": HOUSE_THEMES[i], "dms": r["dms"], "sign": r["sign"], "sign_lord": r["sign_lord"],
            "star_lord": r["star_lord"], "sub_lord": csl, "sub_sub_lord": ssl, "kp_number": r["kp_number"],
            "csl": {"planet": csl, "signifies": csl_sig, "links_to_other_houses": [h for h in csl_sig if h != i],
                    "is_csl_of_cusps": rule[csl]["sub_lord_of"],
                    "star_lord": star, "star_lord_is_csl_of_cusps": rule[star]["sub_lord_of"],
                    "sub_lord": sub2, "sub_lord_is_csl_of_cusps": rule[sub2]["sub_lord_of"]},
            "ssl": {"planet": ssl, "signifies": ssl_sig, "same_as_csl": ssl == csl,
                    "is_csl_of_cusps": rule[ssl]["sub_lord_of"], "is_ssl_of_cusps": rule[ssl]["sub_sub_lord_of"]},
        })
    out: dict[str, Any] = {"school": SCHOOL, "levels_used": lv, "cusps": rows, "cusp_rulership": rule,
                           "note": "Natal structure only; dasha periods activate it. Birth-time errors change sub and sub-sub lords quickly."}
    if judged is not None:
        fav, den, pc = judged["favourable_houses"], judged["denial_houses"], judged["primary_cusp"]
        row = next((x for x in rows if x["cusp"] == pc), None)
        if row is None:  # primary cusp not among the requested ones: compute its lords directly
            r = kp_lords(chart.cusps[pc - 1])
            ssl_sig = ch.signified(chart, r["sub_sub_lord"], lv)
            ssl_planet = r["sub_sub_lord"]
        else:
            ssl_sig, ssl_planet = row["ssl"]["signifies"], row["ssl"]["planet"]
        fh = sorted(set(ssl_sig) & set(fav))
        dh = sorted(set(ssl_sig) & set(den) - set(fav))
        v, why = ch._verdict(fh, dh, fav)
        out["event_reading"] = {
            "event": judged["event"], "primary_cusp": pc, "favourable_houses": fav, "denial_houses": den,
            "csl_verdict": judged["verdict"], "csl_favourable": judged["favourable_signified_by_csl"], "csl_denial": judged["denial_signified_by_csl"],
            "ssl": ssl_planet, "ssl_favourable": fh, "ssl_denial": dh, "ssl_verdict": v, "ssl_reason": why.replace("CSL", "SSL"),
            "ssl_agrees_with_csl": v == judged["verdict"],
            "note": ("Khullar's school reads the sub-sub lord as the promise and potential of the cusp; the same CSL-style house test "
                     "is applied to it here. The standard KP verdict stays the CSL's."),
        }
    return out
