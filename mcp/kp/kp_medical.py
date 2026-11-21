"""KP health facts (progress.md backlog item 14). Facts only, NOT medical advice, never a diagnosis or a prognosis.

Rules used (KP teaching sources, schools differ; each line is a fact the caller can audit):
* The sub lord of the 6th cusp is the main indicator of health vulnerability; if it signifies 6, 8 or 12 the chart shows a tendency
  to illness; with the 1st house added, recurrence; 5 and 11 (12th from 6 and 8, recovery) are the recovery houses.
* Nature hints by the 6th CSL (or its star lord): Saturn = chronic / lingering, Mercury = complications, Mars = sudden / acute.
* Disease periods: joint dasha lords all signifying 6 / 8 / 12; recovery periods: all signifying 5 / 11.
* Sign-to-body-part table: the traditional Vedic list (general, not specific to Krishnamurti); the planet-to-body table is NOT included
  because no vetted source was found.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any, Optional

import kp_chart as ch
import kp_dasha as da
from kp_core import fmt_dt
from kp_tables import kp_lords

DISEASE = [6, 8, 12]
RECOVERY = [5, 11]
NATURE = {"Saturn": "chronic / lingering", "Mercury": "complications", "Mars": "sudden / acute"}
BODY = {"Aries": "head, face, brain", "Taurus": "face, neck, throat", "Gemini": "shoulders, arms, hands, lungs", "Cancer": "chest, breasts, stomach",
        "Leo": "heart, upper back, spine", "Virgo": "intestines, abdomen, digestion", "Libra": "kidneys, lower back, skin",
        "Scorpio": "genitals, reproductive and excretory organs", "Sagittarius": "hips, thighs", "Capricorn": "knees, bones, joints",
        "Aquarius": "calves, ankles, circulation", "Pisces": "feet, lymph"}
SOURCE = "Krishnamurti Reader 6 (Medical Astrology) as summarised in KP teaching pages; schools differ"


def health_facts(chart: SimpleNamespace, d: da.Dasha, start: datetime, end: datetime, include_periods: bool, limit: int,
                 levels: Optional[str]) -> dict[str, Any]:
    rep = ch.csl_report(chart, 6, levels)
    sub = rep["sub_lord"]
    star = chart.planets[sub]["star_lord"]
    sig = set(rep["sub_lord_signifies"])
    dis, rec = sorted(sig & set(DISEASE)), sorted(sig & set(RECOVERY))
    asc, c6 = kp_lords(chart.asc), kp_lords(chart.cusps[5])
    mars = set(ch.signified(chart, "Mars", levels))
    nature = [{"planet": p, "role": r, "hint": NATURE[p]} for p, r in ((sub, "6th cusp sub lord"), (star, "star lord of the 6th cusp sub lord"))
              if p in NATURE]
    out: dict[str, Any] = {
        "disclaimer": "Astrological facts only. Not medical advice, not a diagnosis, not a prognosis. See a doctor for any health matter.",
        "groups": {"disease_houses": DISEASE, "recovery_houses": RECOVERY, "recurrence_house": 1, "source": SOURCE},
        "sixth_cusp": {"sign": c6["sign"], "dms": c6["dms"], "sub_lord": sub, "star_lord_of_sub_lord": star,
                       "sub_lord_signifies": sorted(sig), "disease_houses_signified": dis, "recovery_houses_signified": rec,
                       "signifies_first_house": 1 in sig, "retrograde_check": rep["retrograde_check"]},
        "reading": {"tendency_to_illness_by_rule": bool(dis), "all_three_disease_houses": len(dis) == 3,
                    "recurrence_by_rule": bool(dis) and 1 in sig, "recovery_support_by_rule": bool(rec),
                    "rule": "6th CSL signifies 6, 8 or 12 = tendency; plus 1 = recurrence; 5 or 11 = recovery support (facts of the rule, not a diagnosis)."},
        "nature_hints": nature or [{"hint": "no Saturn / Mercury / Mars link in the 6th CSL or its star lord"}],
        "body_parts_traditional": {"6th cusp sign": {"sign": c6["sign"], "parts": BODY[c6["sign"]]},
                                   "Ascendant sign": {"sign": asc["sign"], "parts": BODY[asc["sign"]]},
                                   "note": "Traditional sign-body table (general Vedic, not KP-specific); a sign names a region, not a disease."},
        "mars_signifies": {"houses": sorted(mars), "disease_houses": sorted(mars & set(DISEASE))},
        "eighth_and_twelfth_csl": [
            {"cusp": c, "sub_lord": (r := ch.csl_report(chart, c, levels))["sub_lord"], "signifies": r["sub_lord_signifies"]} for c in (8, 12)],
    }
    if include_periods:
        dp = da.find_joint_periods(d, chart, DISEASE, RECOVERY, start, end, 3, False, 2, None, levels, limit, None, chart.opts.retrograde_rule)
        rp = da.find_joint_periods(d, chart, RECOVERY, DISEASE, start, end, 3, False, 2, None, levels, limit, None, chart.opts.retrograde_rule)
        out["window"] = [fmt_dt(start), fmt_dt(end)]
        out["disease_period_rule"] = "Dasha / Bhukti / Antara lords each signify 6, 8 or 12 and together at least two of them."
        out["disease_periods"] = {"count": dp["count"], "truncated": dp.get("truncated", False), "periods": dp["periods"]}
        out["recovery_period_rule"] = "Dasha / Bhukti / Antara lords each signify 5 or 11 and together both."
        out["recovery_periods"] = {"count": rp["count"], "truncated": rp.get("truncated", False), "periods": rp["periods"]}
    return out
