"""Pure KP tables and arithmetic: no swisseph, no MCP, no I/O.

Everything here is deterministic and testable offline.

  * Zodiac division: sign -> star (nakshatra) -> sub -> sub-sub -> sookshma, all proportional to Vimshottari years.
  * The 249 numbered KP sub-divisions (243 subs, 6 of them split by a sign boundary), built from integer arc-seconds.
  * House-group (event) table: favourable and denial houses per matter (textbook KP convention; schools differ).

Planet names used in the API: Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn, Rahu, Ketu.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Any, Optional

SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio",
         "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
SIGN_LORDS = ["Mars", "Venus", "Mercury", "Moon", "Sun", "Mercury", "Venus", "Mars",
              "Jupiter", "Saturn", "Saturn", "Jupiter"]
PLANETS = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]
NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra", "Punarvasu", "Pushya",
    "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni", "Hasta", "Chitra", "Swati",
    "Vishakha", "Anuradha", "Jyeshtha", "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana",
    "Dhanishta", "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati",
]
VIM_ORDER = ["Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury"]
VIM_YEARS = {"Ketu": 7, "Venus": 20, "Sun": 6, "Moon": 10, "Mars": 7, "Rahu": 18,
             "Jupiter": 16, "Saturn": 19, "Mercury": 17}
VIM_TOTAL = 120

# Day lord by Python weekday() (Monday = 0).
DAY_LORD_BY_WEEKDAY = ["Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Sun"]

NAK_ARC = 48000          # arc-seconds in one nakshatra (13d20m)
SIGN_ARC = 108000        # arc-seconds in one sign (30d)
FULL_ARC = 1296000       # arc-seconds in the circle

HOUSE_THEMES = {
    1: "self, body, personality, start of things",
    2: "family, speech, accumulated wealth, income",
    3: "siblings, courage, short journeys, communication, writing",
    4: "home, mother, vehicles, property, basic education, peace of mind",
    5: "children, intelligence, speculation, romance, creativity",
    6: "debts, disease, enemies, service, litigation, daily work",
    7: "spouse, partnerships, marriage, business partners, open enemies",
    8: "longevity, obstacles, sudden events, legacy, death, others' money",
    9: "fortune, father, long journeys, higher education, dharma",
    10: "profession, status, authority, karma, government",
    11: "gains, fulfilment of desires, friends, elder siblings, income",
    12: "loss, expenses, isolation, foreign lands, hospitalisation, moksha",
}


# --------------------------------------------------------------------------------------
# 243 subs (integer arc-seconds) and the 249 numbered divisions
# --------------------------------------------------------------------------------------
def _build_subs() -> list[tuple[int, int, int, str, str]]:
    """[(start_arc, end_arc, nakshatra_index, star_lord, sub_lord)] for all 243 subs."""
    out = []
    for n in range(27):
        star = VIM_ORDER[n % 9]
        i0 = VIM_ORDER.index(star)
        pos = n * NAK_ARC
        for k in range(9):
            sub = VIM_ORDER[(i0 + k) % 9]
            span = 400 * VIM_YEARS[sub]  # 48000 * years / 120
            out.append((pos, pos + span, n, star, sub))
            pos += span
    return out


SUBS = _build_subs()
_SUB_STARTS = [s[0] for s in SUBS]


def _build_numbered() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for a, b, n, star, sub in SUBS:
        cuts = [a]
        k = a // SIGN_ARC + 1
        while k * SIGN_ARC < b:
            cuts.append(k * SIGN_ARC)
            k += 1
        cuts.append(b)
        for x, y in zip(cuts, cuts[1:]):
            sign = x // SIGN_ARC
            out.append({"start_arc": x, "end_arc": y, "sign_index": sign, "nakshatra_index": n,
                        "star_lord": star, "sub_lord": sub, "split_by_sign": len(cuts) > 2})
    for i, row in enumerate(out, start=1):
        row["number"] = i
    return out


NUMBERED = _build_numbered()
_NUM_STARTS = [r["start_arc"] for r in NUMBERED]


def dms(deg: float) -> str:
    """12.5 -> '12°30′00″' (rounded to the second)."""
    total = int(round(deg * 3600.0))
    d, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{d}°{m:02d}′{s:02d}″"


def arc_to_deg(arc: int) -> float:
    return arc / 3600.0


def division_row(number: int) -> dict[str, Any]:
    """Public row for KP number 1..249."""
    if not isinstance(number, int) or isinstance(number, bool) or not 1 <= number <= 249:
        raise ValueError("KP horary number must be an integer from 1 to 249.")
    r = NUMBERED[number - 1]
    return {
        "number": number,
        "start": round(arc_to_deg(r["start_arc"]), 6), "end": round(arc_to_deg(r["end_arc"]), 6),
        "start_dms": dms(arc_to_deg(r["start_arc"])), "end_dms": dms(arc_to_deg(r["end_arc"])),
        "sign": SIGNS[r["sign_index"]], "sign_index": r["sign_index"], "sign_lord": SIGN_LORDS[r["sign_index"]],
        "nakshatra": NAKSHATRAS[r["nakshatra_index"]], "star_lord": r["star_lord"], "sub_lord": r["sub_lord"],
        "split_by_sign": r["split_by_sign"],
    }


def division_number(lon: float) -> int:
    """KP number (1..249) containing a sidereal longitude."""
    arc = int((lon % 360.0) * 3600.0)
    return bisect_right(_NUM_STARTS, min(arc, FULL_ARC - 1)) 


def kp_lords(lon: float) -> dict[str, Any]:
    """Sign / star / sub / sub-sub / sookshma lords of a sidereal longitude (degrees)."""
    lon = lon % 360.0
    arc = lon * 3600.0
    iarc = min(int(arc), FULL_ARC - 1)
    sign = iarc // SIGN_ARC
    idx = bisect_right(_SUB_STARTS, iarc) - 1
    start, end, nak, star, sub = SUBS[idx]
    # sub-sub and sookshma: same proportions, starting from the parent lord
    frac = (arc - start) / (end - start)
    frac = min(max(frac, 0.0), 1.0 - 1e-12)
    chain = []
    parent = sub
    for _ in range(2):
        i0 = VIM_ORDER.index(parent)
        acc = 0.0
        chosen = None
        for k in range(9):
            lord = VIM_ORDER[(i0 + k) % 9]
            w = VIM_YEARS[lord] / VIM_TOTAL
            if frac < acc + w or k == 8:
                chosen = lord
                frac = (frac - acc) / w
                frac = min(max(frac, 0.0), 1.0 - 1e-12)
                break
            acc += w
        chain.append(chosen)
        parent = chosen
    return {
        "longitude": round(lon, 6), "dms": dms(lon),
        "sign": SIGNS[sign], "sign_index": sign, "sign_lord": SIGN_LORDS[sign],
        "degree_in_sign": round(lon - sign * 30.0, 6),
        "nakshatra": NAKSHATRAS[nak], "nakshatra_index": nak + 1,
        "pada": int(((arc - nak * NAK_ARC) // (NAK_ARC / 4)) % 4) + 1,
        "star_lord": star, "sub_lord": sub, "sub_sub_lord": chain[0], "sookshma_lord": chain[1],
        "kp_number": division_number(lon),
    }


def sign_index_of(lon: float) -> int:
    return int((lon % 360.0) // 30.0)


def owned_signs(planet: str) -> list[str]:
    return [SIGNS[i] for i, lord in enumerate(SIGN_LORDS) if lord == planet]


def badhaka_house(lagna_sign_index: int) -> int:
    """Badhaka house: movable signs -> 11th, fixed -> 9th, dual -> 7th (classical rule used in KP texts)."""
    kind = lagna_sign_index % 3  # 0 movable, 1 fixed, 2 dual
    return (11, 9, 7)[kind]


# --------------------------------------------------------------------------------------
# Event / house-group table (textbook KP convention; sources and schools differ)
# --------------------------------------------------------------------------------------
def _e(label: str, cusp: int, fav: list[int], denial: list[int], note: str = "") -> dict[str, Any]:
    return {"label": label, "primary_cusp": cusp, "favourable": fav, "denial": denial, "note": note}


EVENTS: dict[str, dict[str, Any]] = {
    "marriage": _e("Marriage", 7, [2, 7, 11], [1, 6, 10], "12 also points to separation or a foreign spouse."),
    "engagement": _e("Marriage engagement", 7, [3, 9, 11], [1, 6, 10], "Reader 3/4 grouping for engagement."),
    "love_marriage": _e("Love marriage", 5, [5, 7, 11], [1, 6, 10], "Judge the 5th cusp sub lord, then the 7th."),
    "second_marriage": _e("Second marriage", 9, [7, 9, 11], [1, 6, 10], "9th is the 3rd from the 7th."),
    "divorce": _e("Separation or divorce", 7, [1, 6, 10], [2, 7, 11], "1, 6, 10 are the 12th from 2, 7, 11. Also read 12."),
    "child_birth": _e("Birth of a child", 5, [2, 5, 11], [1, 4, 10], "1, 4, 10 are the 12th from 2, 5, 11."),
    "pregnancy": _e("Pregnancy", 5, [2, 5, 11], [1, 4, 10]),
    "second_child": _e("Second child", 7, [2, 7, 11], [4, 10]),
    "third_child": _e("Third child", 9, [2, 9, 11], [4, 10]),
    "education": _e("Basic education", 4, [4, 9, 11], [3, 8, 12], "3 and 8 show completion / negation of studies."),
    "higher_education": _e("Higher education", 9, [4, 9, 11], [3, 8, 12], "9th primary; 4th for the study itself."),
    "service": _e("Service / employment", 10, [2, 6, 10, 11], [1, 5, 9], "1, 5, 9 are the 12th from 2, 6, 10."),
    "government_job": _e("Government job", 10, [2, 6, 10, 11], [1, 5, 9], "Check a strong Sun in the chain."),
    "business": _e("Own business", 10, [2, 7, 10, 11], [6], "Most texts: avoid 6 (service)."),
    "promotion": _e("Promotion / rise in career", 10, [2, 6, 10, 11], [5, 9, 12]),
    "change_of_job": _e("Change of job or profession", 10, [1, 5, 9], [],
                        "1, 5, 9 end the current job (12th from 2, 6, 10); a new job still needs 2, 6, 10, 11."),
    "gain_of_money": _e("Gain of money", 2, [2, 6, 11], [5, 8, 12]),
    "speculation": _e("Gain from speculation", 5, [2, 5, 6, 11], [8, 12]),
    "loan": _e("Obtaining a loan", 6, [2, 6, 11], [5, 12]),
    "name_and_fame": _e("Status and respect", 10, [1, 10, 11], [12]),
    "foreign_travel": _e("Foreign travel / long journey", 9, [3, 9, 12], [4], "3 away from home (12th from 4), 9 long journey, 12 unfamiliar place."),
    "settle_abroad": _e("Settling abroad", 12, [9, 12], [4], "Schools vary; some add 3 and 7."),
    "return_home": _e("Returning home", 4, [3, 9, 11], [12]),
    "property_purchase": _e("Purchase of property / vehicle", 4, [4, 11, 12], [3, 5, 10]),
    "property_sale": _e("Sale of property", 4, [3, 5, 10], [4, 11, 12], "Reader 6 example uses the 3, 5, 10 group."),
    "cure_of_disease": _e("Cure / recovery from disease", 6, [1, 5, 11], [6, 8, 12]),
    "disease": _e("Proneness to disease", 6, [1, 6, 8, 12], []),
    "accident": _e("Accident", 8, [1, 8, 12], [], "Reader grouping 1 / 8, 12."),
    "longevity": _e("Long life", 8, [1, 5, 8, 11], ["badhaka", 2, 7], "Short life: 2, 7 and the badhaka house."),
    "litigation_success": _e("Success in litigation", 6, [1, 6, 11], [5, 8, 12]),
    "competition_success": _e("Success in competition", 6, [1, 6, 11], [5, 8, 12]),
    "partnership": _e("Business partnership", 7, [5, 7, 11], [6, 12]),
    "spiritual_progress": _e("Success in spiritual discipline", 9, [6, 9, 11], [], "Often read with 12."),
}


def event_info(name: str) -> dict[str, Any]:
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    if key not in EVENTS:
        raise ValueError(f"Unknown event '{name}'. Known events: {sorted(EVENTS)}. "
                         "Or pass your own houses with `custom_favourable` / `custom_denial`.")
    return {"key": key, **EVENTS[key]}


# --------------------------------------------------------------------------------------
# Name helpers
# --------------------------------------------------------------------------------------
_ALIASES = {
    "sun": "Sun", "surya": "Sun", "ravi": "Sun", "moon": "Moon", "chandra": "Moon", "mars": "Mars",
    "mangal": "Mars", "kuja": "Mars", "mercury": "Mercury", "budha": "Mercury", "jupiter": "Jupiter",
    "guru": "Jupiter", "venus": "Venus", "shukra": "Venus", "saturn": "Saturn", "shani": "Saturn",
    "rahu": "Rahu", "ketu": "Ketu",
}


def planet_name(value: Any) -> str:
    text = str(value).strip().lower()
    if text in _ALIASES:
        return _ALIASES[text]
    raise ValueError(f"Unknown planet '{value}'. Use one of {PLANETS} (or Sanskrit names like Shani, Guru).")


def sign_name_index(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 11:
        return value
    text = str(value).strip().lower()
    for i, s in enumerate(SIGNS):
        if s.lower() == text or (len(text) >= 3 and s.lower().startswith(text)):
            return i
    raise ValueError(f"Unknown sign '{value}'.")
