"""KP Vimshottari dasha: Dasha -> Bhukti -> Antara -> Sookshma -> Prana, significations and joint-period search.

  * Year length defaults to 365.25 days (KP software convention); configurable in `KPOptions.dasha_year_days`.
  * Sub-period length = parent length x child years / 120, starting from the parent's own lord.
  * A planet "qualifies" for an event when it signifies at least one favourable house; the joint period of Dasha, Bhukti and
    Antara (optionally Sookshma) lords qualifies when every lord qualifies (KP rule: event in the joint period of significators).
Dates are local clock time at the birth place.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Iterator, Optional

from kp_chart import signified
from kp_core import fmt_dt
from kp_retro import period_flags
from kp_tables import NAK_ARC, VIM_ORDER, VIM_TOTAL, VIM_YEARS, planet_name

LEVELS = ["mahadasha", "bhukti", "antara", "sookshma", "prana"]
LEVEL_LABELS = {1: "Dasha (Mahadasha)", 2: "Bhukti (Antardasha)", 3: "Antara (Pratyantardasha)", 4: "Sookshma", 5: "Prana"}
MAX_ROWS = 2500


class DashaError(ValueError):
    """Caller-fixable dasha problem."""


class Dasha:
    """Vimshottari periods from the Moon's longitude at birth."""

    def __init__(self, birth_dt: datetime, moon_lon: float, year_days: float = 365.25, max_years: float = 125.0):
        self.birth = birth_dt
        self.year_days = year_days
        lon = moon_lon % 360.0
        nak = min(int(lon * 3600.0 // NAK_ARC), 26)
        self.start_lord = VIM_ORDER[nak % 9]
        self.elapsed = (lon * 3600.0 - nak * NAK_ARC) / NAK_ARC  # fraction of the first nakshatra already passed
        first_days = VIM_YEARS[self.start_lord] * year_days
        t = birth_dt - timedelta(days=self.elapsed * first_days)
        self.mds: list[dict[str, Any]] = []
        i0 = VIM_ORDER.index(self.start_lord)
        k = 0
        while (t - birth_dt).total_seconds() / 86400.0 / year_days < max_years:
            lord = VIM_ORDER[(i0 + k) % 9]
            days = VIM_YEARS[lord] * year_days
            end = t + timedelta(days=days)
            self.mds.append({"lords": [lord], "start": t, "end": end, "days": days})
            t = end
            k += 1

    # ---- structure ----------------------------------------------------------------------
    def balance_at_birth(self) -> dict[str, Any]:
        yrs = VIM_YEARS[self.start_lord] * (1.0 - self.elapsed)
        y = int(yrs)
        m = int((yrs - y) * 12)
        d = int(round(((yrs - y) * 12 - m) * 30))
        return {"lord": self.start_lord, "fraction_elapsed_at_birth": round(self.elapsed, 6),
                "balance_years": round(yrs, 6), "balance_ymd": f"{y}y {m}m {d}d"}

    def children(self, period: dict[str, Any]) -> list[dict[str, Any]]:
        lord = period["lords"][-1]
        i0 = VIM_ORDER.index(lord)
        out = []
        cum = 0.0
        for k in range(9):
            c = VIM_ORDER[(i0 + k) % 9]
            days = period["days"] * VIM_YEARS[c] / VIM_TOTAL
            start = period["start"] + timedelta(days=cum)
            cum += days
            end = period["end"] if k == 8 else period["start"] + timedelta(days=cum)
            out.append({"lords": period["lords"] + [c], "start": start, "end": end, "days": days})
        return out

    def find(self, lords: list[str]) -> dict[str, Any]:
        """Period for a lord path such as ['Rahu', 'Jupiter'] (first matching Dasha that ends after birth)."""
        path = [planet_name(x) for x in lords]
        cur = next((m for m in self.mds if m["lords"][0] == path[0] and m["end"] > self.birth), None)
        if cur is None:
            raise DashaError(f"No {path[0]} Mahadasha found in this life span.")
        for lord in path[1:]:
            cur = next((c for c in self.children(cur) if c["lords"][-1] == lord), None)
            if cur is None:
                raise DashaError(f"Lord path {path} is invalid.")
        return cur

    def _deeper(self, node: dict[str, Any], level: int, start: datetime, end: datetime) -> Iterator[dict[str, Any]]:
        cur_level = len(node["lords"])
        if cur_level == level:
            yield node
            return
        for c in self.children(node):
            if c["end"] > start and c["start"] < end:
                yield from self._deeper(c, level, start, end)


def make_dasha(chart: SimpleNamespace, birth_dt: datetime) -> Dasha:
    return Dasha(birth_dt, chart.planets["Moon"]["longitude"], chart.opts.dasha_year_days)


def row(p: dict[str, Any], chart: Optional[SimpleNamespace] = None, with_sig: bool = False) -> dict[str, Any]:
    lvl = len(p["lords"])
    r = {"level": LEVELS[lvl - 1], "label": LEVEL_LABELS[lvl], "path": "/".join(p["lords"]), "lords": p["lords"],
         "start": fmt_dt(p["start"]), "end": fmt_dt(p["end"]), "days": round(p["days"], 3)}
    if with_sig and chart is not None:
        r["lord_signifies"] = signified(chart, p["lords"][-1])
    return r


def periods_table(d: Dasha, chart: SimpleNamespace, level: int, start: datetime, end: datetime,
                  parent: Optional[list[str]] = None, with_sig: bool = True) -> list[dict[str, Any]]:
    if not 1 <= level <= 5:
        raise DashaError("level must be 1 (Dasha) .. 5 (Prana).")
    par = d.find(parent) if parent else None
    if par is not None and len(par["lords"]) >= level:
        raise DashaError("`level` must be deeper than the parent path.")
    out = []
    gen = d._deeper(par, level, start, end) if par is not None else (
        x for m in d.mds if m["end"] > start and m["start"] < end for x in d._deeper(m, level, start, end))
    for p in gen:
        out.append(row(p, chart, with_sig))
        if len(out) > MAX_ROWS:
            raise DashaError(f"More than {MAX_ROWS} periods: narrow the window or use a shallower level.")
    return out


def running_ladder(d: Dasha, chart: SimpleNamespace, at: datetime, depth: int = 4) -> list[dict[str, Any]]:
    cur = next((m for m in d.mds if m["start"] <= at < m["end"]), None)
    if cur is None:
        raise DashaError("The date is outside the computed dasha span (birth .. about 125 years).")
    ladder = [cur]
    for _ in range(depth - 1):
        kids = d.children(cur)
        cur = next((c for c in kids if c["start"] <= at < c["end"]), kids[-1])
        ladder.append(cur)
    out = []
    for p in ladder:
        r = row(p, chart, True)
        r["remaining_days"] = round((p["end"] - at).total_seconds() / 86400.0, 2)
        r["elapsed_percent"] = round(100.0 * (at - p["start"]).total_seconds() / (p["end"] - p["start"]).total_seconds(), 2)
        out.append(r)
    return out


# --------------------------------------------------------------------------------------
# Joint-period (event timing) search
# --------------------------------------------------------------------------------------
def lord_status(chart: SimpleNamespace, lord: str, fav: set[int], den: set[int], levels: Optional[str] = None) -> dict[str, Any]:
    s = set(signified(chart, lord, levels))
    sub = chart.planets[lord]["sub_lord"]
    ss = set(signified(chart, sub, levels))
    return {"lord": lord, "signifies": sorted(s), "favourable": sorted(s & fav), "denial": sorted((s & den) - fav),
            "sub_lord": sub, "sub_lord_favourable": sorted(ss & fav), "sub_lord_denial": sorted((ss & den) - fav)}


def find_joint_periods(d: Dasha, chart: SimpleNamespace, fav: list[int], den: list[int], start: datetime, end: datetime,
                       depth: int = 3, exclude_denial: bool = False, min_coverage: Optional[int] = None,
                       rp: Optional[list[str]] = None, levels: Optional[str] = None, limit: int = 400,
                       allowed: Optional[set[str]] = None, retro_rule: str = "report_only") -> dict[str, Any]:
    """Periods inside [start, end] in which every lord down to `depth` signifies a favourable house."""
    if not 2 <= depth <= 5:
        raise DashaError("depth must be 2 (Dasha+Bhukti) .. 5.")
    fav_s, den_s = set(fav), set(den) - set(fav)
    need = min_coverage if min_coverage is not None else min(2, len(fav_s))
    cache: dict[str, dict[str, Any]] = {}

    def st(lord: str) -> dict[str, Any]:
        if lord not in cache:
            cache[lord] = lord_status(chart, lord, fav_s, den_s, levels)
        return cache[lord]

    def ok(lord: str) -> bool:
        s = st(lord)
        if allowed is not None and lord not in allowed:
            return False
        return bool(s["favourable"]) and not (exclude_denial and s["denial"])

    rp_set = set(rp or [])
    results: list[dict[str, Any]] = []
    truncated = False

    def walk(node: dict[str, Any]) -> None:
        nonlocal truncated
        if truncated or not ok(node["lords"][-1]):
            return
        if len(node["lords"]) == depth:
            cover = sorted(set().union(*[set(st(x)["favourable"]) for x in node["lords"]]))
            if len(cover) >= need:
                r = row(node)
                r["coverage_houses"] = cover
                r["coverage_count"] = len(cover)
                r["denial_touched"] = sorted(set().union(*[set(st(x)["denial"]) for x in node["lords"]]))
                pf = period_flags(chart, node["lords"], retro_rule)
                if pf["retrograde_flags"]:
                    r.update(pf)
                if rp_set:
                    r["ruling_planet_lords"] = [x for x in node["lords"] if x in rp_set]
                results.append(r)
                if len(results) >= limit:
                    truncated = True
            return
        for c in d.children(node):
            if c["end"] > start and c["start"] < end:
                walk(c)

    for m in d.mds:
        if m["end"] > start and m["start"] < end:
            walk(m)
    return {"depth": depth, "levels": [LEVEL_LABELS[i] for i in range(1, depth + 1)], "favourable_houses": sorted(fav_s),
            "denial_houses": sorted(den_s), "min_coverage": need, "exclude_denial": exclude_denial,
            "fruitful_filter": sorted(allowed) if allowed is not None else None, "retrograde_rule": retro_rule,
            "lord_status": {k: v for k, v in sorted(cache.items())}, "count": len(results), "truncated": truncated,
            "periods": results,
            "rule": ("KP: an event falls in the joint period of Dasha, Bhukti and Antara lords that signify the event houses; "
                     "the cuspal sub lord must promise it first, and ruling planets / transits then narrow the window.")}
