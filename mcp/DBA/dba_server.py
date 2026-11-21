"""DBA MCP server (Dasha-Bhukti-Antara timing). Facts and stated rules only, no predictions.

Run as a script (NOT `python -m`):  python mcp/DBA/main.py --http   (port 8003)
"""
from __future__ import annotations

import functools
import os
from datetime import datetime, timedelta
from fractions import Fraction
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from mcp.server.fastmcp import FastMCP

import dba_algebra as alg
import dba_astro as astro
import dba_bands as bands
import dba_resolution as res
import dba_solararc as solar
import dba_validate as val
import dba_vimshottari as vim
from dba_core import to_fraction
from dba_time import to_utc

ENGINE_VERSION = "0.1"
mcp = FastMCP("dba")

BIRTH_HELP = "Local birth date/time YYYY-MM-DDTHH:MM[:SS]."
RULE = {"rule_id": "VIM-BPHS46.12-16", "school": "Parashari",
        "source": "BPHS ch.46 (Sanskrit numbering) v.12-16, Vimshottari; sub-periods proportional (parent x child/120)"}


def tool(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*a: Any, **k: Any) -> Any:
        try:
            return fn(*a, **k)
        except (ValueError, NotImplementedError, KeyError) as exc:
            raise ValueError(str(exc)) from exc
    return mcp.tool()(wrapper)


# ---------- helpers ----------
def _parse(s: str) -> datetime:
    return datetime.fromisoformat(s.strip().replace("Z", ""))


def _utc(local_s, timezone, longitude, ambiguous, nonexistent):
    return to_utc(_parse(local_s), tz_name=timezone, longitude_deg=longitude,
                  ambiguous=ambiguous, nonexistent=nonexistent)


def _days(a: datetime, b: datetime) -> Fraction:
    d = b - a
    return Fraction(d.days) + Fraction(d.seconds, 86400) + Fraction(d.microseconds, 86400 * 10**6)


class Chart:
    def __init__(self, birth, timezone, longitude, ayanamsa, year_days, ambiguous, nonexistent):
        self.tz, self.lon = timezone, longitude
        self.ambiguous, self.nonexistent = ambiguous, nonexistent
        self.timeinfo = _utc(birth, timezone, longitude, ambiguous, nonexistent)
        self.utc = self.timeinfo["utc_used"]
        self.moon = astro.moon(self.utc, ayanamsa)
        self.moon_deg = Fraction(self.moon["longitude_deg"])
        self.year_days = year_days
        self.yd = solar.resolve_year_mode(year_days, self.utc)   # Fraction, or SunArcMap for true_solar_arc*
        self.ayanamsa = self.moon["ayanamsa"]

    def t_of(self, local_s):
        return _days(self.utc, _utc(local_s, self.tz, self.lon, self.ambiguous, self.nonexistent)["utc_used"])

    def when(self, t):
        u = self.utc + timedelta(seconds=int((to_fraction(t) * 86400 + Fraction(1, 2)) // 1))
        if self.tz:
            loc = u.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo(self.tz)).replace(tzinfo=None)
        else:
            loc = u + timedelta(hours=self.timeinfo["offset_hours"])
        return {"utc": u.isoformat(timespec="seconds"), "local": loc.isoformat(timespec="seconds")}

    def period(self, p):
        s, e = self.when(p.start), self.when(p.end)
        return {"path": list(p.path), "level": vim.LEVEL_NAMES[p.level - 1],
                "start_utc": s["utc"], "end_utc": e["utc"], "start_local": s["local"], "end_local": e["local"],
                "duration_days": float(p.duration)}

    def provenance(self, **extra):
        out = dict(RULE)
        out.update({"engine_version": ENGINE_VERSION,
                    "options_used": {"ayanamsa": self.ayanamsa, "year_days": float(vim.nominal_year_days(self.yd)),
                                     "year_mode": str(self.year_days), "timezone": self.tz or "LMT",
                                     "tz_rule_used": self.timeinfo["tz_rule_used"],
                                     "utc_used": self.utc.isoformat(timespec="seconds")},
                    "warnings": self.timeinfo["warnings"]})
        out.update(extra)
        return out


def _chart(birth, timezone, longitude, ayanamsa, year_days, ambiguous, nonexistent):
    if timezone is None and longitude is None:
        raise ValueError("give timezone (IANA name) or longitude (local mean time)")
    if timezone is not None:
        longitude = None
    return Chart(birth, timezone, longitude, ayanamsa, year_days, ambiguous, nonexistent)


# ---------- tools ----------
@tool
def dba_year_convention_table() -> dict[str, Any]:
    """Year-length conventions and what each does to a Vimshottari timeline (days per dasha 'year', shift of a 120-year cycle vs 365.25). No birth data."""
    base = vim.year_days_from("julian")
    rows = [{"option": k, "days_per_year": float(vim.year_days_from(k)),
             "shift_over_120_years_days_vs_julian": float(120 * (vim.year_days_from(k) - base))}
            for k in vim.YEAR_ALIASES]
    return {"default": vim.DEFAULT_YEAR_MODE, "options": rows, "also_accepted": "fixed:<number> or a plain number",
            "birth_dependent_options": {
                "true_solar_arc": "Sun-return year, tropical longitude (PyJHora default); days per year vary with the Sun's speed, sub-periods proportional in Sun arc",
                "true_solar_arc_sidereal": "same with sidereal (Lahiri) longitude"}, **{"source": RULE["source"]}}


@tool
def dba_get_ladder(birth: str, timezone: Optional[str] = None, longitude: Optional[float] = None,
                   at: Optional[str] = None, depth: int = 6, ayanamsa: str = "LAHIRI",
                   year_days: str = "julian", ambiguous: str = "earlier", nonexistent: str = "shift_forward") -> dict[str, Any]:
    """Running Vimshottari period at each level (Maha..Deha, depth 1-6) at local date/time `at` (default: birth). Boundary belongs to the new period. Birth: local YYYY-MM-DDTHH:MM, plus IANA timezone or longitude (local mean time)."""
    c = _chart(birth, timezone, longitude, ayanamsa, year_days, ambiguous, nonexistent)
    t = c.t_of(at) if at else Fraction(0)
    lad = vim.ladder_at(c.moon_deg, t, c.yd, depth)
    return {"moon": {k: c.moon[k] for k in ("longitude_deg", "nakshatra_index", "maha_lord", "fraction_elapsed")},
            "at": c.when(t), "ladder": [c.period(p) for p in lad],
            "next_change": [c.when(x) for x in vim.next_changes(lad)], "provenance": c.provenance()}


@tool
def dba_get_boundaries(birth: str, timezone: Optional[str] = None, longitude: Optional[float] = None,
                       parent_path: Optional[list[str]] = None, ayanamsa: str = "LAHIRI", year_days: str = "julian",
                       maha_count: int = 9, ambiguous: str = "earlier", nonexistent: str = "shift_forward") -> dict[str, Any]:
    """Mahadasha list (parent_path empty) or the nine sub-periods of the period named by parent_path, e.g. ["Venus","Sun"] = Sun antardasha inside Venus Maha. Exact boundaries."""
    c = _chart(birth, timezone, longitude, ayanamsa, year_days, ambiguous, nonexistent)
    mahas = vim.mahadashas(c.moon_deg, c.yd, count=max(9, maha_count))
    if not parent_path:
        return {"periods": [c.period(p) for p in mahas[:maha_count]], "truncated": False,
                "provenance": c.provenance()}
    for p in mahas:
        if p.lord == parent_path[0]:
            cur = p
            break
    else:
        raise ValueError("unknown lord %r" % parent_path[0])
    if len(parent_path) > len(vim.LEVEL_NAMES) - 1:
        raise ValueError("parent_path too deep")
    for lord in parent_path[1:]:
        cur = next((x for x in vim.children(cur) if x.lord == lord), None)
        if cur is None:
            raise ValueError("bad path")
    return {"parent": c.period(cur), "periods": [c.period(x) for x in vim.children(cur)], "provenance": c.provenance()}


@tool
def dba_find_periods(birth: str, timezone: Optional[str] = None, longitude: Optional[float] = None,
                     lords_by_level: Optional[list[list[str]]] = None, start: Optional[str] = None,
                     end: Optional[str] = None, ayanamsa: str = "LAHIRI", year_days: str = "julian",
                     max_results: int = 500, ambiguous: str = "earlier", nonexistent: str = "shift_forward") -> dict[str, Any]:
    """Periods in [start, end) whose lord at each level is in the given set, e.g. lords_by_level=[["Venus","Moon"],["Sun"]] = Maha lord Venus/Moon AND Antar lord Sun. Returns merged windows, exact `coverage` (share of the searched range marked) and `truncated`. Default range: birth to birth + 120 years."""
    if not lords_by_level:
        raise ValueError("lords_by_level required, e.g. [[\"Venus\"],[\"Sun\",\"Moon\"]]")
    c = _chart(birth, timezone, longitude, ayanamsa, year_days, ambiguous, nonexistent)
    lo = c.t_of(start) if start else Fraction(0)
    hi = c.t_of(end) if end else 120 * vim.nominal_year_days(c.yd)
    level = len(lords_by_level)
    sets = [set(s) for s in lords_by_level]
    r = alg.find_periods(c.moon_deg, lambda p: all(p.path[i] in sets[i] for i in range(level)), level, lo, hi,
                         c.yd, max_results)
    clipped = [alg.Window(max(p.start, lo), min(p.end, hi), (p.path,)) for p in r["periods"]]
    wins = alg.merge_adjacent(clipped)
    cov = alg.coverage_fraction(wins, lo, hi)
    return {"range": {"start": c.when(lo), "end": c.when(hi)}, "coverage": float(cov), "coverage_exact": str(cov),
            "n_periods": len(r["periods"]), "truncated": r["truncated"],
            "windows": [{"start": c.when(w.start), "end": c.when(w.end), "days": float(w.duration)} for w in wins],
            "provenance": c.provenance(note="coverage is the specificity of the rule: share of the range it marks")}


@tool
def dba_resolution_report(birth: str, timezone: Optional[str] = None, longitude: Optional[float] = None,
                          at: Optional[str] = None, birth_sigma_min: float = 0, ayanamsas: Optional[list[str]] = None,
                          year_modes: Optional[list[str]] = None, depth: int = 6, ayanamsa: str = "LAHIRI",
                          year_days: str = "julian", ambiguous: str = "earlier",
                          nonexistent: str = "shift_forward") -> dict[str, Any]:
    """How far birth-time error (minutes, +-sigma), year-length options and ayanamsa options move the dates; flags levels shorter than that as below_resolution; warns when the Moon is near a nakshatra boundary (first Maha lord could change)."""
    c = _chart(birth, timezone, longitude, ayanamsa, year_days, ambiguous, nonexistent)
    t = c.t_of(at) if at else Fraction(0)
    alts = [Fraction(astro.moon(c.utc, a)["longitude_deg"]) for a in (ayanamsas or []) if a.upper() != c.ayanamsa]
    modes = [solar.resolve_year_mode(m, c.utc) for m in (year_modes or [year_days])]
    rep = res.resolution_report(c.moon_deg, c.moon["speed_deg_per_day"], t, birth_sigma_min, modes, alts, depth)
    rep["at"] = c.when(t)
    rep["provenance"] = c.provenance()
    return rep


@tool
def dba_ayanamsa_shift_table(birth: str, timezone: Optional[str] = None, longitude: Optional[float] = None,
                             year_days: str = "julian", ambiguous: str = "earlier",
                             nonexistent: str = "shift_forward") -> dict[str, Any]:
    """Per ayanamsa option (LAHIRI, KP_NEW, KP_OLD, RAMAN, FAGAN_BRADLEY, TRUE_CITRA): Moon longitude, first Maha lord and Maha boundary shift in days versus Lahiri."""
    c = _chart(birth, timezone, longitude, "LAHIRI", year_days, ambiguous, nonexistent)
    return {"rows": astro.ayanamsa_shift_table(c.utc, c.yd), "provenance": c.provenance()}


@tool
def dba_rahu_ketu(birth: str, timezone: Optional[str] = None, longitude: Optional[float] = None,
                  node: str = "mean", ayanamsa: str = "LAHIRI", ambiguous: str = "earlier",
                  nonexistent: str = "shift_forward") -> dict[str, Any]:
    """Sidereal Rahu/Ketu longitude for mean or true node."""
    c = _chart(birth, timezone, longitude, ayanamsa, "julian", ambiguous, nonexistent)
    return {**astro.rahu_ketu(c.utc, ayanamsa, node), "provenance": c.provenance()}


# ---------- validation tools (Phase 4) ----------
_DEFAULT_REGISTRY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation", "prereg.json")

RULESET_HELP = ("ruleset = {event_type, rules:[{level, constraints:{'1':[lords],'2':[lords]}, mode:'all'|'any', label}], "
                "combine:'union'|'intersect', ayanamsa, year_days, age_years:[lo,hi], min_precision:'day'|'month'|'year', "
                "split_seed, tune_fraction, null:'coverage'|'age_density', age_bandwidth_years, n_perm, perm_seed}")


def _help(fn):
    fn.__doc__ = (fn.__doc__ or "") + " " + RULESET_HELP
    return fn


def _ruleset(d: dict) -> val.RuleSet:
    d = dict(d)
    rules = tuple(val.RuleSpec("lord_sets", {"level": r["level"], "constraints": r["constraints"], "mode": r.get("mode", "all")},
                               r.get("label", "")) for r in d.pop("rules"))
    extra = set(d) - (set(val.RuleSet.__dataclass_fields__) - {"rules"})
    if extra:
        raise ValueError("unknown ruleset fields: %s" % sorted(extra))
    if "age_years" in d:
        d["age_years"] = tuple(d["age_years"])
    return val.RuleSet(rules=rules, **d)


def _cases(path: str):
    if path.lower().endswith(".json"):
        return val.load_cases_json(path)
    if path.lower().endswith(".csv"):
        return val.load_cases_csv(path)
    raise ValueError("cases_path must end in .json or .csv (columns: %s)" % ", ".join(val.CSV_COLUMNS))


def _registry(path):
    p = path or _DEFAULT_REGISTRY
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    return p


@tool
@_help
def dba_validation_register(ruleset: dict, note: str = "", registry_path: Optional[str] = None) -> dict[str, Any]:
    """Pre-register a rule set BEFORE scoring: freezes rules, options, split and test settings by hash. Held-out cases can only be scored for a registered hash. """
    rs = _ruleset(ruleset)
    h = val.register(rs, _registry(registry_path), note)
    return {"hash": h, "canonical": rs.canonical(), "registry": _registry(registry_path), "harness_version": val.HARNESS_VERSION}


@tool
@_help
def dba_backtest(cases_path: str, ruleset: dict, which: str = "tune", registry_path: Optional[str] = None,
                 include_events: bool = False, with_probability: bool = False,
                 with_matching: bool = False) -> dict[str, Any]:
    """Score a rule set against dated events in a user-supplied cases file (.json or .csv; one event per CSV row). which='tune' or 'heldout' (heldout needs dba_validation_register first; every look is counted). Reports hit rate with CI, exact coverage, chance hit rate, lift and p-values under the coverage null, the age-density null (outside view) and a chart-shuffle control. with_probability=True also turns the rule into a forecast over time bins (strength r fitted on the tune split) and scores it on `which` with log score, Brier score, paired bootstrap and a calibration table against a no-chart baseline. with_matching=True adds a blind chart-matching test (charts vs shuffled event histories: rank of the true chart, top-k, permutation p-value)."""
    rs = _ruleset(ruleset)
    return val.evaluate(rs, _cases(cases_path), which, _registry(registry_path), include_events, with_probability, with_matching)


@tool
@_help
def dba_stack_report(cases_path: str, ruleset: dict) -> dict[str, Any]:
    """Add the rules of a rule set one at a time (union) on the tune split: chance hit rate, hit rate, lift and the lift of each increment. Shows how much of a gain from stacking techniques is just wider coverage. """
    rs = _ruleset(ruleset)
    tune, _ = val.split_cases(_cases(cases_path), rs.split_seed, rs.tune_fraction)
    return val.stack_report(rs, [val.prepare(c, rs) for c in tune])


@tool
def dba_chance_baseline(birth: str, timezone: Optional[str] = None, longitude: Optional[float] = None,
                        lords_by_level: Optional[list[list[str]]] = None, mode: str = "all",
                        age_from: float = 0, age_to: float = 80, ayanamsa: str = "LAHIRI", year_days: str = "julian",
                        ambiguous: str = "earlier", nonexistent: str = "shift_forward") -> dict[str, Any]:
    """For one chart and one lord-set rule: exact coverage of the age range and the exact chance that a random event of day / month / year precision would fall in a window by luck. Needs no event data; use it to see how specific a rule is before testing it."""
    if not lords_by_level:
        raise ValueError("lords_by_level required, e.g. [[\"Venus\"],[\"Sun\",\"Moon\"]]")
    if mode not in ("all", "any"):
        raise ValueError("mode must be all or any")
    c = _chart(birth, timezone, longitude, ayanamsa, year_days, ambiguous, nonexistent)
    lo, hi = Fraction(age_from) * val.AGE_YEAR_DAYS, Fraction(age_to) * val.AGE_YEAR_DAYS
    level = len(lords_by_level)
    sets = [set(x) for x in lords_by_level]
    agg = all if mode == "all" else any
    r = alg.find_periods(c.moon_deg, lambda p: agg(p.path[i] in sets[i] for i in range(level)), level, lo, hi, c.yd, 300000)
    if r["truncated"]:
        raise ValueError("too many periods; use a coarser level")
    wins = alg.intersect(alg.merge_adjacent(r["periods"]), [alg.Window(lo, hi)])
    chance = {k: float(val.chance_probability(wins, lo, hi, L)) for k, L in
              (("day", Fraction(1)), ("month", Fraction(3044, 100)), ("year", val.AGE_YEAR_DAYS))}
    return {"age_range_years": [age_from, age_to], "coverage": float(alg.coverage_fraction(wins, lo, hi)),
            "n_windows": len(wins), "chance_of_hit_by_luck": chance,
            "note": "Compare a measured hit rate with the chance value for the precision of the events used.",
            "provenance": c.provenance()}


@tool
def dba_window_bands(birth: str, timezone: Optional[str] = None, longitude: Optional[float] = None,
                     lords_by_level: Optional[list[list[str]]] = None, mode: str = "all",
                     start: Optional[str] = None, end: Optional[str] = None, birth_sigma_min: float = 0,
                     ayanamsas: Optional[list[str]] = None, year_modes: Optional[list[str]] = None,
                     ayanamsa: str = "LAHIRI", year_days: str = "julian", max_windows: int = 300,
                     ambiguous: str = "earlier", nonexistent: str = "shift_forward") -> dict[str, Any]:
    """Uncertainty bands for a lord-set rule: recompute its windows under birth-time error (+-birth_sigma_min), alternative ayanamsas and alternative year lengths. `certain` = windows marked in every scenario, `possible` = marked in at least one; exact coverage of each. A big gap between the two means the rule's dates are fragile. Default range: birth to birth + 120 years."""
    if not lords_by_level:
        raise ValueError("lords_by_level required, e.g. [[\"Venus\"],[\"Sun\",\"Moon\"]]")
    if mode not in ("all", "any"):
        raise ValueError("mode must be all or any")
    c = _chart(birth, timezone, longitude, ayanamsa, year_days, ambiguous, nonexistent)
    lo = c.t_of(start) if start else Fraction(0)
    hi = c.t_of(end) if end else 120 * vim.nominal_year_days(c.yd)
    level = len(lords_by_level)
    sets = [set(x) for x in lords_by_level]
    agg = all if mode == "all" else any
    alts = [Fraction(astro.moon(c.utc, a)["longitude_deg"]) for a in (ayanamsas or []) if a.upper() != c.ayanamsa]
    scen = bands.make_scenarios(c.moon_deg, c.moon["speed_deg_per_day"], birth_sigma_min, alts,
                                 [solar.resolve_year_mode(m, c.utc) for m in (year_modes or [year_days])])
    r = bands.window_bands(lambda p: agg(p.path[i] in sets[i] for i in range(level)), level, lo, hi, scen)

    def listing(ws):
        return {"n": len(ws), "truncated": len(ws) > max_windows,
                "windows": [{"start": c.when(w.start), "end": c.when(w.end), "days": float(w.duration)} for w in ws[:max_windows]]}
    return {"range": {"start": c.when(lo), "end": c.when(hi)}, "n_scenarios": len(scen),
            "coverage_certain": float(r["coverage_certain"]), "coverage_possible": float(r["coverage_possible"]),
            "coverage_by_scenario": [{"scenario": k, "coverage": float(v)} for k, v in r["coverage_by_scenario"]],
            "certain": listing(r["certain"]), "possible": listing(r["possible"]),
            "provenance": c.provenance(note="scenarios: birth-time shifts of -sigma/0/+sigma, nominal + alternative ayanamsa Moon, year modes")}
