"""Dasha engine: all dasha calculations on top of PyJHora (no MCP code in here).

Every public function takes a `BirthState` and returns plain JSON-serialisable dicts.
PyJHora is imported lazily (first real calculation), so this module can be imported and
tested without the astrology libraries installed (see `set_backend` for test stubs).

PyJHora entry points used (v5.0.0, verified from its source):
    dhasa_selector.get_dhasa_root_nodes(0, name, dob, tob, place, divisional_chart_factor=..)
        -> [(lords, label, (y,m,d,fh), (y,m,d,fh)), ...]          Mahadasha level
    dhasa_selector.get_dhasa_children(name, lords, start, parent_end=end, jd_at_dob=.., place=..)
        -> same row shape, one level below                          Antardasha ... Deha
    applicability.applicability_check(dob, tob, place, divisional_chart_factor)
    charts.divisional_chart(jd, place, divisional_chart_factor)
        -> [['L',(sign,deg)], [0,(sign,deg)], ... [8,(sign,deg)]]
"""

from __future__ import annotations

import calendar
import difflib
import io
import sys
import threading
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Callable, Optional

from state import BirthState

YEAR_DAYS = 365.2425  # calendar year, used only for age / duration display
MAX_DEPTH = 6
LEVEL_NAMES = [
    "Mahadasha",
    "Antardasha (Bhukti)",
    "Pratyantardasha",
    "Sookshma-antardasha",
    "Prana-antardasha",
    "Deha-antardasha",
]

SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio",
         "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
PLANETS = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]
NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra", "Punarvasu", "Pushya",
    "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni", "Hasta", "Chitra", "Swati",
    "Vishakha", "Anuradha", "Jyeshtha", "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana",
    "Dhanishta", "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati",
]
NAK_LEN = 360.0 / 27.0
# Vimsottari: lord order starting from Ketu (planet ids) and years per lord
VIM_ORDER = [8, 5, 0, 1, 2, 7, 4, 6, 3]
VIM_YEARS = {8: 7, 5: 20, 0: 6, 1: 10, 2: 7, 7: 18, 4: 16, 6: 19, 3: 17}

# Classical tables used for lord analysis (Parashari)
SIGN_LORDS = [2, 5, 3, 1, 0, 3, 5, 2, 4, 6, 6, 4]
EXALTATION = {0: 0, 1: 1, 2: 9, 3: 5, 4: 3, 5: 11, 6: 6}
DEBILITATION = {p: (s + 6) % 12 for p, s in EXALTATION.items()}
NAT_FRIENDS = {0: {1, 2, 4}, 1: {0, 3}, 2: {0, 1, 4}, 3: {0, 5}, 4: {0, 1, 2}, 5: {3, 6}, 6: {3, 5}}
NAT_ENEMIES = {0: {5, 6}, 1: set(), 2: {3}, 3: {1}, 4: {3, 5}, 5: {0, 1}, 6: {0, 1, 2}}
SPECIAL_ASPECTS = {2: {4, 8}, 4: {5, 9}, 6: {3, 10}}  # on top of the 7th that every planet has
PAIR_NAMES = {1: "same sign", 2: "2/12", 3: "3/11", 4: "4/10 (kendra)", 5: "5/9 (trikona)",
              6: "6/8 (shashtashtaka)", 7: "1/7 (opposition)"}

_ANNUAL = {"patyayini", "varsha_vimsottari", "varsha_narayana", "sudarsana_chakra", "mudda"}
_EXTRA_SYSTEMS = ["rasi_bhukthi_vimsottari"]
_CONDITIONAL_DEFAULT = ["ashtottari", "chaturaaseeti_sama", "dwadasottari", "dwisatpathi",
                        "panchottari", "satabdika", "shashtisama"]
_UNCHECKED_BY_LIBRARY = ["shattrimsa_sama", "shodasottari"]
_DESCRIPTIONS = {
    "vimsottari": "Most widely used planetary dasha, driven by the Moon's nakshatra at birth.",
    "yoga_vimsottari": "Vimsottari variant based on the yoga at birth.",
    "rasi_bhukthi_vimsottari": "Vimsottari with sign-based sub-periods (antardasha level only).",
    "ashtottari": "Conditional nakshatra-based planetary dasha (8 planets, no Ketu).",
    "yogini": "Eight-yogini nakshatra-based dasha that repeats in cycles.",
    "tara": "Tara dasha (nakshatra/tara based planetary dasha).",
    "kalachakra": "Kalachakra dasha: sign-based, driven by the Moon's nakshatra pada.",
    "narayana": "Sign-based dasha (Jaimini style) counted from the lagna.",
    "chara": "Jaimini Chara dasha (sign-based).",
    "sthira": "Jaimini Sthira dasha (sign-based, fixed durations).",
    "drig": "Jaimini Drig dasha (sign-based, by aspects).",
    "karaka": "Dasha of the Jaimini karakas.",
    "naisargika": "Natural (fixed-order) planetary dasha.",
    "sudarsana_chakra": "Sudarsana chakra: lagna, Moon and Sun charts progressed together.",
}
_CATEGORY_DESC = {
    "graha": "Planet-based dasha.",
    "rasi": "Sign (rasi) based dasha.",
    "nakshatra": "Nakshatra-based dasha.",
    "karaka": "Karaka-based dasha.",
    "special": "Special dasha.",
    "annual": "Annual / progressed dasha (results are for the birth chart unless options select otherwise).",
}


class DashaError(ValueError):
    """A problem the caller can fix (bad system name, date out of range, missing library...)."""


# --------------------------------------------------------------------------------------
# Backend (PyJHora) loading
# --------------------------------------------------------------------------------------
_LOCK = threading.RLock()
_BACKEND: Any = None
_CTX: dict[str, Any] = {}
_ROOT_CACHE: dict[Any, list] = {}
_CHILD_CACHE: dict[Any, list] = {}
_POS_CACHE: dict[str, Any] = {}
_CACHE_LIMIT = 512


def set_backend(backend: Any) -> None:
    """Inject a backend (used by tests). It needs: const, utils, drik, selector, applicability, charts."""
    global _BACKEND
    with _LOCK:
        _BACKEND = backend
        for cache in (_CTX, _ROOT_CACHE, _CHILD_CACHE, _POS_CACHE):
            cache.clear()


def _backend() -> Any:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    with _LOCK:
        if _BACKEND is not None:
            return _BACKEND
        sys_path_before = list(sys.path)
        try:
            # PyJHora's package __init__ files append folders (computed from the CURRENT working directory, not from
            # themselves) to sys.path and print the whole path list. Nobody needs that: hide the output (on the stdio
            # transport it would also corrupt the MCP stream) and undo the sys.path change in the `finally` below.
            with redirect_stdout(io.StringIO()):
                from jhora import const, utils
                from jhora.horoscope.chart import charts
                from jhora.horoscope.dhasa import dhasa_selector
                from jhora.horoscope.dhasa.graha import applicability
        except ImportError as exc:  # includes ModuleNotFoundError
            raise DashaError(
                f"PyJHora (or one of its dependencies: '{getattr(exc, 'name', exc)}') is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc
        finally:
            sys.path[:] = [p for p in sys.path if p in sys_path_before]
        try:
            utils.set_language("en")
        except Exception:  # language files are optional for our purposes
            pass
        _BACKEND = SimpleNamespace(
            const=const,
            utils=utils,
            drik=utils.hora_engine_proxy,
            selector=dhasa_selector,
            applicability=applicability,
            charts=charts,
        )
        return _BACKEND


def _cache_put(cache: dict, key: Any, value: Any) -> Any:
    if len(cache) >= _CACHE_LIMIT:
        cache.clear()
    cache[key] = value
    return value


def _context(state: BirthState, b: Any) -> Any:
    key = state.fingerprint()
    ctx = _CTX.get(key)
    if ctx is not None:
        return ctx
    jd = b.utils.julian_day_number(state.dob_tuple, state.tob_tuple)
    tz = state.timezone_offset()
    if tz is None:
        tz = b.utils.get_place_timezone_offset(state.latitude, state.longitude, jd)
    try:
        if state.elevation_m is not None:
            place = b.drik.Place(state.place_name, state.latitude, state.longitude, tz, elevation=state.elevation_m)
        else:
            place = b.drik.Place(state.place_name, state.latitude, state.longitude, tz)
    except TypeError:
        place = b.drik.Place(state.place_name, state.latitude, state.longitude, tz)
    ctx = SimpleNamespace(
        jd=jd,
        dob=b.drik.Date(*state.dob_tuple),
        tob=state.tob_tuple,
        place=place,
        tz=tz,
        dcf=state.options.divisional_chart_factor,
    )
    return _cache_put(_CTX, key, ctx)


@contextmanager
def _calc(state: BirthState):
    """Serialise calculations and apply the state's dasha-year length (PyJHora keeps it global)."""
    b = _backend()
    with _LOCK:
        if not hasattr(b, "orig_year"):
            b.orig_year = b.const.dhasa_year_duration_default
        year_type = state.options.year_type
        b.const.dhasa_year_duration_default = (
            b.const.DHASA_YEAR_DURATION[year_type] if year_type else b.orig_year
        )
        try:
            b.drik.set_ayanamsa_mode(state.options.ayanamsa)
        except Exception as exc:
            raise DashaError(f"Could not set ayanamsa '{state.options.ayanamsa}' ({type(exc).__name__}: {exc})") from exc
        yield b, _context(state, b)


def _root_kwargs(state: BirthState, b: Any) -> dict[str, Any]:
    kwargs = state.options.root_kwargs()
    if state.options.savana_year_method:
        kwargs["savana_year_method"] = b.const.SAVANA_YEAR_METHOD[state.options.savana_year_method]
    return kwargs


def _child_kwargs(state: BirthState, b: Any) -> dict[str, Any]:
    kwargs = state.options.child_kwargs()
    if state.options.savana_year_method:
        kwargs["savana_year_method"] = b.const.SAVANA_YEAR_METHOD[state.options.savana_year_method]
    return kwargs


# --------------------------------------------------------------------------------------
# Time helpers and the Period type
# --------------------------------------------------------------------------------------
def _tuple_to_dt(t: Any) -> datetime:
    try:
        y, m, d, fh = t[0], t[1], t[2], (t[3] if len(t) > 3 else 0.0)
        return datetime(int(y), int(m), int(d)) + timedelta(hours=float(fh))
    except Exception as exc:
        raise DashaError(f"Could not read date tuple {t!r}") from exc


def _tup(x: Any) -> tuple:
    if isinstance(x, (list, tuple)):
        return tuple(_tup(i) if isinstance(i, (list, tuple)) else i for i in x)
    return (x,)


def _json(x: Any) -> Any:
    if isinstance(x, (list, tuple)):
        return [_json(i) for i in x]
    return x


def _ymd(a: datetime, b: datetime) -> dict[str, int]:
    if b < a:
        a, b = b, a
    y, m, d = b.year - a.year, b.month - a.month, b.day - a.day
    if d < 0:
        m -= 1
        pm, py = (b.month - 1, b.year) if b.month > 1 else (12, b.year - 1)
        d += calendar.monthrange(py, pm)[1]
    if m < 0:
        y -= 1
        m += 12
    return {"years": y, "months": m, "days": d}


def _age(birth: datetime, t: datetime) -> float:
    return round((t - birth).total_seconds() / (YEAR_DAYS * 86400.0), 4)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def parse_when(value: Optional[str]) -> datetime:
    """Accept None (= now), 'YYYY-MM-DD' or an ISO datetime. Time zones are ignored (birth-place local time)."""
    if value is None or str(value).strip() == "":
        return datetime.now().replace(microsecond=0)
    try:
        dt = datetime.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise DashaError(f"Invalid date '{value}'. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS") from exc
    return dt.replace(tzinfo=None)


@dataclass(frozen=True)
class Period:
    lords: tuple
    label: str
    start: tuple
    end: tuple
    start_dt: datetime
    end_dt: datetime


def _make_period(row: Any) -> Period:
    lords, label, start, end = row[0], row[1], row[2], row[3]
    return Period(_tup(lords), str(label), tuple(start), tuple(end), _tuple_to_dt(start), _tuple_to_dt(end))


def _period_dict(p: Period, birth: datetime, level: Optional[int] = None) -> dict[str, Any]:
    days = (p.end_dt - p.start_dt).total_seconds() / 86400.0
    out: dict[str, Any] = {}
    if level is not None:
        out["level"] = level
        out["level_name"] = LEVEL_NAMES[level - 1]
    out.update(
        lord=p.label,
        lords_raw=_json(p.lords),
        start=_iso(p.start_dt),
        end=_iso(p.end_dt),
        duration_days=round(days, 2),
        duration_years=round(days / YEAR_DAYS, 4),
        age_at_start=_age(birth, p.start_dt),
        age_at_end=_age(birth, p.end_dt),
    )
    return out


def _find(periods: list[Period], t: datetime) -> Optional[Period]:
    """Period containing t (periods are chronological). Tiny gaps between siblings belong to the earlier one."""
    best = None
    for p in periods:
        if p.start_dt <= t:
            best = p
        else:
            break
    if best is None:
        return None
    if t < best.end_dt:
        return best
    return None if best is periods[-1] else best


def _norm(x: Any) -> str:
    """Lower-case letters/digits only: PyJHora labels carry symbols, e.g. 'Jupiter♃'."""
    return "".join(ch for ch in str(x).lower() if ch.isalnum())


def _same(label: str, wanted: Any) -> bool:
    return _norm(label) == _norm(wanted)


def _check_depth(depth: int) -> int:
    if not 1 <= int(depth) <= MAX_DEPTH:
        raise DashaError("depth must be between 1 (Mahadasha) and 6 (Deha-antardasha)")
    return int(depth)


# --------------------------------------------------------------------------------------
# Systems
# --------------------------------------------------------------------------------------
_ALIASES = {
    "vimshottari": "vimsottari", "vimshottari_dasha": "vimsottari", "vimsottari_dasha": "vimsottari",
    "ashtottari_dasha": "ashtottari", "yogini_dasha": "yogini", "kalachakra_dasha": "kalachakra",
    "jaimini_chara": "chara", "chara_dasha": "chara", "shatabdika": "satabdika",
    "shodashottari": "shodasottari", "dwadashottari": "dwadasottari", "panchottari_dasha": "panchottari",
    "shashtihayani": "shashtisama", "shashti_hayani": "shashtisama",
}


def _supported_names(b: Any) -> list[str]:
    names = list(b.selector.list_supported_dhasas())
    for extra in _EXTRA_SYSTEMS:
        if extra not in names and b.selector.is_dhasa_supported(extra):
            names.append(extra)
    return names


def resolve_system(name: str) -> str:
    b = _backend()
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    key = _ALIASES.get(key, key)
    supported = _supported_names(b)
    if key in supported:
        return key
    close = difflib.get_close_matches(key, supported, n=5, cutoff=0.5)
    hint = f" Did you mean: {close}?" if close else ""
    raise DashaError(f"Unknown dasha system '{name}'.{hint} Call list_dasha_systems to see all {len(supported)}.")


def _category(b: Any, name: str) -> str:
    c = b.const
    if name in _ANNUAL:
        return "annual"
    if name in c.nakshathra_dhasas:
        return "nakshatra"
    if name in c.karaka_dhasas:
        return "karaka"
    if name in c.special_dhasas:
        return "special"
    if name in c.rasi_dhasas:
        return "rasi"
    return "graha"


def _conditional(b: Any) -> list[str]:
    return list(getattr(b.applicability, "_conditional_dhasas", _CONDITIONAL_DEFAULT))


def list_systems() -> dict[str, Any]:
    b = _backend()
    conditional = set(_conditional(b))
    systems = []
    for name in _supported_names(b):
        cat = _category(b, name)
        systems.append({
            "name": name,
            "category": cat,
            "conditional": name in conditional,
            "lifespan_years": getattr(b.const, f"human_life_span_for_{name}_dhasa", None),
            "description": _DESCRIPTIONS.get(name, _CATEGORY_DESC.get(cat, "")),
        })
    by_cat: dict[str, int] = {}
    for s in systems:
        by_cat[s["category"]] = by_cat.get(s["category"], 0) + 1
    return {
        "count": len(systems),
        "by_category": by_cat,
        "conditional_dashas_note": "Conditional systems apply only to charts that meet their rule; use check_dasha_applicability.",
        "systems": systems,
    }


# --------------------------------------------------------------------------------------
# Period computation (roots, children, ladder)
# --------------------------------------------------------------------------------------
def _roots(state: BirthState, system: str) -> list[Period]:
    key = (state.fingerprint(), system)
    cached = _ROOT_CACHE.get(key)
    if cached is not None:
        return cached
    with _calc(state) as (b, ctx):
        try:
            rows = b.selector.get_dhasa_root_nodes(
                0, system, ctx.dob, ctx.tob, ctx.place,
                divisional_chart_factor=ctx.dcf, **_root_kwargs(state, b),
            )
        except DashaError:
            raise
        except Exception as exc:
            raise DashaError(f"{system}: could not compute Mahadasha periods ({type(exc).__name__}: {exc})") from exc
    periods = [_make_period(r) for r in rows]
    if not periods:
        raise DashaError(f"{system}: no dasha periods returned for this chart")
    return _cache_put(_ROOT_CACHE, key, periods)


def _children(state: BirthState, system: str, parent: Period) -> list[Period]:
    key = (state.fingerprint(), system, parent.lords, parent.start)
    cached = _CHILD_CACHE.get(key)
    if cached is not None:
        return cached
    with _calc(state) as (b, ctx):
        core = dict(jd_at_dob=ctx.jd, place=ctx.place, divisional_chart_factor=ctx.dcf)
        call = lambda **extra: b.selector.get_dhasa_children(  # noqa: E731
            system, parent.lords, parent.start, parent_end=parent.end, **core, **extra
        )
        try:
            try:
                rows = call(**_child_kwargs(state, b))
            except TypeError:  # an option this dasha's sub-period function does not accept
                rows = call()
        except Exception as exc:
            raise DashaError(
                f"{system}: could not compute sub-periods of {parent.label} ({type(exc).__name__}: {exc})"
            ) from exc
    return _cache_put(_CHILD_CACHE, key, [_make_period(r) for r in rows])


def _ladder(state: BirthState, system: str, at: datetime, depth: int) -> list[Period]:
    depth = _check_depth(depth)
    roots = _roots(state, system)
    current = _find(roots, at)
    if current is None:
        raise DashaError(
            f"{_iso(at)} is outside the {system} dasha range "
            f"({_iso(roots[0].start_dt)} to {_iso(roots[-1].end_dt)})"
        )
    ladder = [current]
    for _ in range(depth - 1):
        nxt = _find(_children(state, system, ladder[-1]), at)
        if nxt is None:
            break
        ladder.append(nxt)
    return ladder


def _ladder_dict(ladder: list[Period], at: datetime, birth: datetime) -> list[dict[str, Any]]:
    rows = []
    for i, p in enumerate(ladder, start=1):
        total = (p.end_dt - p.start_dt).total_seconds()
        elapsed = (at - p.start_dt).total_seconds()
        d = _period_dict(p, birth, level=i)
        d["elapsed_percent"] = round(100.0 * max(0.0, min(elapsed, total)) / total, 2) if total else 100.0
        d["remaining"] = _ymd(at, p.end_dt)
        d["remaining_days"] = round(max(0.0, (p.end_dt - at).total_seconds() / 86400.0), 2)
        rows.append(d)
    return rows


def _header(state: BirthState, system: str) -> dict[str, Any]:
    return {"system": system, "person": state.summary()}


# --------------------------------------------------------------------------------------
# Public API: basis, applicability, balance, periods
# --------------------------------------------------------------------------------------
def _parse_positions(pp: Any) -> tuple[tuple[int, float], dict[int, tuple[int, float]]]:
    lagna, planets = None, {}
    for item in pp:
        key, val = item[0], item[1]
        if key == "L":
            lagna = (int(val[0]), float(val[1]))
        elif isinstance(key, int) and 0 <= key <= 8:
            planets[key] = (int(val[0]), float(val[1]))
    if lagna is None or len(planets) < 9:
        raise DashaError("Unexpected planet-position format returned by PyJHora")
    return lagna, planets


def _positions(state: BirthState) -> tuple[tuple[int, float], dict[int, tuple[int, float]]]:
    key = state.fingerprint()
    cached = _POS_CACHE.get(key)
    if cached is not None:
        return cached
    with _calc(state) as (b, ctx):
        try:
            pp = b.charts.divisional_chart(ctx.jd, ctx.place, divisional_chart_factor=ctx.dcf)
        except Exception as exc:
            raise DashaError(f"Could not compute chart positions ({type(exc).__name__}: {exc})") from exc
    return _cache_put(_POS_CACHE, key, _parse_positions(pp))


def _nak(sign: int, deg: float) -> dict[str, Any]:
    absolute = sign * 30 + deg
    idx = int(absolute // NAK_LEN) % 27
    within = absolute - int(absolute // NAK_LEN) * NAK_LEN
    pada = min(4, int(within // (NAK_LEN / 4)) + 1)
    return {
        "nakshatra": NAKSHATRAS[idx],
        "nakshatra_number": idx + 1,
        "pada": pada,
        "nakshatra_lord": PLANETS[VIM_ORDER[idx % 9]],
        "fraction_traversed": round(within / NAK_LEN, 6),
    }


def get_basis(state: BirthState) -> dict[str, Any]:
    lagna, planets = _positions(state)
    moon_sign, moon_deg = planets[1]
    nak = _nak(moon_sign, moon_deg)
    idx = nak["nakshatra_number"] - 1
    lord_id = VIM_ORDER[idx % 9]
    balance_years = (1.0 - nak["fraction_traversed"]) * VIM_YEARS[lord_id]
    table = []
    for pid in range(9):
        s, d = planets[pid]
        table.append({"planet": PLANETS[pid], "id": pid, "sign": SIGNS[s], "degree_in_sign": round(d, 4),
                      "house_from_lagna": (s - lagna[0]) % 12 + 1, **_nak(s, d)})
    return {
        "person": state.summary(),
        "chart": f"D{state.options.divisional_chart_factor}",
        "lagna": {"sign": SIGNS[lagna[0]], "degree_in_sign": round(lagna[1], 4), **_nak(*lagna)},
        "moon": {"sign": SIGNS[moon_sign], "degree_in_sign": round(moon_deg, 4), **nak},
        "dasha_basis": {
            "janma_nakshatra": nak["nakshatra"],
            "pada": nak["pada"],
            "vimsottari_starting_lord": PLANETS[lord_id],
            "classical_vimsottari_balance_years": round(balance_years, 4),
            "note": "Classical cross-check with a 360-day-free fraction method; the library result from "
                    "get_dasha_balance may differ slightly depending on the chosen year length.",
        },
        "planets": table,
    }


def check_applicability(state: BirthState) -> dict[str, Any]:
    with _calc(state) as (b, ctx):
        try:
            applicable = list(b.applicability.applicability_check(ctx.dob, ctx.tob, ctx.place,
                                                                  divisional_chart_factor=ctx.dcf))
        except Exception as exc:
            raise DashaError(f"Applicability check failed ({type(exc).__name__}: {exc})") from exc
        conditional = _conditional(b)
    return {
        "person": state.summary(),
        "applicable_conditional_dashas": applicable,
        "not_applicable": [c for c in conditional if c not in applicable],
        "applicability_not_checked_by_library": _UNCHECKED_BY_LIBRARY,
        "note": "Systems not listed above (e.g. vimsottari, yogini, rasi dashas) have no applicability condition.",
    }


def get_balance(state: BirthState, system: str = "vimsottari") -> dict[str, Any]:
    system = resolve_system(system)
    roots = _roots(state, system)
    birth = state.local_birth_datetime
    current = _find(roots, birth)
    if current is None:
        raise DashaError(f"Birth moment falls outside the computed {system} periods")
    return {
        **_header(state, system),
        "running_at_birth": _period_dict(current, birth, level=1),
        "elapsed_at_birth": _ymd(current.start_dt, birth),
        "balance_at_birth": _ymd(birth, current.end_dt),
        "balance_years": round((current.end_dt - birth).total_seconds() / (YEAR_DAYS * 86400.0), 4),
        "year_type": state.options.year_type or "library default",
    }


def get_mahadashas(state: BirthState, system: str = "vimsottari") -> dict[str, Any]:
    system = resolve_system(system)
    roots = _roots(state, system)
    birth = state.local_birth_datetime
    periods = []
    for i, p in enumerate(roots, start=1):
        d = _period_dict(p, birth, level=1)
        d["index"] = i
        d["running_at_birth"] = p.start_dt <= birth < p.end_dt
        periods.append(d)
    return {**_header(state, system), "count": len(periods),
            "range": {"start": _iso(roots[0].start_dt), "end": _iso(roots[-1].end_dt)}, "mahadashas": periods}


def _select_maha(roots: list[Period], maha: Any, occurrence: int) -> Period:
    if isinstance(maha, int) or (isinstance(maha, str) and maha.strip().isdigit()):
        idx = int(maha)
        if not 1 <= idx <= len(roots):
            raise DashaError(f"maha index must be 1..{len(roots)}")
        return roots[idx - 1]
    matches = [p for p in roots if _same(p.label, maha)]
    if not matches:
        raise DashaError(f"No Mahadasha '{maha}'. Available: {sorted({p.label for p in roots})}")
    if not 1 <= occurrence <= len(matches):
        raise DashaError(f"'{maha}' occurs {len(matches)} time(s); occurrence must be 1..{len(matches)}")
    return matches[occurrence - 1]


def _expand(state: BirthState, system: str, parent: Period, level: int, levels_below: int,
            birth: datetime, budget: list[int]) -> list[dict[str, Any]]:
    out = []
    for kid in _children(state, system, parent):
        budget[0] -= 1
        if budget[0] < 0:
            raise DashaError("Too many sub-periods requested; lower levels_below")
        d = _period_dict(kid, birth, level=level + 1)
        if levels_below > 1 and level + 1 < MAX_DEPTH:
            d["children"] = _expand(state, system, kid, level + 1, levels_below - 1, birth, budget)
        out.append(d)
    return out


def get_sub_periods(state: BirthState, system: str = "vimsottari", maha: Any = 1,
                    path: Optional[list[str]] = None, occurrence: int = 1,
                    levels_below: int = 1) -> dict[str, Any]:
    system = resolve_system(system)
    if not 1 <= levels_below <= 3:
        raise DashaError("levels_below must be 1..3")
    birth = state.local_birth_datetime
    parent = _select_maha(_roots(state, system), maha, occurrence)
    trail = [parent]
    for name in path or []:
        kids = _children(state, system, trail[-1])
        hit = [k for k in kids if _same(k.label, name)]
        if not hit:
            raise DashaError(f"'{name}' is not a sub-period of {' > '.join(p.label for p in trail)}. "
                             f"Available: {[k.label for k in kids]}")
        trail.append(hit[0])
    level = len(trail)
    if level >= MAX_DEPTH:
        raise DashaError("Deha-antardasha is the deepest level; it has no sub-periods")
    return {
        **_header(state, system),
        "parent_path": [p.label for p in trail],
        "parent": _period_dict(trail[-1], birth, level=level),
        "children_level_name": LEVEL_NAMES[level],
        "sub_periods": _expand(state, system, trail[-1], level, levels_below, birth, [3000]),
    }


def get_current(state: BirthState, system: str = "vimsottari", on: Optional[str] = None,
                depth: int = 3) -> dict[str, Any]:
    system = resolve_system(system)
    at = parse_when(on)
    birth = state.local_birth_datetime
    ladder = _ladder(state, system, at, depth)
    return {
        **_header(state, system),
        "as_of": _iso(at),
        "age_years": _age(birth, at),
        "running": " > ".join(p.label for p in ladder),
        "levels": _ladder_dict(ladder, at, birth),
    }


def get_at_age(state: BirthState, system: str = "vimsottari", age: float = 30.0, depth: int = 3) -> dict[str, Any]:
    if age < 0:
        raise DashaError("age must be 0 or more")
    at = state.local_birth_datetime + timedelta(days=float(age) * YEAR_DAYS)
    out = get_current(state, system, _iso(at), depth)
    out["requested_age_years"] = age
    return out


def get_upcoming(state: BirthState, system: str = "vimsottari", on: Optional[str] = None,
                 level: int = 2, count: int = 5) -> dict[str, Any]:
    system = resolve_system(system)
    level = _check_depth(level)
    if not 1 <= count <= 50:
        raise DashaError("count must be 1..50")
    at = parse_when(on)
    birth = state.local_birth_datetime
    ladder = _ladder(state, system, at, level)
    if len(ladder) < level:
        raise DashaError(f"{system} has no sub-periods down to level {level} at this date")
    items = [(ladder, _period_dict(ladder[-1], birth, level=level))]
    items[0][1]["status"] = "running"
    for _ in range(count):
        probe = items[-1][0][-1].end_dt + timedelta(seconds=1)
        try:
            nxt = _ladder(state, system, probe, level)
        except DashaError:
            break
        if len(nxt) < level or nxt[-1] == items[-1][0][-1]:
            break
        d = _period_dict(nxt[-1], birth, level=level)
        d["status"] = "upcoming"
        items.append((nxt, d))
    rows = []
    for lad, d in items:
        d["path"] = " > ".join(p.label for p in lad)
        rows.append(d)
    return {**_header(state, system), "as_of": _iso(at), "level": level,
            "level_name": LEVEL_NAMES[level - 1], "periods": rows}


def _walk(state: BirthState, system: str, start: datetime, end: datetime, depth: int,
          accept: Optional[Callable[[Period], bool]], limit: int, budget: int = 6000):
    roots = _roots(state, system)
    out: list[tuple[list[str], Period]] = []
    visits = [0]
    truncated = [False]

    def rec(p: Period, lvl: int, trail: list[str]) -> None:
        if truncated[0] or p.end_dt <= start or p.start_dt >= end:
            return
        visits[0] += 1
        if visits[0] > budget:
            raise DashaError("Range/depth too large. Narrow the date range or lower the depth.")
        here = trail + [p.label]
        if lvl == depth:
            if accept is None or accept(p):
                if len(out) >= limit:
                    truncated[0] = True
                    return
                out.append((here, p))
            return
        for kid in _children(state, system, p):
            rec(kid, lvl + 1, here)

    for r in roots:
        rec(r, 1, [])
    return out, truncated[0]


def get_timeline(state: BirthState, system: str = "vimsottari", start: Optional[str] = None,
                 end: Optional[str] = None, depth: int = 2, limit: int = 200) -> dict[str, Any]:
    system = resolve_system(system)
    depth = _check_depth(depth)
    if not 1 <= limit <= 1000:
        raise DashaError("limit must be 1..1000")
    birth = state.local_birth_datetime
    s = parse_when(start) if start else birth
    e = parse_when(end) if end else s + timedelta(days=YEAR_DAYS * 10)
    if e <= s:
        raise DashaError("end must be after start")
    rows, truncated = _walk(state, system, s, e, depth, None, limit)
    periods = []
    for trail, p in rows:
        d = _period_dict(p, birth, level=depth)
        d["path"] = " > ".join(trail)
        periods.append(d)
    return {**_header(state, system), "from": _iso(s), "to": _iso(e), "depth": depth,
            "level_name": LEVEL_NAMES[depth - 1], "count": len(periods), "truncated": truncated,
            "periods": periods}


def search_periods(state: BirthState, system: str = "vimsottari", lord: str = "Jupiter", level: int = 2,
                   start: Optional[str] = None, end: Optional[str] = None, limit: int = 100) -> dict[str, Any]:
    system = resolve_system(system)
    level = _check_depth(level)
    if not 1 <= limit <= 1000:
        raise DashaError("limit must be 1..1000")
    birth = state.local_birth_datetime
    s = parse_when(start) if start else birth
    e = parse_when(end) if end else s + timedelta(days=YEAR_DAYS * 100)
    if e <= s:
        raise DashaError("end must be after start")
    rows, truncated = _walk(state, system, s, e, level, lambda p: _same(p.label, lord), limit)
    periods = []
    for trail, p in rows:
        d = _period_dict(p, birth, level=level)
        d["path"] = " > ".join(trail)
        periods.append(d)
    return {**_header(state, system), "lord": lord, "level": level, "level_name": LEVEL_NAMES[level - 1],
            "from": _iso(s), "to": _iso(e), "count": len(periods), "truncated": truncated, "periods": periods}


def compare_systems(state: BirthState, systems: Optional[list[str]] = None, on: Optional[str] = None,
                    depth: int = 2) -> dict[str, Any]:
    names = systems or ["vimsottari", "yogini", "ashtottari", "chara", "narayana", "kalachakra"]
    at = parse_when(on)
    birth = state.local_birth_datetime
    results = {}
    for raw in names:
        try:
            system = resolve_system(raw)
            ladder = _ladder(state, system, at, depth)
            results[system] = {"running": " > ".join(p.label for p in ladder),
                               "levels": _ladder_dict(ladder, at, birth)}
        except DashaError as exc:
            results[str(raw)] = {"error": str(exc)}
    return {"person": state.summary(), "as_of": _iso(at), "depth": depth, "systems": results}


# --------------------------------------------------------------------------------------
# Dasha-lord analysis
# --------------------------------------------------------------------------------------
def _temp_relation(sign_a: int, sign_b: int) -> str:
    return "friend" if ((sign_b - sign_a) % 12 + 1) in (2, 3, 4, 10, 11, 12) else "enemy"


def _natural(a: int, b: int) -> str:
    if b in NAT_FRIENDS.get(a, ()):
        return "friend"
    if b in NAT_ENEMIES.get(a, ()):
        return "enemy"
    return "neutral"


def _compound(a: int, b: int, signs: dict[int, tuple[int, float]]) -> str:
    nat, temp = _natural(a, b), _temp_relation(signs[a][0], signs[b][0])
    table = {("friend", "friend"): "great friend", ("friend", "enemy"): "neutral",
             ("neutral", "friend"): "friend", ("neutral", "enemy"): "enemy",
             ("enemy", "friend"): "neutral", ("enemy", "enemy"): "great enemy"}
    return table[(nat, temp)]


def _dignity(pid: int, planets: dict[int, tuple[int, float]]) -> str:
    if pid > 6:
        return "n/a (node)"
    sign = planets[pid][0]
    if EXALTATION[pid] == sign:
        return "exalted"
    if DEBILITATION[pid] == sign:
        return "debilitated"
    lord = SIGN_LORDS[sign]
    if lord == pid:
        return "own sign"
    return f"in {_compound(pid, lord, planets)}'s sign"


def _planet_info(pid: int, lagna: tuple[int, float], planets: dict[int, tuple[int, float]]) -> dict[str, Any]:
    sign, deg = planets[pid]
    moon_sign = planets[1][0]
    aspecting = []
    for other, (osign, _) in planets.items():
        if other == pid:
            continue
        distance = (sign - osign) % 12 + 1
        if distance in ({7} | SPECIAL_ASPECTS.get(other, set())):
            aspecting.append(PLANETS[other])
    return {
        "planet": PLANETS[pid],
        "sign": SIGNS[sign],
        "degree_in_sign": round(deg, 4),
        "house_from_lagna": (sign - lagna[0]) % 12 + 1,
        "house_from_moon": (sign - moon_sign) % 12 + 1,
        **_nak(sign, deg),
        "dignity": _dignity(pid, planets),
        "sign_lord": PLANETS[SIGN_LORDS[sign]],
        "houses_owned": [(s - lagna[0]) % 12 + 1 for s in range(12) if SIGN_LORDS[s] == pid] if pid <= 6 else [],
        "conjunct_with": [PLANETS[o] for o, (s, _) in planets.items() if o != pid and s == sign],
        "aspected_by": aspecting,
    }


def _sign_info(sign: int, lagna: tuple[int, float], planets: dict[int, tuple[int, float]]) -> dict[str, Any]:
    lord = SIGN_LORDS[sign]
    return {
        "sign": SIGNS[sign],
        "house_from_lagna": (sign - lagna[0]) % 12 + 1,
        "sign_lord": PLANETS[lord],
        "sign_lord_position": _planet_info(lord, lagna, planets),
        "occupants": [PLANETS[o] for o, (s, _) in planets.items() if s == sign],
    }


def _pair(a: int, b: int, planets: dict[int, tuple[int, float]]) -> dict[str, Any]:
    d = (planets[b][0] - planets[a][0]) % 12 + 1
    folded = d if d <= 7 else 13 - d
    out = {"from": PLANETS[a], "to": PLANETS[b], "house_of_to_from_from": d, "pair": PAIR_NAMES[folded]}
    if a <= 6 and b <= 6:
        out["natural_relation"] = _natural(a, b)
        out["temporary_relation"] = _temp_relation(planets[a][0], planets[b][0])
        out["compound_relation"] = _compound(a, b, planets)
    return out


def analyze_lords(state: BirthState, system: str = "vimsottari", on: Optional[str] = None,
                  depth: int = 3) -> dict[str, Any]:
    system = resolve_system(system)
    at = parse_when(on)
    birth = state.local_birth_datetime
    ladder = _ladder(state, system, at, depth)
    b = _backend()
    category = _category(b, system)
    lagna, planets = _positions(state)
    levels = []
    for i, p in enumerate(ladder, start=1):
        lord = p.lords[-1]
        entry = {"level": i, "level_name": LEVEL_NAMES[i - 1], "lord": p.label,
                 "start": _iso(p.start_dt), "end": _iso(p.end_dt)}
        if isinstance(lord, int) and category == "graha" and 0 <= lord <= 8:
            entry["position"] = _planet_info(lord, lagna, planets)
        elif isinstance(lord, int) and category == "rasi" and 0 <= lord <= 11:
            entry["position"] = _sign_info(lord, lagna, planets)
        else:
            entry["position"] = None
            entry["note"] = "Position analysis is available for planet-based and sign-based dashas only."
        levels.append(entry)
    relations = []
    if category == "graha":
        ids = [p.lords[-1] for p in ladder]
        if all(isinstance(x, int) and 0 <= x <= 8 for x in ids):
            for j in range(1, len(ids)):
                r = _pair(ids[j - 1], ids[j], planets)
                r["levels"] = f"{LEVEL_NAMES[j - 1]} -> {LEVEL_NAMES[j]}"
                relations.append(r)
    return {
        **_header(state, system),
        "as_of": _iso(at),
        "age_years": _age(birth, at),
        "chart": f"D{state.options.divisional_chart_factor}",
        "lagna": {"sign": SIGNS[lagna[0]], "degree_in_sign": round(lagna[1], 4)},
        "running": " > ".join(p.label for p in ladder),
        "levels": levels,
        "lord_relationships": relations,
        "method": "Parashari: classical exaltation/debilitation, natural + temporary (compound) friendship, "
                  "7th aspect plus Mars 4/8, Jupiter 5/9, Saturn 3/10. Facts only; no predictions.",
    }
