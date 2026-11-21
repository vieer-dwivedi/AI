"""Transit (gochara) engine on top of PyJHora (no MCP code in here).

Every public function takes a `BirthState` and returns plain JSON-serialisable dicts.
PyJHora is imported lazily, so this module can be imported without the astrology libraries
(see `set_backend` for test stubs).

Design (see progress.md, "PyJHora findings"):
  * PyJHora's own searchers can take 20 s+, so event searches use OUR adaptive scanner + bisection on
    `sidereal_longitude`, always inside a bounded window.
  * Every PyJHora call runs inside `redirect_stdout` (some functions print debug text; on stdio that would corrupt MCP).
  * Julian days from PyJHora are LOCAL; UTC = jd - tz/24. `sidereal_longitude` takes UTC jd and Swiss-ephemeris ids.

Planet ids used in the API: 0 Sun, 1 Moon, 2 Mars, 3 Mercury, 4 Jupiter, 5 Venus, 6 Saturn, 7 Rahu, 8 Ketu.
"""

from __future__ import annotations

import io
import sys
import threading
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from state import BirthState

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
EPOCH_JD = 2451545.0  # 2000-01-01 12:00
EPOCH_DT = datetime(2000, 1, 1, 12, 0, 0)

# Upper bounds of apparent speed (deg/day) used to size scanner steps (sidereal; includes retrograde).
SPEED_BOUND = {0: 1.1, 1: 15.5, 2: 0.85, 3: 2.4, 4: 0.3, 5: 1.4, 6: 0.15, 7: 0.07, 8: 0.07}
# Scanner step caps per planet (days). The adaptive step is already bounded by distance/speed_bound, so these only
# stop the scanner from taking absurdly long strides when the target is far away.
MAX_STEP = {0: 15.0, 1: 1.0, 2: 20.0, 3: 5.0, 4: 40.0, 5: 8.0, 6: 60.0, 7: 90.0, 8: 90.0}
# Station search step (days): stations are far apart, so coarse steps are safe.
STATION_STEP = {2: 4.0, 3: 1.0, 4: 6.0, 5: 2.0, 6: 6.0}

# Saturn from natal Moon (classical): 12, 1, 2 = Sade Sati phases; 4 = Kantaka/Ardha-ashtama; 8 = Ashtama.
SADE_SATI_HOUSES = {12: "rising (12th from Moon)", 1: "peak (over Moon sign)", 2: "setting (2nd from Moon)"}
DHAIYA_HOUSES = {4: "Ardha-ashtama / Kantaka Shani (4th from Moon)", 8: "Ashtama Shani (8th from Moon)"}

# Classical gochara (Parashari): houses from natal Moon in which a transiting planet is generally favourable.
GOCHARA_GOOD = {
    0: {3, 6, 10, 11},
    1: {1, 3, 6, 7, 10, 11},
    2: {3, 6, 11},
    3: {2, 4, 6, 8, 10, 11},
    4: {2, 5, 7, 9, 11},
    5: {1, 2, 3, 4, 5, 8, 9, 11, 12},
    6: {3, 6, 11},
    7: {3, 6, 10, 11},
    8: {3, 6, 10, 11},
}
# Full-sign (rasi) aspects as house distances counted from the planet (7th for all; extras for Mars/Jupiter/Saturn).
ASPECT_HOUSES = {p: {7} for p in range(9)}
ASPECT_HOUSES[2] = {4, 7, 8}
ASPECT_HOUSES[4] = {5, 7, 9}
ASPECT_HOUSES[6] = {3, 7, 10}
ASPECT_HOUSES[7] = {5, 7, 9}
ASPECT_HOUSES[8] = {5, 7, 9}

MAX_EVENTS = 200


class TransitError(ValueError):
    """A problem the caller can fix (bad planet name, date out of range, missing library...)."""


# --------------------------------------------------------------------------------------
# Backend (PyJHora) loading
# --------------------------------------------------------------------------------------
_LOCK = threading.RLock()
_BACKEND: Any = None
_CTX: dict[str, Any] = {}
_NATAL: dict[str, Any] = {}
_CACHE_LIMIT = 256


def set_backend(backend: Any) -> None:
    """Inject a backend (used by tests). It needs: const, utils, drik, charts."""
    global _BACKEND
    with _LOCK:
        _BACKEND = backend
        _CTX.clear()
        _NATAL.clear()


def _backend() -> Any:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    with _LOCK:
        if _BACKEND is not None:
            return _BACKEND
        sys_path_before = list(sys.path)
        try:
            # PyJHora's __init__ files print and extend sys.path; hide the output and undo the path change.
            with redirect_stdout(io.StringIO()):
                from jhora import const, utils
                from jhora.horoscope.chart import ashtakavarga, charts
                from jhora.horoscope.transit import saham, tajaka, tajaka_yoga  # tajaka_yoga patched below
        except ImportError as exc:
            raise TransitError(
                f"PyJHora (or one of its dependencies: '{getattr(exc, 'name', exc)}') is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc
        finally:
            sys.path[:] = [p for p in sys.path if p in sys_path_before]
        try:
            utils.set_language("en")
        except Exception:
            pass
        try:  # PyJHora bug: tajaka_yoga uses `charts` without importing it
            tajaka_yoga.charts = charts
        except Exception:
            pass
        _BACKEND = SimpleNamespace(const=const, utils=utils, drik=utils.hora_engine_proxy, charts=charts,
                                   tajaka=tajaka, tajaka_yoga=tajaka_yoga, saham=saham,
                                   ashtakavarga=ashtakavarga)
        return _BACKEND


def _quiet(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a PyJHora function with its stdout swallowed."""
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


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
    ctx = SimpleNamespace(jd=jd, dob=b.drik.Date(*state.dob_tuple), tob=state.tob_tuple, place=place, tz=tz,
                          dcf=state.options.divisional_chart_factor)
    return _cache_put(_CTX, key, ctx)


@contextmanager
def _calc(state: BirthState):
    """Serialise calculations (PyJHora keeps the ayanamsa global), apply the ayanamsa, hide stdout."""
    b = _backend()
    with _LOCK:
        try:
            _quiet(b.drik.set_ayanamsa_mode, state.options.ayanamsa)
        except Exception as exc:
            raise TransitError(f"Could not set ayanamsa '{state.options.ayanamsa}' ({type(exc).__name__}: {exc})") from exc
        yield b, _context(state, b)


# --------------------------------------------------------------------------------------
# Time helpers
# --------------------------------------------------------------------------------------
def jd_from_local(dt: datetime) -> float:
    """Local-clock datetime -> PyJHora-style local Julian day."""
    return EPOCH_JD + (dt - EPOCH_DT).total_seconds() / 86400.0


def local_from_jd(jd_local: float) -> datetime:
    return EPOCH_DT + timedelta(days=jd_local - EPOCH_JD)


def parse_when(value: Optional[str], state: BirthState, default: Optional[datetime] = None) -> datetime:
    """Parse 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM[:SS]' (local at the birth place). None -> `default` or now."""
    if value is None or str(value).strip() == "":
        if default is not None:
            return default
        return now_local(state)
    text = str(value).strip().replace(" ", "T")
    try:
        if "T" not in text:
            return datetime.fromisoformat(text + "T00:00:00")
        dt = datetime.fromisoformat(text)
        return dt.replace(tzinfo=None)
    except ValueError as exc:
        raise TransitError(f"Bad date '{value}': use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS (local time at the birth place).") from exc


def now_local(state: BirthState) -> datetime:
    if state.timezone is not None:
        return datetime.now(ZoneInfo(state.timezone)).replace(tzinfo=None, microsecond=0)
    off = state.timezone_offset_hours
    if off is None:
        off = 0.0
    return (datetime.utcnow() + timedelta(hours=off)).replace(microsecond=0)


def _tz_at(state: BirthState, ctx: Any, local_dt: datetime) -> float:
    tz = state.tz_for(local_dt)
    return ctx.tz if tz is None else tz


def jd_utc_from_local(state: BirthState, ctx: Any, dt: datetime) -> float:
    return jd_from_local(dt) - _tz_at(state, ctx, dt) / 24.0


def local_dt_from_utc(state: BirthState, ctx: Any, jd_utc: float) -> datetime:
    """UTC jd -> local clock time at the birth place (daylight saving aware when a named timezone is given)."""
    approx = local_from_jd(jd_utc + ctx.tz / 24.0)
    tz = _tz_at(state, ctx, approx)
    return local_from_jd(jd_utc + tz / 24.0)


def fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _window(state: BirthState, start: Optional[str], end: Optional[str], default_days: float,
            max_days: float, backwards: bool = False) -> tuple[datetime, datetime]:
    """Resolve [start, end] (local datetimes). Default: now .. now + default_days (or backwards)."""
    now = now_local(state)
    if start is None and end is None:
        a, b = (now - timedelta(days=default_days), now) if backwards else (now, now + timedelta(days=default_days))
    elif start is not None and end is None:
        a = parse_when(start, state)
        b = a + timedelta(days=default_days)
    elif start is None:
        b = parse_when(end, state)
        a = now if now < b else b - timedelta(days=default_days)
    else:
        a, b = parse_when(start, state), parse_when(end, state)
    if b <= a:
        raise TransitError("`end` must be after `start`.")
    if (b - a).days > max_days:
        raise TransitError(f"Search window too long ({(b - a).days} days); the maximum is {int(max_days)} days "
                           "(about %.0f years). Split the search into several calls." % (max_days / 365.25))
    if a.year < 1000 or b.year > 3000:
        raise TransitError("Dates must be between the years 1000 and 3000.")
    return a, b


# --------------------------------------------------------------------------------------
# Planet naming
# --------------------------------------------------------------------------------------
_PLANET_ALIASES = {
    "sun": 0, "surya": 0, "ravi": 0, "moon": 1, "chandra": 1, "mars": 2, "mangal": 2, "kuja": 2,
    "mercury": 3, "budha": 3, "jupiter": 4, "guru": 4, "brihaspati": 4, "venus": 5, "shukra": 5,
    "saturn": 6, "shani": 6, "rahu": 7, "north node": 7, "ketu": 8, "south node": 8,
}


def planet_id(value: Any) -> int:
    """Accept 0..8, a name ('Saturn', 'shani') -> planet id."""
    if isinstance(value, bool):
        raise TransitError(f"Unknown planet '{value}'.")
    if isinstance(value, int) and 0 <= value <= 8:
        return value
    text = str(value).strip().lower()
    if text.isdigit() and 0 <= int(text) <= 8:
        return int(text)
    if text in _PLANET_ALIASES:
        return _PLANET_ALIASES[text]
    raise TransitError(f"Unknown planet '{value}'. Use one of {PLANETS} (or Sanskrit names like Shani, Guru).")


def sign_id(value: Any) -> int:
    """Accept 0..11, 1..12 is NOT accepted (ambiguous); use a sign name, or 0-11."""
    if isinstance(value, int) and not isinstance(value, bool):
        if 0 <= value <= 11:
            return value
        raise TransitError("Sign index must be 0 (Aries) .. 11 (Pisces); or give the sign name.")
    text = str(value).strip().lower()
    for i, name in enumerate(SIGNS):
        if name.lower() == text or name.lower().startswith(text) and len(text) >= 3:
            return i
    if text.isdigit() and 0 <= int(text) <= 11:
        return int(text)
    raise TransitError(f"Unknown sign '{value}'. Use one of {SIGNS}.")


# --------------------------------------------------------------------------------------
# Raw positions
# --------------------------------------------------------------------------------------
def _lon(b: Any, jd_utc: float, p: int) -> float:
    return _quiet(b.drik.sidereal_longitude, jd_utc, b.drik.ephemeris_planet_index(p)) % 360.0


def _speed(b: Any, jd_utc: float, p: int, h: float = 0.01) -> float:
    """Daily motion (deg/day, negative = retrograde) by central difference."""
    d = (_lon(b, jd_utc + h, p) - _lon(b, jd_utc - h, p) + 180.0) % 360.0 - 180.0
    return d / (2 * h)


def _wrap(x: float) -> float:
    """Wrap an angle difference to [-180, 180)."""
    return (x + 180.0) % 360.0 - 180.0


def describe_longitude(lon: float) -> dict[str, Any]:
    lon %= 360.0
    sign = int(lon // 30)
    nak = int(lon // NAK_LEN) % 27
    pada = int((lon % NAK_LEN) // (NAK_LEN / 4)) + 1
    return {
        "longitude": round(lon, 4),
        "sign": SIGNS[sign],
        "sign_index": sign,
        "degree_in_sign": round(lon - sign * 30, 4),
        "nakshatra": NAKSHATRAS[nak],
        "nakshatra_index": nak + 1,
        "pada": pada,
    }


def _planet_state(b: Any, jd_utc: float, p: int) -> dict[str, Any]:
    lon = _lon(b, jd_utc, p)
    sp = _speed(b, jd_utc, p)
    out = {"planet": PLANETS[p], **describe_longitude(lon), "speed_deg_per_day": round(sp, 4),
           "retrograde": bool(sp < 0)}
    if p in (7, 8):
        out["retrograde"] = True  # the nodes are always (mean) retrograde; shown for completeness
        out["note"] = "Nodes move backwards on average; the 'retrograde' flag is informational only."
    return out


# --------------------------------------------------------------------------------------
# Natal chart
# --------------------------------------------------------------------------------------
def _natal(state: BirthState, b: Any, ctx: Any) -> SimpleNamespace:
    """Natal longitudes (lagna + 9 planets) in the state's reference chart (D1 by default)."""
    key = state.fingerprint()
    cached = _NATAL.get(key)
    if cached is not None:
        return cached
    rows = _quiet(b.charts.divisional_chart, ctx.jd, ctx.place, divisional_chart_factor=ctx.dcf)
    lons: dict[Any, float] = {}
    for row in rows:
        ident, (sign, deg) = row[0], row[1]
        lons[ident] = (int(sign) * 30.0 + float(deg)) % 360.0
    if "L" not in lons or any(p not in lons for p in range(9)):
        raise TransitError("Could not read the natal chart from PyJHora (unexpected chart format).")
    nat = SimpleNamespace(
        lagna=lons["L"], lon={p: lons[p] for p in range(9)},
        moon_sign=int(lons[1] // 30), lagna_sign=int(lons["L"] // 30),
    )
    return _cache_put(_NATAL, key, nat)


def _house(from_sign: int, sign: int) -> int:
    return (sign - from_sign) % 12 + 1


# --------------------------------------------------------------------------------------
# Adaptive scanner + bisection
# --------------------------------------------------------------------------------------
def _bisect(g: Callable[[float], float], lo: float, hi: float, glo: float, tol: float = 2e-6) -> float:
    """Root of g in [lo, hi] where g(lo)=glo and g(hi) have opposite signs. tol in days (~0.2 s)."""
    for _ in range(60):
        if hi - lo <= tol:
            break
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        if (gm < 0) == (glo < 0):
            lo, glo = mid, gm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def find_crossings(g: Callable[[float], float], t0: float, t1: float, speed_bound: float,
                   min_step: float, max_step: float, limit: int = MAX_EVENTS,
                   backwards: bool = False) -> list[tuple[float, int]]:
    """Zero crossings of an angle-difference function `g` (values wrapped to [-180, 180)) in [t0, t1].

    The step adapts to the distance from the target (|g| / speed_bound), so far-away stretches are skipped
    quickly and the scanner slows down near the target. Returns [(jd, direction)], direction +1 when g goes
    from negative to positive. Jumps across the +-180 wrap are ignored.
    """
    out: list[tuple[float, int]] = []
    t = t0
    gt = g(t)
    guard = 0
    while t < t1 and len(out) < limit:
        guard += 1
        if guard > 200000:
            break
        step = min(max_step, max(min_step, 0.7 * abs(gt) / speed_bound))
        nt = min(t + step, t1)
        gn = g(nt)
        if abs(gt) < 90.0 and abs(gn) < 90.0 and (gt < 0) != (gn < 0) and gt != 0.0:
            root = _bisect(g, t, nt, gt)
            out.append((root, 1 if gn > gt else -1))
        t, gt = nt, gn
    return out


def _longitude_target_fn(b: Any, p: int, target: float) -> Callable[[float], float]:
    return lambda jd: _wrap(_lon(b, jd, p) - target)


def _event_row(state: BirthState, ctx: Any, jd_utc: float) -> dict[str, Any]:
    return {"datetime_local": fmt_dt(local_dt_from_utc(state, ctx, jd_utc)), "jd_utc": round(jd_utc, 6)}
