"""Swiss Ephemeris backend for the KP server (no MCP code here).

  * One lock serialises calculations because the sidereal mode is global state in swisseph.
  * Positions are sidereal (selected KP ayanamsa), geocentric, with speed (retrograde detection).
  * Houses: Placidus (KP standard). Beyond the polar circle Placidus is undefined; we fall back to Porphyry and report it.
  * `swisseph` is imported lazily, so `kp_tables` / `state` stay usable without it.

Internal planet order = the API order: Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn, Rahu, Ketu.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Iterator, Optional

from kp_tables import PLANETS, DAY_LORD_BY_WEEKDAY
from state import KPOptions, _Place

AYANAMSA_MODE = {"KP_NEW": 45, "KP_OLD": 5, "LAHIRI": 1, "RAMAN": 3}
SWE_ID = {"Sun": 0, "Moon": 1, "Mercury": 2, "Venus": 3, "Mars": 4, "Jupiter": 5, "Saturn": 6,
          "Uranus": 7, "Neptune": 8, "Pluto": 9}
OUTER = ["Uranus", "Neptune", "Pluto"]
EPOCH_JD = 2451545.0  # 2000-01-01 12:00 UT
EPOCH_DT = datetime(2000, 1, 1, 12, 0, 0)

_LOCK = threading.RLock()
_SWE: Any = None


class KPError(ValueError):
    """A problem the caller can fix (bad input, missing library, polar latitude...)."""


def swe() -> Any:
    global _SWE
    if _SWE is None:
        try:
            import swisseph  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise KPError("pyswisseph is not installed. Run: pip install -r requirements.txt") from exc
        _SWE = swisseph
    return _SWE


def wrap180(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


class Ctx:
    """Calculation context: a place + options with the sidereal mode already applied (use via `calc`)."""

    def __init__(self, place: _Place, opts: KPOptions):
        self.s = swe()
        self.place = place
        self.opts = opts
        self.flags = self.s.FLG_SWIEPH | self.s.FLG_SPEED | self.s.FLG_SIDEREAL
        self.node_id = self.s.MEAN_NODE if opts.node_type == "mean" else self.s.TRUE_NODE

    # ---- time -------------------------------------------------------------------------
    def jd_from_local(self, dt: datetime) -> float:
        """Local clock datetime -> Julian day (UT)."""
        utc = dt - timedelta(hours=self.place.tz_at(dt))
        h = utc.hour + utc.minute / 60.0 + (utc.second + utc.microsecond / 1e6) / 3600.0
        return self.s.julday(utc.year, utc.month, utc.day, h)

    def local_from_jd(self, jd: float) -> datetime:
        """Julian day (UT) -> local clock datetime (daylight saving aware)."""
        y, m, d, h = self.s.revjul(jd)
        base = datetime(y, m, d) + timedelta(hours=h)
        approx = base + timedelta(hours=self.place.tz_at(base))
        return base + timedelta(hours=self.place.tz_at(approx))

    # ---- ayanamsa & positions -----------------------------------------------------------
    def ayanamsa(self, jd: float) -> float:
        return self.s.get_ayanamsa_ut(jd)

    def lon_speed(self, name: str, jd: float) -> tuple[float, float]:
        """Sidereal longitude (deg) and speed (deg/day) of a planet name (Rahu/Ketu included)."""
        if name in ("Rahu", "Ketu"):
            xx, _ = self.s.calc_ut(jd, self.node_id, self.flags)
            lon = xx[0] if name == "Rahu" else xx[0] + 180.0
            return lon % 360.0, xx[3]
        if name not in SWE_ID:
            raise KPError(f"Unknown body '{name}'.")
        xx, _ = self.s.calc_ut(jd, SWE_ID[name], self.flags)
        return xx[0] % 360.0, xx[3]

    def lon(self, name: str, jd: float) -> float:
        return self.lon_speed(name, jd)[0]

    def positions(self, jd: float, include_outer: Optional[bool] = None) -> dict[str, dict[str, float]]:
        names = list(PLANETS) + (OUTER if (self.opts.include_outer if include_outer is None else include_outer) else [])
        out = {}
        for n in names:
            lon, sp = self.lon_speed(n, jd)
            retro = sp < 0 if n not in ("Rahu", "Ketu") else True
            out[n] = {"lon": lon, "speed": sp, "retro": bool(retro)}
        return out

    # ---- houses -------------------------------------------------------------------------
    def cusps(self, jd: float, lat: Optional[float] = None, lon: Optional[float] = None) -> dict[str, Any]:
        """12 sidereal cusps (list index 0 = cusp 1), ascendant, MC, system actually used, optional warning."""
        lat = self.place.latitude if lat is None else lat
        lon = self.place.longitude if lon is None else lon
        want = b"P" if self.opts.house_system == "placidus" else b"O"
        warning = None
        used = "placidus" if want == b"P" else "porphyry"
        try:
            cusps, ascmc = self.s.houses_ex(jd, lat, lon, want, self.s.FLG_SIDEREAL)
        except Exception:
            if want == b"O":
                raise KPError("Could not compute houses for this latitude.")
            try:
                cusps, ascmc = self.s.houses_ex(jd, lat, lon, b"O", self.s.FLG_SIDEREAL)
            except Exception as exc:
                raise KPError("Could not compute houses for this latitude.") from exc
            used = "porphyry"
            warning = (f"Placidus houses are undefined at latitude {lat:.2f} (beyond the polar circle); "
                       "Porphyry cusps were used instead. KP sub lords of cusps are not standard KP here.")
        return {"cusps": [c % 360.0 for c in cusps[:12]], "asc": ascmc[0] % 360.0, "mc": ascmc[1] % 360.0,
                "armc": ascmc[2], "system": used, "warning": warning}

    def asc_only(self, jd: float, lat: float, lon: float) -> float:
        """Sidereal ascendant (Porphyry call: same Ascendant, never fails at high latitude)."""
        _, ascmc = self.s.houses_ex(jd, lat, lon, b"O", self.s.FLG_SIDEREAL)
        return ascmc[0] % 360.0

    def solve_time_for_ascendant(self, target: float, jd_center: float, lat: float, lon: float) -> float:
        """Julian day (UT) within +-1.2 days of `jd_center` at which the sidereal Ascendant equals `target`."""
        def f(jd: float) -> float:
            return wrap180(self.asc_only(jd, lat, lon) - target)

        lo = jd_center - 0.6
        end = jd_center + 0.6
        step = 0.004
        t, ft = lo, f(lo)
        best: Optional[float] = None
        while t < end:
            nt = t + step
            fn = f(nt)
            if abs(ft) < 90 and abs(fn) < 90 and (ft < 0) != (fn < 0):
                a, b, fa = t, nt, ft
                for _ in range(60):
                    mid = 0.5 * (a + b)
                    fm = f(mid)
                    if (fm < 0) == (fa < 0):
                        a, fa = mid, fm
                    else:
                        b = mid
                    if b - a < 1e-9:
                        break
                cand = 0.5 * (a + b)
                if best is None or abs(cand - jd_center) < abs(best - jd_center):
                    best = cand
            t, ft = nt, fn
        if best is None:
            raise KPError("Could not find the moment when this Ascendant rises at this place (extreme latitude?).")
        return best

    # ---- sunrise / Hindu weekday ----------------------------------------------------------
    def sunrise_after(self, jd_start: float) -> Optional[float]:
        try:
            res, tret = self.s.rise_trans(jd_start, self.s.SUN, self.s.CALC_RISE,
                                          (self.place.longitude, self.place.latitude, self.place.elevation_m or 0.0),
                                          0.0, 0.0, self.s.FLG_SWIEPH)
        except Exception:
            return None
        return tret[0] if res == 0 else None

    def day_lord(self, dt: datetime) -> dict[str, Any]:
        """Hindu vara: the weekday runs from sunrise to the next sunrise (not from midnight)."""
        midnight = datetime(dt.year, dt.month, dt.day)
        sr = self.sunrise_after(self.jd_from_local(midnight))
        jd = self.jd_from_local(dt)
        weekday = midnight.weekday()
        sunrise_local: Optional[datetime] = None
        if sr is None:
            note = "No sunrise found (polar day/night): the civil weekday was used."
        else:
            sunrise_local = self.local_from_jd(sr)
            note = None
            if jd < sr:
                weekday = (weekday - 1) % 7
                note = "Before sunrise: the previous weekday's lord rules."
        return {"day_lord": DAY_LORD_BY_WEEKDAY[weekday], "weekday_index": weekday,
                "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][weekday],
                "sunrise_local": sunrise_local.strftime("%Y-%m-%dT%H:%M:%S") if sunrise_local else None, "note": note}


@contextmanager
def calc(place: _Place) -> Iterator[Ctx]:
    """Serialise calculations, apply the ayanamsa, yield a context."""
    s = swe()
    with _LOCK:
        mode = AYANAMSA_MODE[place.options.ayanamsa]
        s.set_sid_mode(mode, 0, 0)
        yield Ctx(place, place.options)


def ayanamsa_values(jd: float) -> dict[str, float]:
    """Ayanamsa (degrees) of every supported mode at a Julian day (UT)."""
    s = swe()
    with _LOCK:
        out = {}
        for name, mode in AYANAMSA_MODE.items():
            s.set_sid_mode(mode, 0, 0)
            out[name] = s.get_ayanamsa_ut(jd)
        return out


def julday_utc(year: int, month: int, day: int, hour: float = 0.0) -> float:
    return swe().julday(year, month, day, hour)


def fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def parse_dt(value: Optional[str], default: Optional[datetime] = None) -> datetime:
    """'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM[:SS]' (local clock). None -> default."""
    if value is None or str(value).strip() == "":
        if default is None:
            raise KPError("A date is required (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS).")
        return default
    text = str(value).strip().replace(" ", "T")
    try:
        dt = datetime.fromisoformat(text if "T" in text else text + "T00:00:00")
    except ValueError as exc:
        raise KPError(f"Bad date '{value}': use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS (local clock time).") from exc
    return dt.replace(tzinfo=None)
