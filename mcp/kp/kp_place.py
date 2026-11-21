"""Place helper (progress.md backlog item 21): coordinates and timezone for a place, so callers do not invent them.

* `timezone_info`: offline. IANA zone from latitude / longitude (timezonefinder), UTC offset at a local date / time from the system
  tz database (historical rules included), with flags for daylight saving, an ambiguous local time (clock repeated) and a
  non-existent one (clock skipped).
* `lookup_place`: needs the internet (OpenStreetMap Nominatim through geopy); results always carry the offline timezone.
Facts only: no guess is made when the lookup fails.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from kp_core import KPError

_TF = None


def _tzfinder():
    global _TF
    if _TF is None:
        from timezonefinder import TimezoneFinder
        _TF = TimezoneFinder()
    return _TF


def _check_coords(lat: float, lon: float) -> None:
    if not -90.0 <= lat <= 90.0:
        raise KPError("latitude must be between -90 and 90.")
    if not -180.0 <= lon <= 180.0:
        raise KPError("longitude must be between -180 and 180.")


def timezone_info(lat: float, lon: float, when: Optional[datetime] = None) -> dict[str, Any]:
    _check_coords(lat, lon)
    name = _tzfinder().timezone_at(lat=lat, lng=lon)
    if not name:
        raise KPError("No timezone found for these coordinates (open sea?). Give an IANA timezone name yourself.")
    out: dict[str, Any] = {"latitude": lat, "longitude": lon, "timezone": name}
    if when is None:
        return out
    z = ZoneInfo(name)
    a, b = when.replace(tzinfo=z, fold=0), when.replace(tzinfo=z, fold=1)
    oa, ob = a.utcoffset(), b.utcoffset()
    # round trip through UTC detects a local time that does not exist (clock skipped forward)
    back = a.astimezone(timezone.utc).astimezone(z).replace(tzinfo=None)
    ambiguous = oa != ob and back == when
    nonexistent = back != when
    dst = a.dst()
    out.update({"local_datetime": when.strftime("%Y-%m-%dT%H:%M:%S"),
                "utc_offset_hours": oa.total_seconds() / 3600.0 if oa is not None else None,
                "is_dst": bool(dst and dst.total_seconds() != 0), "ambiguous_local_time": bool(ambiguous),
                "nonexistent_local_time": bool(nonexistent),
                "alternative_utc_offset_hours": ob.total_seconds() / 3600.0 if ambiguous and ob is not None else None})
    notes = []
    if ambiguous:
        notes.append("This local clock time happened twice (daylight saving ends): the first offset is used; ask which one applies.")
    if nonexistent:
        notes.append("This local clock time did not exist (daylight saving starts): check the birth record.")
    if when.year < 1950 and name.startswith("Asia/"):
        notes.append("Before 1950 many places used local mean time, war time or other offsets: verify the offset against the birth record.")
    if notes:
        out["notes"] = notes
    return out


def _nominatim(query: str, limit: int) -> list[dict[str, Any]]:
    try:
        from geopy.geocoders import Nominatim
        g = Nominatim(user_agent="kp-mcp-place-helper", timeout=8)
        res = g.geocode(query, exactly_one=False, limit=limit, language="en")
    except Exception as exc:  # network, rate limit, missing package
        raise KPError(f"Place lookup needs internet access and failed ({type(exc).__name__}). Give latitude, longitude and timezone directly.") from exc
    return [{"name": r.address, "latitude": float(r.latitude), "longitude": float(r.longitude)} for r in (res or [])]


def lookup_place(query: str, limit: int = 3, geocoder: Optional[Callable[[str, int], list[dict[str, Any]]]] = None) -> dict[str, Any]:
    q = (query or "").strip()
    if not q:
        raise KPError("`query` must not be empty.")
    if not 1 <= limit <= 10:
        raise KPError("limit must be 1 to 10.")
    rows = (geocoder or _nominatim)(q, limit)
    out = []
    for r in rows:
        try:
            tz = timezone_info(r["latitude"], r["longitude"])["timezone"]
        except KPError:
            tz = None
        out.append({**r, "timezone": tz})
    return {"query": q, "count": len(out), "places": out,
            "note": ("Coordinates come from OpenStreetMap Nominatim (decimal degrees, east and north positive). Check that the first match is the "
                     "place meant; use `get_timezone_info` with the birth date to check the UTC offset."),
            "source": "OpenStreetMap Nominatim via geopy; timezone from timezonefinder"}
