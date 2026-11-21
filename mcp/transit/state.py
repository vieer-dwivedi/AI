"""State models for the Transit MCP server.

The MCP client (agent) sends a `BirthState` with every tool call. The server holds no
session data; it reads this state and calculates transits (gochara) from it.

Mirrors `mcp/dasha/state.py` so the same birth dict works for both servers. Dasha-only
options are accepted and ignored. This module is pure Python + pydantic (no PyJHora import),
so it can be validated and tested without the astrology libraries installed.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TransitOptions(BaseModel):
    """Optional calculation settings.

    Unknown keys (the Dasha-only options such as `year_type`, `antardhasa_option`, ...) are
    accepted and ignored, so one birth dict can be sent to both servers.
    """

    model_config = ConfigDict(extra="ignore")

    divisional_chart_factor: int = Field(
        1, ge=1, le=300, description="Natal chart used as the reference: 1 = Rasi (D1), 9 = Navamsa (D9), etc."
    )
    ayanamsa: str = Field(
        "LAHIRI",
        description=(
            "Sidereal ayanamsa, e.g. LAHIRI (default), RAMAN, KP, FAGAN, YUKTESHWAR. PyJHora's own default "
            "(TRUE_PUSHYA) and the other star-based TRUE_* modes crash on some machines' swisseph."
        ),
    )

    @field_validator("ayanamsa")
    @classmethod
    def _check_ayanamsa(cls, v: str) -> str:
        return v.strip().upper()


class BirthState(BaseModel):
    """Birth details + options. This is the 'state' every Transit MCP tool receives."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field("Native", description="Person's name (label only).")
    date_of_birth: date = Field(..., description="Local birth date, ISO format YYYY-MM-DD.")
    time_of_birth: time = Field(..., description="Local birth time, 24-hour HH:MM or HH:MM:SS.")
    place_name: str = Field("Unknown", description="Birth place label, e.g. 'Mumbai, IN'.")
    latitude: float = Field(..., ge=-90, le=90, description="Birth latitude, north positive.")
    longitude: float = Field(..., ge=-180, le=180, description="Birth longitude, east positive.")
    timezone: Optional[str] = Field(
        None,
        description="IANA timezone name, e.g. 'Asia/Kolkata'. The UTC offset (with daylight saving) is worked out for the birth moment.",
    )
    timezone_offset_hours: Optional[float] = Field(
        None, ge=-14, le=14, description="UTC offset in hours at birth (e.g. 5.5). Used if `timezone` is not given."
    )
    elevation_m: Optional[float] = Field(None, description="Birth place elevation in metres (optional).")
    options: TransitOptions = Field(default_factory=TransitOptions)

    @field_validator("name", "place_name")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip() or "Unknown"

    @field_validator("date_of_birth")
    @classmethod
    def _check_year(cls, v: date) -> date:
        if v.year < 1000:
            raise ValueError("date_of_birth year must be 1000 or later")
        return v

    @model_validator(mode="after")
    def _check_timezone(self) -> "BirthState":
        if self.timezone is not None:
            try:
                ZoneInfo(self.timezone)
            except (ZoneInfoNotFoundError, ValueError) as exc:
                raise ValueError(f"Unknown IANA timezone '{self.timezone}'") from exc
        return self

    # --- helpers -------------------------------------------------------------------------
    @property
    def dob_tuple(self) -> tuple[int, int, int]:
        d = self.date_of_birth
        return (d.year, d.month, d.day)

    @property
    def tob_tuple(self) -> tuple[int, int, int]:
        t = self.time_of_birth
        return (t.hour, t.minute, t.second)

    @property
    def local_birth_datetime(self) -> datetime:
        return datetime.combine(self.date_of_birth, self.time_of_birth.replace(microsecond=0))

    def timezone_offset(self) -> Optional[float]:
        """UTC offset in hours at the birth moment, or None if neither timezone field was given.

        A named IANA timezone wins (it includes daylight saving for that date).
        """
        if self.timezone is not None:
            offset = self.local_birth_datetime.replace(tzinfo=ZoneInfo(self.timezone)).utcoffset()
            return offset.total_seconds() / 3600.0 if offset is not None else 0.0
        return self.timezone_offset_hours

    def tz_for(self, when: datetime) -> Optional[float]:
        """UTC offset in hours at an arbitrary local moment (daylight saving aware when `timezone` is named).

        Transit dates are searched years ahead, so the offset at birth is not always the offset then.
        """
        if self.timezone is not None:
            offset = when.replace(tzinfo=ZoneInfo(self.timezone)).utcoffset()
            return offset.total_seconds() / 3600.0 if offset is not None else 0.0
        return self.timezone_offset_hours

    def fingerprint(self) -> str:
        """Stable cache key: identical states give identical keys."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, default=str)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "date_of_birth": self.date_of_birth.isoformat(),
            "time_of_birth": self.time_of_birth.strftime("%H:%M:%S"),
            "place": self.place_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": self.timezone,
            "timezone_offset_hours": self.timezone_offset(),
        }
