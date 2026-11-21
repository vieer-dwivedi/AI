"""State models for the Dasha MCP server.

The MCP client (agent) sends a `BirthState` with every tool call. The server holds no
session data; it only reads this state and calculates dashas from it.

This module is pure Python + pydantic (no PyJHora import) so it can be validated and
tested without the astrology libraries installed.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Names mirror jhora.const enums (checked against PyJHora 5.0.0 source)
YEAR_TYPES = {
    "MEAN_SIDEREAL_YEAR": 1,
    "TRUE_SIDEREAL_YEAR": 2,
    "MEAN_TROPICAL_YEAR": 3,
    "TRUE_TROPICAL_YEAR": 4,  # PyJHora default
    "SAVANA_YEAR": 5,  # 360 days
    "MEAN_LUNAR_YEAR": 6,
    "TRUE_LUNAR_YEAR": 7,
    "GREGORIAN_YEAR": 8,
}

SAVANA_YEAR_METHODS = {
    "MIDPOINT": 0,
    "PROSPECTIVE_0_to_360_JHORA": 1,
    "RETROSPECTIVE_360_to_0": -1,
}


class DashaOptions(BaseModel):
    """Optional calculation settings. Anything left as None uses PyJHora's own default."""

    model_config = ConfigDict(extra="forbid")

    divisional_chart_factor: int = Field(
        1, ge=1, le=300, description="1 = Rasi (D1). 9 = Navamsa (D9), 12 = Dwadasamsa, etc."
    )
    ayanamsa: str = Field(
        "LAHIRI",
        description=(
            "Sidereal ayanamsa, e.g. LAHIRI (default), RAMAN, KP, FAGAN, YUKTESHWAR. Note: PyJHora's own default "
            "(TRUE_PUSHYA) and the other star-based TRUE_* modes crash on this machine's swisseph (true Rahu/Ketu)."
        ),
    )
    year_type: Optional[str] = Field(
        None,
        description=f"Length of a dasha year. One of {list(YEAR_TYPES)}. Default: PyJHora default (TRUE_TROPICAL_YEAR).",
    )
    savana_year_method: Optional[str] = Field(
        None, description=f"Only used with SAVANA_YEAR. One of {list(SAVANA_YEAR_METHODS)}."
    )
    antardhasa_option: Optional[int] = Field(
        None,
        ge=1,
        le=6,
        description=(
            "Order of sub-periods: 1 start from dasha lord forward (default), 2 from dasha lord backward, "
            "3/4 from next lord forward/backward, 5/6 from previous lord forward/backward."
        ),
    )
    chart_method: Optional[int] = Field(None, description="PyJHora chart_method (varga calculation method).")
    star_position_from_moon: Optional[int] = Field(
        None, description="1 = Moon (default). Other values start dasha from a different reference point."
    )
    seed_star: Optional[int] = Field(None, ge=1, le=27, description="Seed nakshatra (1-27) for nakshatra-based dashas.")
    dhasa_starting_planet: Optional[int] = Field(None, description="Planet id the dasha cycle starts from.")
    use_tribhagi_variation: Optional[bool] = Field(None, description="Vimsottari tribhagi (1/3 length) variation.")
    use_rasi_bhukthi_variation: Optional[bool] = Field(None, description="Rasi-based bhukthi variation (L2 only).")
    dhasa_method: Optional[int] = Field(None, description="Method variant for dashas that have several (e.g. padhanadhamsa).")
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Advanced: extra keyword arguments passed straight to the PyJHora dasha function.",
    )

    @field_validator("ayanamsa")
    @classmethod
    def _check_ayanamsa(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("year_type")
    @classmethod
    def _check_year_type(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip().upper()
        if v not in YEAR_TYPES:
            raise ValueError(f"year_type must be one of {list(YEAR_TYPES)}")
        return v

    @field_validator("savana_year_method")
    @classmethod
    def _check_savana_method(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        for name in SAVANA_YEAR_METHODS:
            if name.upper() == v.strip().upper():
                return name
        raise ValueError(f"savana_year_method must be one of {list(SAVANA_YEAR_METHODS)}")

    # --- keyword builders ---------------------------------------------------------------
    def root_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the PyJHora root (Mahadasha) call. Only values that were set."""
        kwargs: dict[str, Any] = {}
        for key in (
            "chart_method",
            "star_position_from_moon",
            "seed_star",
            "dhasa_starting_planet",
            "use_tribhagi_variation",
            "use_rasi_bhukthi_variation",
            "dhasa_method",
        ):
            value = getattr(self, key)
            if value is not None:
                kwargs[key] = value
        if self.antardhasa_option is not None:
            kwargs["antardhasa_option"] = self.antardhasa_option
        kwargs.update(self.extra)
        return kwargs

    def child_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for sub-period calls (kept small: only options that change sub-period maths)."""
        kwargs: dict[str, Any] = {}
        if self.chart_method is not None:
            kwargs["chart_method"] = self.chart_method
        if self.antardhasa_option is not None:
            kwargs["antardhasa_option"] = self.antardhasa_option
        if self.use_rasi_bhukthi_variation is not None:
            kwargs["use_rasi_bhukthi_variation"] = self.use_rasi_bhukthi_variation
        return kwargs


class BirthState(BaseModel):
    """Birth details + options. This is the 'state' every Dasha MCP tool receives."""

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
    options: DashaOptions = Field(default_factory=DashaOptions)

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
