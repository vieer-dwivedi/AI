"""State models for the KP MCP server.

Every tool receives one of these from the MCP client (agent); the server holds no session data.

  * `KPState`     : birth details + KP options. Same field names as `mcp/dasha` and `mcp/transit`, so one birth dict works
                    for all three servers (unknown option keys are accepted and ignored).
  * `MomentState` : a moment and a place (no birth data) for horary, ruling planets and transit snapshots.

Pure Python + pydantic (no swisseph import): can be validated and tested without the astrology libraries.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time
from datetime import date as _Date, time as _Time
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AYANAMSAS = ("KP_NEW", "KP_OLD", "LAHIRI", "RAMAN")
_AYANAMSA_ALIASES = {
    "KP": "KP_NEW", "KPNA": "KP_NEW", "KP_NEWCOMB": "KP_NEW", "KP-NEW": "KP_NEW", "KP_NEW": "KP_NEW", "KRISHNAMURTI": "KP_OLD",
    "KP_OLD": "KP_OLD", "KP-OLD": "KP_OLD", "KPOLD": "KP_OLD", "LAHIRI": "LAHIRI", "CHITRAPAKSHA": "LAHIRI", "RAMAN": "RAMAN",
}


class KPOptions(BaseModel):
    """Calculation settings. Unknown keys (the Dasha/Transit-only options) are accepted and ignored."""

    model_config = ConfigDict(extra="ignore")

    ayanamsa: str = Field(
        "KP_NEW",
        description=("Sidereal ayanamsa. KP_NEW (default; Swiss Ephemeris Krishnamurti VP291, matches the published 'KP New'), "
                     "KP_OLD (Swiss Ephemeris Krishnamurti, about Lahiri minus 5m49s), LAHIRI, RAMAN. "
                     "Do not mix ayanamsas between charts."),
    )
    node_type: str = Field("mean", description="Rahu/Ketu: 'mean' (default, endorsed by KP forums) or 'true'. A node's sub lord can differ.")
    house_system: str = Field("placidus", description="'placidus' (KP standard). Beyond the polar circle Placidus is undefined: the server falls back to Porphyry and says so.")
    dasha_year_days: float = Field(365.25, gt=300, lt=400, description="Days in a dasha year. KP software uses 365.25 (default); some use 360 or 365.2422.")
    include_outer: bool = Field(False, description="Add Uranus, Neptune and Pluto to the chart (KP textbooks use 12 bodies; they are not used in significators by default).")
    nodes_as_agents: bool = Field(True, description="Rahu/Ketu also signify the houses of the planets they represent (conjunction, sign lord, star lord).")
    significator_levels: str = Field("ABCD", description="Which levels count as 'signifies' in judgments, e.g. 'ABCD' (default) or 'ABC'.")
    significator_method: str = Field("kp4level", description="Significator view returned by get_significators: 'kp4level' (default, Krishnamurti A-D), 'four_step' (Gondhalekar / Shri Bhatt 4-step theory, primary/secondary) or 'six_fold' (A-D plus planets in the sub of occupant and owner). Judgment tools still use A-D.")
    retrograde_rule: str = Field("report_only", description="'report_only' (default: facts only), 'ksk_horary_cusp' (cuspal sub lord retrograde or in a retrograde star -> delay flag) or 'strict' (that, plus retrograde / retrograde-star dasha lords flag a period as delayed). Never denies a matter by itself.")
    include_type_e: bool = Field(False, description="Add type-E significators (planets conjunct / aspecting the significators) as a SEPARATE block in get_significators; never mixed into A-D.")
    type_e_basis: str = Field("orb", description="'orb' (default): conjunction / aspect within `type_e_orb_deg` of the exact point; 'sign': sign relation only (as for node agents).")
    type_e_orb_deg: float = Field(200.0 / 60.0, gt=0, le=15, description="Orb in degrees for type E (default 3d20', as in the 4-step school).")

    cusp_orb_deg: Optional[float] = Field(None, ge=0, le=15, description="Cusp proximity (bhava chalit) rule, off by default (null or 0). With an orb (usual 3.3333 = 3d20'), a planet standing that close behind a cusp also signifies the next house and a planet just past a cusp is a strong significator of it. Adds `near_cusp` facts and `signifies_with_near_cusp`; A-D stay unchanged unless `cusp_proximity_apply` is set.")
    cusp_proximity_apply: bool = Field(False, description="Needs `cusp_orb_deg`. Feeds the cusp-proximity shift into A-D: the planet also counts as an occupant (level B) of the next house, so planets in its star count at level A there. Judgments, fruitful-significator selection and period search then use it.")

    csl_extension: str = Field("off", description="Not pure KP, off by default. 'own_bhava' (a CSL with no planet in its star also signifies the bhava of its own cusp), 'sub_chain' (the CSL's sub lord is also an outright significator: its occupied and owned houses are added) or 'both'. Adds a separate `csl_extension` block to CSL reports and an `extended_reading` next to the unchanged verdict in judge_event.")

    @field_validator("csl_extension")
    @classmethod
    def _csl_ext(cls, v: str) -> str:
        m = v.strip().lower().replace("-", "_").replace(" ", "_")
        m = {"none": "off", "false": "off", "ksk": "own_bhava", "own": "own_bhava", "kar": "sub_chain", "sub": "sub_chain", "all": "both"}.get(m, m)
        if m not in ("off", "own_bhava", "sub_chain", "both"):
            raise ValueError("csl_extension must be 'off', 'own_bhava', 'sub_chain' or 'both'")
        return m

    @model_validator(mode="after")
    def _cusp_apply(self) -> "KPOptions":
        if self.cusp_proximity_apply and not self.cusp_orb_deg:
            raise ValueError("cusp_proximity_apply needs cusp_orb_deg (for example 3.3333 for 3d20')")
        return self

    @field_validator("ayanamsa")
    @classmethod
    def _ayanamsa(cls, v: str) -> str:
        key = v.strip().upper().replace(" ", "_")
        if key not in _AYANAMSA_ALIASES:
            raise ValueError(f"ayanamsa must be one of {list(AYANAMSAS)} (aliases: KP, KPNA, KRISHNAMURTI)")
        return _AYANAMSA_ALIASES[key]

    @field_validator("node_type")
    @classmethod
    def _node(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("mean", "true"):
            raise ValueError("node_type must be 'mean' or 'true'")
        return v

    @field_validator("house_system")
    @classmethod
    def _house(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("placidus", "porphyry"):
            raise ValueError("house_system must be 'placidus' or 'porphyry'")
        return v

    @field_validator("type_e_basis")
    @classmethod
    def _tebasis(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("orb", "sign"):
            raise ValueError("type_e_basis must be 'orb' or 'sign'")
        return v

    @field_validator("retrograde_rule")
    @classmethod
    def _retro(cls, v: str) -> str:
        m = v.strip().lower().replace("-", "_").replace(" ", "_")
        m = {"report": "report_only", "none": "report_only", "ksk": "ksk_horary_cusp", "horary": "ksk_horary_cusp"}.get(m, m)
        if m not in ("report_only", "ksk_horary_cusp", "strict"):
            raise ValueError("retrograde_rule must be 'report_only', 'ksk_horary_cusp' or 'strict'")
        return m

    @field_validator("significator_method")
    @classmethod
    def _method(cls, v: str) -> str:
        m = v.strip().lower().replace("-", "_").replace(" ", "_")
        m = {"4level": "kp4level", "kp": "kp4level", "4step": "four_step", "6fold": "six_fold"}.get(m, m)
        if m not in ("kp4level", "four_step", "six_fold"):
            raise ValueError("significator_method must be 'kp4level', 'four_step' or 'six_fold'")
        return m

    @field_validator("significator_levels")
    @classmethod
    def _levels(cls, v: str) -> str:
        v = "".join(sorted(set(v.strip().upper())))
        if not v or any(c not in "ABCD" for c in v):
            raise ValueError("significator_levels must use the letters A, B, C, D")
        return v


class _Place(BaseModel):
    model_config = ConfigDict(extra="forbid")

    place_name: str = Field("Unknown", description="Place label, e.g. 'Mumbai, IN'.")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude, north positive.")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude, east positive.")
    timezone: Optional[str] = Field(None, description="IANA timezone, e.g. 'Asia/Kolkata' (daylight saving aware).")
    timezone_offset_hours: Optional[float] = Field(None, ge=-14, le=14, description="UTC offset in hours (e.g. 5.5). Used if `timezone` is not given.")
    elevation_m: Optional[float] = Field(None, description="Elevation in metres (optional, used for sunrise).")
    options: KPOptions = Field(default_factory=KPOptions)

    @field_validator("place_name")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip() or "Unknown"

    @model_validator(mode="after")
    def _check_tz(self) -> "_Place":
        if self.timezone is not None:
            try:
                ZoneInfo(self.timezone)
            except (ZoneInfoNotFoundError, ValueError) as exc:
                raise ValueError(f"Unknown IANA timezone '{self.timezone}'") from exc
        if self.timezone is None and self.timezone_offset_hours is None:
            raise ValueError("Give `timezone` (IANA name) or `timezone_offset_hours`.")
        return self

    def tz_at(self, when: datetime) -> float:
        """UTC offset in hours at a local moment (daylight saving aware when a named timezone is given)."""
        if self.timezone is not None:
            off = when.replace(tzinfo=ZoneInfo(self.timezone)).utcoffset()
            return off.total_seconds() / 3600.0 if off is not None else 0.0
        return float(self.timezone_offset_hours or 0.0)

    def fingerprint(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, default=str)


class KPState(_Place):
    """Birth details + options: the 'state' of every natal KP tool."""

    name: str = Field("Native", description="Person's name (label only).")
    date_of_birth: date = Field(..., description="Local birth date, YYYY-MM-DD.")
    time_of_birth: time = Field(..., description="Local birth time, 24-hour HH:MM or HH:MM:SS. KP is extremely sensitive to it: cuspal sub lords change every few minutes.")

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return v.strip() or "Native"

    @field_validator("date_of_birth")
    @classmethod
    def _year(cls, v: date) -> date:
        if not 1000 <= v.year <= 3000:
            raise ValueError("date_of_birth year must be between 1000 and 3000")
        return v

    @property
    def local_datetime(self) -> datetime:
        return datetime.combine(self.date_of_birth, self.time_of_birth.replace(microsecond=0))

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name, "date_of_birth": self.date_of_birth.isoformat(),
            "time_of_birth": self.time_of_birth.strftime("%H:%M:%S"), "place": self.place_name,
            "latitude": self.latitude, "longitude": self.longitude, "timezone": self.timezone,
            "timezone_offset_hours": self.tz_at(self.local_datetime),
            "ayanamsa": self.options.ayanamsa, "node_type": self.options.node_type,
        }


class MomentState(_Place):
    """A moment and place (no birth data): horary, ruling planets, transit snapshots.

    `date` and `time` are LOCAL clock time at the place. Leave both out to use the current moment.
    """

    date: Optional[_Date] = Field(None, description="Local date YYYY-MM-DD. Omit for 'now'.")
    time: Optional[_Time] = Field(None, description="Local time HH:MM[:SS]. Omit for 'now' (or midnight if a date is given).")

    @property
    def local_datetime(self) -> datetime:
        if self.date is None and self.time is None:
            return now_at(self)
        d = self.date or now_at(self).date()
        t = self.time or _Time(0, 0, 0)
        return datetime.combine(d, t.replace(microsecond=0))

    def summary(self) -> dict[str, Any]:
        dt = self.local_datetime
        return {"datetime_local": dt.strftime("%Y-%m-%dT%H:%M:%S"), "place": self.place_name,
                "latitude": self.latitude, "longitude": self.longitude, "timezone": self.timezone,
                "timezone_offset_hours": self.tz_at(dt), "ayanamsa": self.options.ayanamsa,
                "node_type": self.options.node_type}


def now_at(place: _Place) -> datetime:
    """Current local clock time at the place."""
    if place.timezone is not None:
        return datetime.now(ZoneInfo(place.timezone)).replace(tzinfo=None, microsecond=0)
    from datetime import timedelta, timezone
    off = float(place.timezone_offset_hours or 0.0)
    return (datetime.now(timezone.utc) + timedelta(hours=off)).replace(tzinfo=None, microsecond=0)
