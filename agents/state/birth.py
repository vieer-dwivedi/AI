"""Birth details needed to cast a chart: model, per-user storage, place lookup and console prompts.

How it is used
  main.py                      get_birth_for_user(user_id)  -> saved details are reused, or asked once and saved
                               under the user id (table `birth_profiles` in chat.db)
  orchestrator.ask(..., birth=birth.to_state())   -> the details travel through the graph STATE to every agent
                               that declared `wants_birth=True` (see agents/core/registry.py)
  agents/tests/try_agent_birth.py   uses the same prompts so you can test an agent with any birth details

Everything that talks to the keyboard takes `input_fn` / `print_fn` so tests can script the answers.
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.state import createdb

MIN_YEAR = 1900
DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y")
TIME_FORMATS = ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M%p")
GEOCODER_USER_AGENT = "vedic-astrology-learning-project"


# ----------------------------------------------------------------------------------------------
# Parsing + validation helpers (raise ValueError with a message that is shown to the user)
# ----------------------------------------------------------------------------------------------
def check_date(d: date) -> date:
    if d.year < MIN_YEAR:
        raise ValueError(f"Birth year must be {MIN_YEAR} or later.")
    if d > date.today():
        raise ValueError("Date of birth cannot be in the future.")
    return d


def check_timezone(tz: str) -> str:
    tz = (tz or "").strip()
    try:
        ZoneInfo(tz)
    except Exception:  # ZoneInfoNotFoundError, ValueError (empty / malformed key)
        raise ValueError(f"'{tz}' is not a valid IANA timezone (example: Asia/Kolkata).") from None
    return tz


def parse_date(text: str) -> date:
    text = text.strip()
    for fmt in DATE_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt).date()
        except ValueError:
            continue
        return check_date(parsed)
    raise ValueError("Use DD-MM-YYYY or YYYY-MM-DD, for example 15-05-1990.")


def parse_time(text: str) -> time:
    text = text.strip().upper().replace(".", "")
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    raise ValueError("Use 24-hour HH:MM (for example 06:45 or 18:45) or 12-hour with AM/PM (6:45 PM).")


def parse_float(text: str, low: float, high: float, label: str) -> float:
    try:
        value = float(text.strip())
    except ValueError:
        raise ValueError(f"{label} must be a number.") from None
    if not low <= value <= high:
        raise ValueError(f"{label} must be between {low} and {high}.")
    return value


# ----------------------------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------------------------
class BirthDetails(BaseModel):
    """Everything required to cast a chart. `to_state()` is the dict that goes into agent state."""

    model_config = ConfigDict(extra="forbid")

    name: str = "Native"
    date_of_birth: date
    time_of_birth: time
    place_name: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return v.strip() or "Native"

    @field_validator("place_name")
    @classmethod
    def _place(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Place of birth is required.")
        return v

    @field_validator("date_of_birth")
    @classmethod
    def _dob(cls, v: date) -> date:
        return check_date(v)

    @field_validator("timezone")
    @classmethod
    def _tz(cls, v: str) -> str:
        return check_timezone(v)

    def to_state(self) -> dict[str, Any]:
        """Plain dict in the shape the Dasha agent / MCP server expects (ISO date, 24-hour time)."""
        t = self.time_of_birth
        return {
            "name": self.name,
            "date_of_birth": self.date_of_birth.isoformat(),
            "time_of_birth": t.strftime("%H:%M:%S") if t.second else t.strftime("%H:%M"),
            "place_name": self.place_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": self.timezone,
        }

    def summary(self) -> str:
        s = self.to_state()
        return (
            f"{s['name']}: born {s['date_of_birth']} at {s['time_of_birth']} in {s['place_name']} "
            f"(lat {s['latitude']:.4f}, lon {s['longitude']:.4f}, {s['timezone']})"
        )


# ----------------------------------------------------------------------------------------------
# Storage under the user id
# ----------------------------------------------------------------------------------------------
def load_birth(user_id: str) -> Optional[BirthDetails]:
    createdb.init_db()
    data = createdb.load_birth_profile(user_id)
    if not data:
        return None
    try:
        return BirthDetails(**data)
    except ValueError:  # damaged / outdated row: treat as "not saved" so the user is asked again
        return None


def save_birth(user_id: str, birth: BirthDetails) -> None:
    createdb.init_db()
    createdb.save_birth_profile(user_id, birth.model_dump(mode="json"))


# ----------------------------------------------------------------------------------------------
# Place lookup (OpenStreetMap through geopy; timezone from coordinates, offline)
# ----------------------------------------------------------------------------------------------
def timezone_for(latitude: float, longitude: float) -> Optional[str]:
    try:
        from timezonefinder import TimezoneFinder

        return TimezoneFinder().timezone_at(lat=latitude, lng=longitude)
    except Exception:
        return None


def lookup_place(query: str, timeout: int = 10) -> Optional[dict[str, Any]]:
    """Place name -> {address, latitude, longitude, timezone}. Needs internet. None when nothing matches."""
    from geopy.geocoders import Nominatim

    hit = Nominatim(user_agent=GEOCODER_USER_AGENT, timeout=timeout).geocode(query, exactly_one=True, language="en")
    if hit is None:
        return None
    tz = timezone_for(hit.latitude, hit.longitude)
    if tz is None:
        return None
    return {"address": hit.address, "latitude": hit.latitude, "longitude": hit.longitude, "timezone": tz}


# ----------------------------------------------------------------------------------------------
# Console prompts
# ----------------------------------------------------------------------------------------------
def _yes_no(question: str, input_fn: Callable[[str], str], print_fn: Callable[..., None]) -> bool:
    while True:
        answer = input_fn(f"{question} (y/n): ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print_fn("Please type y or n.")


def _ask(prompt, parse, input_fn, print_fn, default=None, shown=None):
    """Ask until `parse` accepts the answer. Enter keeps `default` (if there is one)."""
    hint = f" [{shown if shown is not None else default}]" if default is not None else ""
    while True:
        raw = input_fn(f"{prompt}{hint}: ").strip()
        if not raw:
            if default is not None:
                return default
            print_fn("This is required.")
            continue
        try:
            return parse(raw)
        except ValueError as exc:
            print_fn(f"  {exc}")


def _ask_place(existing, input_fn, print_fn, lookup):
    """Returns (place_name, latitude, longitude, timezone)."""
    while True:
        place = _ask(
            "Place of birth (city, state, country)", str, input_fn, print_fn,
            default=existing.place_name if existing else None,
        )
        if existing and place == existing.place_name:
            return place, existing.latitude, existing.longitude, existing.timezone

        found = None
        try:
            found = lookup(place)
        except Exception as exc:  # no internet, rate limit, ...
            print_fn(f"  Could not look up '{place}' online ({type(exc).__name__}).")
        else:
            if found is None:
                print_fn(f"  No match for '{place}'. Try adding the state and country.")

        if found:
            print_fn(f"  Found: {found['address']}")
            print_fn(f"         latitude {found['latitude']:.4f}, longitude {found['longitude']:.4f}, timezone {found['timezone']}")
            if _yes_no("Is this the right place?", input_fn, print_fn):
                return place, found["latitude"], found["longitude"], found["timezone"]
            print_fn("  OK, try again and be more specific (add state / country).")
            continue

        if not _yes_no("Enter latitude, longitude and timezone yourself?", input_fn, print_fn):
            continue
        lat = _ask("Latitude (north positive, e.g. 19.0760)", lambda s: parse_float(s, -90, 90, "Latitude"), input_fn, print_fn)
        lon = _ask("Longitude (east positive, e.g. 72.8777)", lambda s: parse_float(s, -180, 180, "Longitude"), input_fn, print_fn)
        guess = timezone_for(lat, lon)
        tz = _ask("Timezone (IANA name, e.g. Asia/Kolkata)", check_timezone, input_fn, print_fn, default=guess)
        return place, lat, lon, tz


def prompt_birth_details(
    existing: Optional[BirthDetails] = None,
    default_name: Optional[str] = None,
    input_fn: Optional[Callable[[str], str]] = None,
    print_fn: Callable[..., None] = print,
    lookup: Optional[Callable[[str], Optional[dict]]] = None,
) -> BirthDetails:
    """Ask for every detail needed for a chart. With `existing`, pressing Enter keeps the current value."""
    input_fn = input_fn or input
    lookup = lookup or lookup_place
    ex = existing

    print_fn("\nBirth details (needed to cast the chart)." + (" Press Enter to keep a value in [brackets]." if ex else ""))
    name = _ask("Name", lambda s: s.strip(), input_fn, print_fn, default=(ex.name if ex else default_name))
    dob = _ask(
        "Date of birth (DD-MM-YYYY or YYYY-MM-DD)", parse_date, input_fn, print_fn,
        default=ex.date_of_birth if ex else None, shown=ex.date_of_birth.isoformat() if ex else None,
    )
    tob = _ask(
        "Time of birth (24-hour HH:MM, or like 6:45 PM; as exact as you can)", parse_time, input_fn, print_fn,
        default=ex.time_of_birth if ex else None, shown=ex.to_state()["time_of_birth"] if ex else None,
    )
    place, lat, lon, tz = _ask_place(ex, input_fn, print_fn, lookup)

    return BirthDetails(
        name=name, date_of_birth=dob, time_of_birth=tob, place_name=place, latitude=lat, longitude=lon, timezone=tz
    )


def update_birth_for_user(
    user_id: str,
    existing: Optional[BirthDetails] = None,
    input_fn: Optional[Callable[[str], str]] = None,
    print_fn: Callable[..., None] = print,
    lookup: Optional[Callable[[str], Optional[dict]]] = None,
) -> BirthDetails:
    """Ask for (new) birth details and save them under the user id."""
    birth = prompt_birth_details(existing, user_id, input_fn, print_fn, lookup)
    save_birth(user_id, birth)
    print_fn(f"Birth details saved for '{user_id}'.")
    return birth


def get_birth_for_user(
    user_id: str,
    input_fn: Optional[Callable[[str], str]] = None,
    print_fn: Callable[..., None] = print,
    lookup: Optional[Callable[[str], Optional[dict]]] = None,
) -> BirthDetails:
    """Saved details for this user id (confirmed), otherwise ask for them once and save them."""
    input_fn = input_fn or input
    saved = load_birth(user_id)
    if saved:
        print_fn(f"Birth details on file for '{user_id}':\n  {saved.summary()}")
        if _yes_no("Use these birth details?", input_fn, print_fn):
            return saved
        return update_birth_for_user(user_id, saved, input_fn, print_fn, lookup)
    print_fn(f"No birth details on file for '{user_id}'. They are needed to cast your chart.")
    return update_birth_for_user(user_id, None, input_fn, print_fn, lookup)
