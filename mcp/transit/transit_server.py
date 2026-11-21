"""Transit (gochara) MCP server (PyJHora). Every tool receives the birth `state` and returns JSON.

Run as a script (NOT with `python -m`), so this folder, not the project root, is on sys.path
and the folder name `mcp` does not shadow the official `mcp` SDK:

    python mcp/transit/main.py --http     # streamable-http on 127.0.0.1:8001/mcp
    python mcp/transit/main.py            # stdio
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional

from mcp.server.fastmcp import FastMCP

import transit_engine as engine
import transit_calendar as cal
import transit_events as ev
import transit_tajaka as taj
import transit_classical as cls
import transit_extra as xt
from state import BirthState

mcp = FastMCP("transit")

DATE_HELP = "Local date/time at the birth place: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS."


def tool(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Register `fn` as an MCP tool; engine errors become clean tool errors."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except engine.TransitError as exc:
            raise ValueError(str(exc)) from exc

    return mcp.tool()(wrapper)


@tool
def get_transit_positions(state: BirthState, on_date: Optional[str] = None, planets: Optional[list[str]] = None) -> dict[str, Any]:
    """Where the planets are at a date (default now): sign, degree, nakshatra/pada, speed, retrograde, and the house
    each one occupies counted from the natal Moon and the natal lagna. `planets`: e.g. ["Saturn", "Jupiter"] (default all 9).
    on_date: local date/time at the birth place, YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS."""
    return ev.get_positions(state, on_date, planets)


@tool
def get_gochara(state: BirthState, on_date: Optional[str] = None) -> dict[str, Any]:
    """Classical Parashari gochara table at a date: every planet's house from the natal Moon and whether it is in its
    traditionally favourable set of houses. Facts only; vedha and ashtakavarga are not applied."""
    return ev.get_gochara(state, on_date)


@tool
def get_transit_vs_natal(state: BirthState, on_date: Optional[str] = None, orb: float = 5.0) -> dict[str, Any]:
    """Transiting planets against natal planets at a date: same-sign conjunctions, rasi aspects (7th for all; Mars 4/8,
    Jupiter 5/9, Saturn 3/10, nodes 5/9) and degree conjunctions/oppositions within `orb` degrees (max 15)."""
    return ev.get_transit_vs_natal(state, on_date, orb)


@tool
def get_planet_ingresses(
    state: BirthState, planet: str = "Saturn", kind: str = "sign", start: Optional[str] = None,
    end: Optional[str] = None, limit: int = 20,
) -> dict[str, Any]:
    """Moments a planet enters a new sign, nakshatra or pada (`kind` = sign | nakshatra | pada), retrograde re-entries
    included. Default window: now to +2 years (Moon: +60 days). Slow planets can be scanned for centuries."""
    return ev.get_ingresses(state, planet, kind, start, end, limit)


@tool
def get_next_sign_change(state: BirthState, planet: str = "Jupiter", on_date: Optional[str] = None) -> dict[str, Any]:
    """Current position of a planet and the next time it changes sign (forward or retrograde)."""
    return ev.get_next_sign_change(state, planet, on_date)


@tool
def get_planet_stations(state: BirthState, planet: str = "Jupiter", start: Optional[str] = None, end: Optional[str] = None) -> dict[str, Any]:
    """Stationary-retrograde and stationary-direct dates for Mars, Mercury, Jupiter, Venus or Saturn, and the retrograde
    periods between them (default window: now to +3 years)."""
    return ev.get_stations(state, planet, start, end)


@tool
def search_transit_to_natal(
    state: BirthState, transit_planet: str = "Saturn", natal_point: str = "Moon", aspects: Optional[list[str]] = None,
    start: Optional[str] = None, end: Optional[str] = None,
) -> dict[str, Any]:
    """Exact dates a transiting planet conjoins / opposes (optionally trine, square) a natal planet or the natal 'Lagna'.
    aspects: any of conjunction, opposition, trine, square (default conjunction + opposition). Default window: now to +5 years
    (+1 year for fast planets)."""
    return ev.search_transit_to_natal(state, transit_planet, natal_point, aspects, start, end)


@tool
def search_planet_pair(
    state: BirthState, planet_a: str = "Jupiter", planet_b: str = "Saturn", aspects: Optional[list[str]] = None,
    start: Optional[str] = None, end: Optional[str] = None,
) -> dict[str, Any]:
    """Exact dates two transiting planets conjoin / oppose (optionally trine, square) each other, e.g. the Jupiter-Saturn
    conjunction. Default window: now to +3 years; up to 100 years for two slow planets (Jupiter, Saturn, Rahu, Ketu)."""
    return ev.search_planet_pair(state, planet_a, planet_b, aspects, start, end)


@tool
def get_sade_sati(state: BirthState, on_date: Optional[str] = None, years_back: int = 15, years_forward: int = 35) -> dict[str, Any]:
    """Is Sade Sati (Saturn 12th/1st/2nd from the natal Moon sign) or a Dhaiya (4th/8th) running at a date, plus every
    such Saturn stay in the surrounding window (default 15 years back, 35 forward) with exact dates. Takes ~10 s."""
    return ev.get_sade_sati(state, on_date, years_back, years_forward)


@tool
def get_daily_moon(state: BirthState, on_date: Optional[str] = None) -> dict[str, Any]:
    """Moon's sign/nakshatra at a moment with Chandrabalam (house from natal Moon, Chandrashtama flag), Tarabalam
    (tara from the birth nakshatra, good/bad) and when the Moon next changes sign and nakshatra. Muhurta-style inputs."""
    return cal.get_daily_moon(state, on_date)


@tool
def get_eclipses(state: BirthState, start: Optional[str] = None, end: Optional[str] = None, kind: str = "both") -> dict[str, Any]:
    """Solar and/or lunar eclipses in a window (default now to +2 years, max 100): type, time of maximum, sign/nakshatra,
    house from natal Moon and lagna, and whether (and how much) it is visible at the birth place. kind = solar | lunar | both."""
    return cal.get_eclipses(state, start, end, kind)


@tool
def get_sankrantis(state: BirthState, start: Optional[str] = None, end: Optional[str] = None) -> dict[str, Any]:
    """Every Sankranti (Sun entering a sidereal sign, e.g. Makara Sankranti) in a window (default next 12 months, max 20 years)."""
    return cal.get_sankrantis(state, start, end)


@tool
def get_lagna_timeline(state: BirthState, on_date: Optional[str] = None, hours: float = 24.0) -> dict[str, Any]:
    """The rising sign (lagna) at the birth place from a moment for `hours` (default 24, max 72): start/end of each lagna
    and its nakshatra/pada at the start."""
    return cal.get_lagna_timeline(state, on_date, hours)


# ------------------------------------------------------------------ Tajaka / Varshaphal
@tool
def get_varshaphal(state: BirthState, age: Optional[int] = None, year: Optional[int] = None, include_yogas: bool = True) -> dict[str, Any]:
    """Tajaka annual chart (Varsha Pravesh): exact solar-return moment, annual lagna and planets (houses from the annual
    lagna), Muntha, lord of the year and Tajaka yogas (Ithasala, Eesarpha, Kamboola, Radda, Nakta, Yamaya, Duhphali Kutta,
    Ishkavala, Induvara). `age` = completed years (default current age) or give a calendar `year`."""
    return taj.get_varshaphal(state, age, year, include_yogas)


@tool
def get_monthly_chart(state: BirthState, age: Optional[int] = None, year: Optional[int] = None, month: int = 1,
                      sixty_hour: Optional[int] = None) -> dict[str, Any]:
    """Tajaka monthly chart (Maasa Pravesh) for month 1..12 of the year of life (1 = the solar return itself) with the lord
    of the month; give `sixty_hour` 1..12 for the 60-hour (2.5-day) chart inside that month."""
    return taj.get_monthly_chart(state, age, year, month, sixty_hour)


@tool
def get_sahams(state: BirthState, age: Optional[int] = None, year: Optional[int] = None) -> dict[str, Any]:
    """The 36 Sahams (Arabic parts, e.g. Punya, Vivaha, Karma) of the natal chart, or of the Varsha chart if age/year is given."""
    return taj.get_sahams(state, age, year)


# ------------------------------------------------------------------ Ashtakavarga / Vedha / classical rules
@tool
def get_ashtakavarga_transit(state: BirthState, on_date: Optional[str] = None) -> dict[str, Any]:
    """Ashtakavarga scoring of the transit at a date: per planet Sun..Saturn the Bhinnashtakavarga bindus (0-8) and
    Sarvashtakavarga bindus of the sign it occupies, plus its Kakshya (3d45' strip), the Kakshya lord and whether it gave a bindu."""
    return cls.get_ashtakavarga_transit(state, on_date)


@tool
def get_ashtakavarga_tables(state: BirthState) -> dict[str, Any]:
    """Natal Bhinnashtakavarga (each planet + Lagna, bindus per sign) and Sarvashtakavarga, the tables behind transit scoring."""
    return cls.get_ashtakavarga_tables(state)


@tool
def get_vedha(state: BirthState, on_date: Optional[str] = None) -> dict[str, Any]:
    """Classical Vedha: which favourable gochara planets are obstructed by a planet in the paired vedha house
    (Sun-Saturn and Moon-Mercury exempt)."""
    return cls.get_vedha(state, on_date)


@tool
def get_murthi_nirnaya(state: BirthState, planet: str = "Saturn", start: Optional[str] = None, end: Optional[str] = None) -> dict[str, Any]:
    """Murthi Nirnaya (gold/silver/copper/iron form) of each sign entry of Mars, Jupiter, Saturn, Rahu or Ketu, decided by the
    Moon's house from the natal Moon at the moment of entry. Default: Saturn, next 10 years."""
    return cls.get_murthi_nirnaya(state, planet, start, end)


@tool
def get_double_transit(state: BirthState, on_date: Optional[str] = None, reference: str = "lagna") -> dict[str, Any]:
    """Double transit: houses (from natal `lagna` or `moon`) influenced by Jupiter, by Saturn and by both at once."""
    return cls.get_double_transit(state, on_date, reference)


@tool
def get_planet_conditions(state: BirthState, on_date: Optional[str] = None) -> dict[str, Any]:
    """Per planet at a date: retrograde/stationary, dignity (exalted, debilitated, own), combustion, gandanta and graha yuddha."""
    return cls.get_planet_conditions(state, on_date)


@tool
def get_panchanga(state: BirthState, on_date: Optional[str] = None) -> dict[str, Any]:
    """Panchanga at a moment: vara, tithi, nakshatra, yoga and karana, each with its end time (local)."""
    return cls.get_panchanga(state, on_date)


@tool
def get_ayana_points(state: BirthState, start: Optional[str] = None, end: Optional[str] = None) -> dict[str, Any]:
    """Equinoxes and solstices in a window (default next 12 months)."""
    return cls.get_ayana_points(state, start, end)


@tool
def get_planet_returns(state: BirthState, planet: str = "Saturn", count: int = 3, start: Optional[str] = None) -> dict[str, Any]:
    """Next `count` returns of a transiting planet to its natal longitude (Saturn return, Jupiter return, lunar return, ...)."""
    return cls.get_planet_returns(state, planet, count, start)


@tool
def get_major_transits(state: BirthState, start: Optional[str] = None, end: Optional[str] = None,
                       planets: Optional[list[str]] = None, include_eclipses: bool = True) -> dict[str, Any]:
    """One chronological calendar (default now to +2 years): sign ingresses and stations of Jupiter, Saturn, Rahu, Ketu
    (or `planets`) plus eclipses, each tagged with the house from the natal Moon. Takes ~20 s."""
    return cls.get_major_transits(state, start, end, planets, include_eclipses)


# ------------------------------------------------------------------ Dasha x transit, KP, nakshatras, calendar extras
@tool
def get_lords_transit(state: BirthState, planets: list[str], on_date: Optional[str] = None) -> dict[str, Any]:
    """Transit of chosen planets (sign, houses from natal Moon/lagna, gochara verdict, natal placement). Use it for dasha x transit:
    get the running Mahadasha/Antardasha/Pratyantara lords from the Dasha MCP, then pass them here. This server computes no dashas."""
    return xt.get_lords_transit(state, planets, on_date)


@tool
def get_kp_transit(state: BirthState, on_date: Optional[str] = None) -> dict[str, Any]:
    """KP lords (sign lord, star lord, sub-lord, sub-sub-lord) of every transiting planet and of the natal planets/lagna.
    Use options.ayanamsa = KP for classical KP work."""
    return xt.get_kp_transit(state, on_date)


@tool
def get_transit_nakshatras(state: BirthState, on_date: Optional[str] = None) -> dict[str, Any]:
    """Each transiting planet's nakshatra and lord, Tara from the birth star (favourable or not), Janma/Anujanma/Trijanma group,
    and the natal planets sharing its nakshatra or sign."""
    return xt.get_transit_nakshatras(state, on_date)


@tool
def get_sudarshana_transit(state: BirthState, on_date: Optional[str] = None) -> dict[str, Any]:
    """Houses of every transiting planet counted from natal lagna, Moon and Sun together (Sudarshana Chakra view)."""
    return xt.get_sudarshana_transit(state, on_date)


@tool
def get_transit_divisional(state: BirthState, on_date: Optional[str] = None, factors: Optional[list[int]] = None) -> dict[str, Any]:
    """Transit planets and transit lagna in divisional charts (default D9 and D10; any of 2,3,4,5,6,7,8,9,10,11,12,16,20,24,27,30,40,45,60,
    at most 6), with houses from that chart's natal lagna and Moon."""
    return xt.get_transit_divisional(state, on_date, factors)


@tool
def get_lunar_phases(state: BirthState, start: Optional[str] = None, end: Optional[str] = None) -> dict[str, Any]:
    """New Moon, first quarter, Full Moon, last quarter moments in a window (default 60 days) with the Moon's sign, nakshatra and
    house from natal Moon/lagna."""
    return xt.get_lunar_phases(state, start, end)


@tool
def get_tithi_dates(state: BirthState, tithis: Optional[list[int]] = None, start: Optional[str] = None, end: Optional[str] = None) -> dict[str, Any]:
    """Start and end of chosen tithis (1-15 Shukla, 16-30 Krishna; default 11 and 26 = Ekadashi; 15 Purnima, 30 Amavasya,
    13/28 Pradosh, 4/19 Chaturthi) in a window (default 45 days, max 1 year)."""
    return xt.get_tithi_dates(state, tithis, start, end)


@tool
def get_combustion_periods(state: BirthState, planet: str = "Venus", start: Optional[str] = None, end: Optional[str] = None) -> dict[str, Any]:
    """Asta (combust) and udaya (rise) dates of Mars, Mercury, Jupiter, Venus or Saturn in a window (default 2 years)."""
    return xt.get_combustion_periods(state, planet, start, end)


@tool
def get_day_timings(state: BirthState, on_date: Optional[str] = None) -> dict[str, Any]:
    """Sunrise/sunset/moonrise/moonset, Rahu Kalam, Yamaganda, Gulika, Abhijit, Durmuhurtam, Brahma/Godhuli/Vijaya/Nishita muhurta,
    Amrit Kalam, Gauri choghadiyas and the 24 horas (with the current one) for a date at the birth place."""
    return xt.get_day_timings(state, on_date)


@tool
def get_transit_summary(state: BirthState, on_date: Optional[str] = None, dasha_lords: Optional[list[str]] = None) -> dict[str, Any]:
    """One-call overview at a date: positions, gochara, Sade Sati status, double transit, vedha, ashtakavarga scoring, panchanga, the transit of
    `dasha_lords` if you pass them (from the Dasha MCP) and the next sign changes of Jupiter, Saturn, Rahu, Ketu. Start here for a general transit reading."""
    return xt.get_transit_summary(state, on_date, dasha_lords)


@tool
def get_jaimini_transit(state: BirthState, on_date: Optional[str] = None, scheme: int = 7) -> dict[str, Any]:
    """Jaimini transit: natal Chara Karakas (7- or 8-karaka scheme) and Karakamsa, with every transiting planet's Jaimini rasi
    drishti onto the natal lagna, karaka planets and Karakamsa, and conjunctions with them."""
    return xt.get_jaimini_transit(state, on_date, scheme)
