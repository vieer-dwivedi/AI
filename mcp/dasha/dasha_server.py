"""Dasha MCP server (PyJHora). Every tool receives the birth `state` and returns JSON.

Run as a script (NOT with `python -m`), so this folder, not the project root, is on sys.path
and the folder name `mcp` does not shadow the official `mcp` SDK:

    python mcp/dasha_server.py            # stdio transport (default, for MCP clients / agents)
    python mcp/dasha_server.py http       # streamable-http on 127.0.0.1:8000/mcp
"""

from __future__ import annotations

import functools
import sys
from typing import Any, Callable, Optional

from mcp.server.fastmcp import FastMCP

import dasha_engine as engine
from state import BirthState

mcp = FastMCP("dasha")

SYSTEM_HELP = "Dasha system name, e.g. 'vimsottari' (default), 'yogini', 'ashtottari', 'chara', 'kalachakra'. See list_dasha_systems."
DATE_HELP = "Local date/time at the birth place: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS. Omit for 'now'."


def tool(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Register `fn` as an MCP tool; engine errors become clean tool errors."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except engine.DashaError as exc:
            raise ValueError(str(exc)) from exc

    return mcp.tool()(wrapper)


@tool
def list_dasha_systems() -> dict[str, Any]:
    """List every dasha system available (graha, rasi, nakshatra, karaka, special, annual), with category,
    whether it is conditional, and its classical lifespan. Needs no birth data."""
    return engine.list_systems()


@tool
def get_dasha_basis(state: BirthState) -> dict[str, Any]:
    """What dashas are computed from: lagna, Moon sign/nakshatra/pada, nakshatra lord, and all planet positions
    (sign, degree, house, nakshatra) in the chosen chart."""
    return engine.get_basis(state)


@tool
def check_dasha_applicability(state: BirthState) -> dict[str, Any]:
    """Which conditional dashas (ashtottari, dwadasottari, panchottari, ...) apply to this chart."""
    return engine.check_applicability(state)


@tool
def get_dasha_balance(state: BirthState, system: str = "vimsottari") -> dict[str, Any]:
    """Balance of the running dasha at birth (elapsed and remaining) for the given system."""
    return engine.get_balance(state, system)


@tool
def get_mahadashas(state: BirthState, system: str = "vimsottari") -> dict[str, Any]:
    """All Mahadasha periods with start/end dates, duration and the age at start/end."""
    return engine.get_mahadashas(state, system)


@tool
def get_sub_periods(
    state: BirthState,
    system: str = "vimsottari",
    maha: str = "1",
    path: Optional[list[str]] = None,
    occurrence: int = 1,
    levels_below: int = 1,
) -> dict[str, Any]:
    """Drill into a period. `maha` = Mahadasha index (1-based, e.g. "3") or lord name (e.g. "Jupiter", with
    `occurrence` when the lord repeats). `path` = further lords to descend, e.g. ["Saturn", "Mercury"] to get the
    sub-periods of that Pratyantara. Returns the next level (Antardasha -> Pratyantara -> Sookshma -> Prana -> Deha);
    `levels_below` (1-3) expands several levels at once."""
    return engine.get_sub_periods(state, system, maha, path, occurrence, levels_below)


@tool
def get_current_dasha(
    state: BirthState, system: str = "vimsottari", on_date: Optional[str] = None, depth: int = 3
) -> dict[str, Any]:
    """Running dasha ladder at a date (default today): lord, dates, % elapsed and time remaining per level.
    depth 1-6 (Maha ... Deha). on_date: local date/time at the birth place, YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS."""
    return engine.get_current(state, system, on_date, depth)


@tool
def get_dasha_at_age(state: BirthState, system: str = "vimsottari", age: float = 30.0, depth: int = 3) -> dict[str, Any]:
    """Running dasha ladder at a given age in years."""
    return engine.get_at_age(state, system, age, depth)


@tool
def get_next_dasha_changes(
    state: BirthState, system: str = "vimsottari", on_date: Optional[str] = None, level: int = 2, count: int = 5
) -> dict[str, Any]:
    """The running period at `level` (1 Maha, 2 Antara, ...) and the next `count` periods after it."""
    return engine.get_upcoming(state, system, on_date, level, count)


@tool
def get_dasha_timeline(
    state: BirthState,
    system: str = "vimsottari",
    start: Optional[str] = None,
    end: Optional[str] = None,
    depth: int = 2,
    limit: int = 200,
) -> dict[str, Any]:
    """All periods at `depth` that overlap a date range (default: birth to 10 years later)."""
    return engine.get_timeline(state, system, start, end, depth, limit)


@tool
def search_dasha_periods(
    state: BirthState,
    system: str = "vimsottari",
    lord: str = "Jupiter",
    level: int = 2,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Every period of a given lord at a level (e.g. all Jupiter Antardashas) within a date range
    (default: birth to 100 years later)."""
    return engine.search_periods(state, system, lord, level, start, end, limit)


@tool
def compare_dasha_systems(
    state: BirthState, systems: Optional[list[str]] = None, on_date: Optional[str] = None, depth: int = 2
) -> dict[str, Any]:
    """Running dasha in several systems at once (default: vimsottari, yogini, ashtottari, chara, narayana, kalachakra)."""
    return engine.compare_systems(state, systems, on_date, depth)


@tool
def analyze_dasha_lords(
    state: BirthState, system: str = "vimsottari", on_date: Optional[str] = None, depth: int = 3
) -> dict[str, Any]:
    """Facts about the running dasha lords: sign, house from lagna/Moon, nakshatra, dignity, houses owned,
    conjunctions, aspects, and the relationship between each level (e.g. 6/8, 2/12). No predictions."""
    return engine.analyze_lords(state, system, on_date, depth)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("http", "streamable-http"):
        mcp.run(transport="streamable-http")
    else:
        mcp.run()
