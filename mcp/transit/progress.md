# Transit MCP server: progress

Server for planetary **transits** (gochara), built on PyJHora 5.0.0, mirroring `mcp/dasha`
(FastMCP, `state` = birth details, lazy PyJHora backend, tool wrapper that turns engine errors into tool errors).

## Where I am
**DONE: 37 tools built, registered, live-verified; 22/22 tests pass; real stdio client round-trip verified.**
Remaining work is only the optional items under "Not covered / ideas" below.
Run anything with the project venv: `.venv/bin/python mcp/transit/...` (PyJHora 5.0.0 + swisseph live in `project/.venv`,
NOT in the WSL venv `0mcp-claude/.vevn`; per project rules do not pip install, only edit requirements.txt).

Run the server: `.venv/bin/python mcp/transit/main.py --http` (port 8001) or without `--http` for stdio.
Run the tests (about 105 s): `.venv/bin/python mcp/transit/test/test_transit.py` (or `-m pytest mcp/transit/test/test_transit.py`).

## Session log (newest first)
- **2026-11-19 (later): split from Dasha.** Removed own Vimshottari code and `get_dasha_transit` from transit; added `get_lords_transit`
  (caller supplies the lords) and an optional `dasha_lords` argument on `get_transit_summary`. Tool count unchanged (37). Tests updated.
- **2026-11-19 (resume after crash), finished.**
  - Found that the earlier run crashed after writing `transit_tajaka.py`, `transit_classical.py` and `test/smoke_tajaka.py`, but BEFORE
    registering them, testing them, or recording them here (progress.md still said 13 tools).
  - Smoke-tested both modules. Only bug: backend lacked `ashtakavarga` (fixed in `_backend()`). Registered 13 tools -> 26.
  - Audited coverage, wrote `transit_extra.py` (transit of dasha lords given by the caller, KP, nakshatras/Tara, Sudarshana, divisional charts, lunar phases,
    tithis, combustion periods, day timings, one-call summary, Jaimini) -> 37 tools.
  - Wrote `test/test_transit.py` (22 tests): offline unit tests, cross-checks against PyJHora and known dates, and an in-process
    call of EVERY registered tool (the suite fails if a tool has no case, so new tools cannot go untested; it also asserts no tool
    writes to stdout).
  - Real stdio client (`mcp.client.stdio`) lists 37 tools, calls one, and gets a clean error for a bad planet name.

## Phases
| # | Phase | Status |
|---|-------|--------|
| 1 | Discovery: inventory PyJHora transit functions | done |
| 2 | Verify every candidate function runs (survey scripts) | done |
| 3 | `state.py` + engine modules | done |
| 4 | `transit_server.py` (MCP tools) + `main.py` | done: 37 tools |
| 5 | Tests (offline + real) and fix failures | done: 22/22 |
| 6 | Coverage audit vs PyJHora + final notes | done (see below) |

## Tool list (37), grouped by topic
Positions and classical gochara
- [x] `get_transit_positions`: planets at a date + house from natal Moon / lagna
- [x] `get_gochara`: classical favourable-house table from natal Moon
- [x] `get_vedha`: obstruction of favourable gochara (Sun-Saturn, Moon-Mercury exempt)
- [x] `get_transit_vs_natal`: sign conjunction, rasi aspects, degree orbs
- [x] `get_sudarshana_transit`: house from lagna, Moon and Sun together
- [x] `get_transit_nakshatras`: nakshatra lord, Tara from birth star, Janma/Anujanma/Trijanma, natal planets sharing nakshatra/sign
- [x] `get_planet_conditions`: retrograde/stationary, dignity, combustion, gandanta, graha yuddha
- [x] `get_double_transit`: Jupiter + Saturn influenced houses from lagna or Moon
- [x] `get_transit_summary`: one-call overview (positions, gochara, Sade Sati, double transit, vedha, ashtakavarga, panchanga, next slow-planet ingresses; optional `dasha_lords` from the Dasha MCP adds their transit)

Events and searches
- [x] `get_planet_ingresses`: sign / nakshatra / pada entries incl. retrograde re-entries (Saturn -> Aries 2027-06-03 matches PyJHora within 2 min)
- [x] `get_next_sign_change`
- [x] `get_planet_stations`: station dates + retrograde periods (Mars..Saturn)
- [x] `search_transit_to_natal`: exact conjunction/opposition/trine/square to a natal planet or lagna
- [x] `search_planet_pair`: transit-transit aspects (Jupiter-Saturn conjunction 2040-10-31 in ~4 s; PyJHora took ~20 s)
- [x] `get_planet_returns`: Saturn return, Jupiter return, lunar return ...
- [x] `get_combustion_periods`: asta / udaya dates
- [x] `get_major_transits`: one chronological calendar of slow-planet ingresses, stations and eclipses (about 20 s)
- [x] `get_murthi_nirnaya`: gold/silver/copper/iron murthi of a sign entry

Saturn specials
- [x] `get_sade_sati`: Sade Sati + Dhaiya (Kantaka / Ashtama) windows (Capricorn Moon: 2017-01 .. 2025-03 matches known dates; ~11 s for 50 years)

Ashtakavarga
- [x] `get_ashtakavarga_transit`: BAV / SAV bindus of the transit sign + Kakshya and Kakshya lord
- [x] `get_ashtakavarga_tables`: natal BAV / SAV (totals verified: 48/49/39/54/56/52/39, SAV 337)

Dasha lords, KP, Jaimini, divisional
- [x] `get_lords_transit`: where chosen planets transit (sign, houses from Moon/lagna, gochara verdict, natal placement). For dasha x transit: take the running MD/AD/PD lords from the Dasha MCP and pass them in
- [x] `get_kp_transit`: sign / star / sub / sub-sub lords of transit and natal planets
- [x] `get_jaimini_transit`: Chara Karakas (7 or 8), Karakamsa, Jaimini rasi drishti from transiting planets
- [x] `get_transit_divisional`: transit planets in D2..D60 with houses from that chart's natal lagna / Moon

Panchanga and muhurta
- [x] `get_daily_moon`: Chandrabalam + Tarabalam, next Moon sign / nakshatra change
- [x] `get_panchanga`: vara, tithi, nakshatra, yoga, karana with end times
- [x] `get_lunar_phases`: new / full / quarter Moons (Full Moon 2026-11-24 14:54 UTC matches known)
- [x] `get_tithi_dates`: start/end of any tithis (Ekadashi, Pradosh, Chaturthi, Purnima, Amavasya ...)
- [x] `get_day_timings`: sunrise/sunset/moonrise/moonset, Rahu Kalam, Yamaganda, Gulika, Abhijit, Durmuhurtam, Brahma/Godhuli/Vijaya/Nishita muhurta, Amrit Kalam, choghadiya, 24 horas
- [x] `get_lagna_timeline`: rising signs through a day

Sky events
- [x] `get_eclipses`: solar + lunar, houses from Moon / lagna, local visibility (2027-02-06 annular, 2027-08-02 total match)
- [x] `get_sankrantis`: Sun sign entries (Makara 2027-01-14)
- [x] `get_ayana_points`: equinoxes and solstices

Tajaka / annual
- [x] `get_varshaphal`: solar-return moment, annual chart, Muntha, lord of the year, Tajaka yogas (Ithasala, Eesarpha, Kamboola, Radda, Nakta, Yamaya, Duhphali Kutta, Ishkavala, Induvara)
- [x] `get_monthly_chart`: Maasa Pravesh and 60-hour charts, lord of the month
- [x] `get_sahams`: the 36 Sahams, natal or annual

## PyJHora findings that shape the design
1. **Slow searcher.** `drik.next_conjunction_of_planet_pair` took ~20 s for Jupiter-Saturn (next one is 2040) and
   `drik.next_planet_nakshathra_pravesha_date` did not return within 25 s. An MCP server must never hang on one question, so
   conjunction/aspect, ingress, nakshatra, station, tithi, phase and combustion searches use **our own adaptive scanner + bisection** on
   `sidereal_longitude`, with bounded search windows. PyJHora's own functions are used (and cross-checked in tests) where fast.
2. **Stdout noise.** Several Tajaka functions `print()` debug text. Every PyJHora call runs inside `redirect_stdout`
   (otherwise the stdio transport would be corrupted). A test asserts that no tool writes to stdout.
3. **PyJHora bug.** `tajaka_yoga` uses `charts` without importing it. Worked around by injecting `tajaka_yoga.charts = charts` in `_backend()`.
4. **Annual chart numbering.** `tajaka.annual_chart(..., years=N)` = the N-th year of life: `years=1` is the birth moment,
   `years=27` is the solar return at age 26. The tools take a calendar year / age and convert.
5. **Time convention.** PyJHora Julian days are *local*; UTC = jd - tz/24. Planet ids 0..8 = Sun..Ketu in charts, but
   `sidereal_longitude` wants Swiss-ephemeris ids (use `drik.ephemeris_planet_index`).
6. `chandrabalam(jd, place)` / `nava_thaara` are computed *from the moment passed in*; our own Tara/Chandra formulas agree with PyJHora.
7. `tajaka.planet_aspects_from_chart` is broken for our chart format (KeyError); use `aspects_of_the_planet`.
8. `drik.next_planet_retrograde_change_date` expects a Date tuple (a float jd raises); `next_planet_stationary_duration` returns the
   start/end of the near-stationary window around the station (used in the test as an envelope for our exact station date).
9. (dasha notes live in `mcp/dasha`; this server no longer touches dasha calculation.)
10. `ashtakavarga.get_ashtaka_varga(house_to_planet_list)` returns `(bhinna[8][12], sarva[12], prastara[8][9][12])`; index 7 = Lagna.

## Files
- `state.py`: `BirthState` / `TransitOptions` (ayanamsa, divisional_chart_factor; unknown Dasha options ignored); `tz_for(local_dt)` gives the DST-aware offset at a future date.
- `transit_engine.py`: lazy backend (const, utils, drik, charts, ashtakavarga, tajaka, tajaka_yoga, saham), calc lock + ayanamsa, time helpers, `find_crossings`, natal chart (`_natal`), planet/sign parsing, gochara tables.
- `transit_events.py`: transit / ingress / station / aspect / Sade Sati tools.
- `transit_calendar.py`: daily Moon, eclipses, sankranti, lagna timeline.
- `transit_tajaka.py`: varshaphal, monthly / 60-hour charts, sahams.
- `transit_classical.py`: ashtakavarga transit + kakshya, vedha, murthi, double transit, planet conditions, panchanga, ayana points, planet returns, major-transits calendar.
- `transit_extra.py`: transit of dasha lords given by the caller, KP, nakshatras/Tara, Sudarshana, divisional, lunar phases, tithis, combustion periods, day timings, Jaimini, summary.
- `transit_server.py` / `main.py`: FastMCP, port 8001 (`TRANSIT_MCP_PORT`), stdio or `--http`.
- `test/test_transit.py`: the real test suite. `test/smoke_*.py`: quick live printouts per module.

## Known limitations / things to watch
- Scanner speed bounds (`SPEED_BOUND`) must stay above each planet's true max speed; step = 0.7 * distance / bound.
- Rahu/Ketu are mean nodes in PyJHora's `sidereal_longitude`; the "retrograde" flag for them is informational.
- Sade Sati default window is 15 y back / 35 y forward to keep the call near 10 s; `get_major_transits` is ~20 s.
- Natal reference chart is D1 unless `options.divisional_chart_factor` is changed.
- Tajaka: Khallasara is not in PyJHora, Manahoo is skipped (PyJHora mixes up Mars/Saturn indices), Induvara/Ishkavala are our own computation.
- `sign_id()` in `transit_engine.py` is only used by the tests; fine to keep.
- Tools state facts and classical rules only; they never predict outcomes.

## Not covered / ideas (all optional)
- Tajaka Panchavargiya / Harsha bala inside the annual chart. (Mudda / Varsha Vimshottari dasha belongs to the Dasha MCP, which already has `mudda` and `varsha_vimsottari`.)
- Arudha lagnas and Jaimini Chara dasha transit overlays; Pushkara navamsa / bhaga flags; Bhava Chalit houses.
- Sodhita (reduced) ashtakavarga and Shodhya pindas for transit scoring (PyJHora has `sodhaya_pindas`).
- Natal-chart analysis (yogas, dosha) is deliberately out of scope: it belongs to a natal / dasha server.

## Decisions / notes
- **Separation of tasks (decided 2026-11-19):** Dasha MCP (`mcp/dasha`, port 8000) owns every dasha system, balances, timelines and dasha-lord
  analysis, including the annual ones (Mudda, Varsha Vimshottari). Transit MCP owns planetary movement, gochara, panchanga/muhurta,
  eclipses, Tajaka annual/monthly charts, ashtakavarga transit scoring, KP/Jaimini transit views. The transit server computes NO dasha:
  it only accepts dasha lords as input (`get_lords_transit`, `get_transit_summary(dasha_lords=...)`). An early own-Vimshottari
  implementation here was removed because it duplicated (and slightly disagreed with) the Dasha MCP.
  Intended agent flow for dasha x transit: `get_current_dasha` (Dasha MCP) -> `get_lords_transit` (Transit MCP).
- Port: Dasha server uses 8000; transit server defaults to **8001** (`TRANSIT_MCP_PORT`).
- Dates in/out of every tool are local clock time at the birth place; event rows also carry `jd_utc`.
- Default ayanamsa LAHIRI (Dasha notes PyJHora's own default TRUE_PUSHYA crashes on this machine's swisseph).
- `state` options accept (and ignore) the Dasha-only options so the same birth dict works for both servers.
