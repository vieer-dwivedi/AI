"""Transit MCP tests (live PyJHora + swisseph). Run with the project venv from the project root:

    .venv/bin/python -m pytest mcp/transit/test/test_transit.py -q
    .venv/bin/python mcp/transit/test/test_transit.py            # same checks without pytest

Sections: offline (no ephemeris needed), cross-checks against PyJHora / known dates, and an in-process call of EVERY
registered MCP tool (a tool without a case in TOOL_CASES fails the suite, so new tools cannot go untested).
"""

from __future__ import annotations

import asyncio
import io
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import transit_engine as E  # noqa: E402
import transit_events as V  # noqa: E402
import transit_calendar as C  # noqa: E402
import transit_classical as K  # noqa: E402
import transit_extra as X  # noqa: E402
import transit_tajaka as T  # noqa: E402
from state import BirthState  # noqa: E402

STATE = BirthState(date_of_birth="1990-05-17", time_of_birth="06:30", latitude=19.0760, longitude=72.8777,
                   timezone="Asia/Kolkata", place_name="Mumbai, IN")
STATE_DICT = STATE.model_dump(mode="json")
ON = "2026-11-19T18:00:00"


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    except Exception as e:  # wrong exception type
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, nothing raised")


# ------------------------------------------------------------------------------------------ offline
def test_state_validation():
    raises(ValueError, BirthState, date_of_birth="1990-05-17", time_of_birth="06:30", latitude=95, longitude=10)
    raises(ValueError, BirthState, date_of_birth="1990-05-17", time_of_birth="06:30", latitude=10, longitude=10, timezone="Mars/Olympus")
    s = BirthState(date_of_birth="1990-05-17", time_of_birth="06:30", latitude=10, longitude=10, timezone="Asia/Kolkata",
                   options={"ayanamsa": "raman", "year_type": "ignored-dasha-option"})
    assert s.options.ayanamsa == "RAMAN"


def test_dst_aware_offset():
    s = BirthState(date_of_birth="1990-05-17", time_of_birth="06:30", latitude=51.5, longitude=-0.1, timezone="Europe/London")
    assert s.tz_for(datetime(2027, 1, 15, 12)) == 0.0 and s.tz_for(datetime(2027, 7, 15, 12)) == 1.0


def test_planet_and_sign_parsing():
    assert E.planet_id("shani") == 6 and E.planet_id("Guru") == 4 and E.planet_id(8) == 8
    raises(E.TransitError, E.planet_id, "Pluto")
    assert E.sign_id("Cap") == 9
    raises(E.TransitError, E.sign_id, "Nowhere")


def test_vedha_table_is_consistent():
    for p, table in K.VEDHA.items():
        assert set(table) <= set(E.GOCHARA_GOOD[p]), f"vedha for a house that is not favourable: planet {p}"
        assert set(table) == set(E.GOCHARA_GOOD[p]), f"favourable house without a vedha house: planet {p}"


def test_kp_lords():
    r = X._kp_lords(0.001)
    assert (r["sign_lord"], r["star_lord"], r["sub_lord"], r["sub_sub_lord"]) == ("Mars", "Ketu", "Ketu", "Ketu")
    assert X._kp_lords(13.3333 - 0.0001)["sub_lord"] == "Mercury"   # last sub of Ashwini
    r = X._kp_lords(13.34)
    assert r["star_lord"] == "Venus" and r["sub_lord"] == "Venus"   # first sub of Bharani


def test_jaimini_aspects():
    assert X.jaimini_aspect_signs(0) == [4, 7, 10]      # Aries: Leo, Scorpio, Aquarius (not Taurus)
    assert X.jaimini_aspect_signs(1) == [3, 6, 9]       # Taurus: Cancer, Libra, Capricorn (not Aries)
    assert X.jaimini_aspect_signs(2) == [5, 8, 11]      # Gemini: Virgo, Sagittarius, Pisces
    for s in range(12):                                  # aspect is mutual
        for t in X.jaimini_aspect_signs(s):
            assert s in X.jaimini_aspect_signs(t)


# ------------------------------------------------------------------------------------------ real: cross-checks
def test_saturn_aries_ingress_matches_pyjhora_and_known_date():
    res = V.get_ingresses(STATE, "Saturn", "sign", "2027-05-01", "2027-07-01")
    ev = [e for e in res["events"] if e["to"] == "Aries"]
    assert ev and ev[0]["datetime_local"].startswith("2027-06-03"), ev
    with E._calc(STATE) as (b, ctx):
        jd = E.jd_from_local(datetime(2026, 11, 19, 12))
        theirs = E._quiet(b.drik.next_planet_entry_date, 6, jd, ctx.place)[0]
        ours = E.jd_from_local(datetime.fromisoformat(ev[0]["datetime_local"]))
        assert abs(theirs - ours) < 120 / 86400, (theirs, ours)


def test_saturn_station_inside_pyjhora_stationary_window():
    res = V.get_stations(STATE, "Saturn", "2026-11-01", "2027-01-31")
    direct = [s for s in res["stations"] if "direct" in s["station"].lower()]
    assert direct, res["stations"]
    with E._calc(STATE) as (b, ctx):
        jd = E.jd_from_local(datetime(2026, 11, 19, 12))
        (y1, m1, d1, _), (y2, m2, d2, _) = E._quiet(b.drik.next_planet_stationary_duration, 6, jd, ctx.place)
        lo, hi = datetime(y1, m1, d1) - timedelta(days=2), datetime(y2, m2, d2) + timedelta(days=2)
        when = datetime.fromisoformat(direct[0]["datetime_local"])
        assert lo <= when <= hi, (lo, when, hi)


def test_lords_transit_takes_lords_from_caller():
    r = X.get_lords_transit(STATE, ["Jupiter", "Sun", "mercury", "Jupiter"], ON)
    assert [p["planet"] for p in r["planets"]] == ["Jupiter", "Sun", "Mercury"]
    raises(E.TransitError, X.get_lords_transit, STATE, [], ON)


def test_own_solar_return_matches_pyjhora():
    r = T.get_varshaphal(STATE, age=36, include_yogas=False)
    with E._calc(STATE) as (b, ctx):
        nat = E._natal(STATE, b, ctx)
        a = E.jd_utc_from_local(STATE, ctx, E.parse_when("2026-05-10", STATE))
        z = E.jd_utc_from_local(STATE, ctx, E.parse_when("2026-05-25", STATE))
        hit = E.find_crossings(E._longitude_target_fn(b, 0, nat.lon[0]), a, z, E.SPEED_BOUND[0], 0.02, E.MAX_STEP[0])[0][0]
        own = E.local_dt_from_utc(STATE, ctx, hit)
    assert abs((own - datetime.fromisoformat(r["solar_return_local"])).total_seconds()) < 5


def test_ashtakavarga_totals():
    t = K.get_ashtakavarga_tables(STATE)
    totals = {k: sum(v) for k, v in t["bhinnashtakavarga"].items() if k != "Lagna"}
    assert totals == {"Sun": 48, "Moon": 49, "Mars": 39, "Mercury": 54, "Jupiter": 56, "Venus": 52, "Saturn": 39}, totals
    assert t["sav_total"] == 337 and sum(t["sarvashtakavarga"]) == 337


def test_sade_sati_known_dates_capricorn_moon():
    r = V.get_sade_sati(STATE, ON, 12, 5)
    rows = r["periods_in_window"]
    assert r["natal_moon_sign"] == "Capricorn" and rows
    assert any(p["from_local"].startswith("2017") for p in rows), [p["from_local"] for p in rows]
    assert any(p["to_local"].startswith("2025-03") for p in rows), [p["to_local"] for p in rows]


def test_known_eclipses():
    r = C.get_eclipses(STATE, "2027-01-01", "2027-12-31", "solar")
    days = {e["datetime_local"][:10]: e["type"] for e in r["events"]}
    assert "2027-02-06" in days and "2027-08-02" in days, days
    assert "annular" in days["2027-02-06"].lower() and "total" in days["2027-08-02"].lower(), days


def test_makara_sankranti_2027():
    r = C.get_sankrantis(STATE, "2027-01-01", "2027-02-01")
    assert r["events"] and r["events"][0]["datetime_local"].startswith("2027-01-1"), r["events"]


def test_full_moon_nov_2026():
    r = X.get_lunar_phases(STATE, "2026-11-20", "2026-11-30")
    fm = [e for e in r["events"] if e["phase"].startswith("Full")][0]
    utc = datetime.fromisoformat(fm["datetime_local"]) - timedelta(hours=5, minutes=30)
    assert abs((utc - datetime(2026, 11, 24, 14, 53)).total_seconds()) < 600, fm


def test_panchanga_internal_consistency():
    r = K.get_panchanga(STATE, ON)
    assert 1 <= r["tithi"]["number"] <= 30 and r["vara"].startswith("Thursday"), r
    assert r["tithi"]["ends_local"] and r["nakshatra"]["ends_local"]


def test_combustion_symmetry_venus_2026():
    r = X.get_combustion_periods(STATE, "Venus", "2026-09-01", "2026-12-31")
    ev = {e["event"].split()[0]: e["datetime_local"] for e in r["events"]}
    assert ev["goes"].startswith("2026-10-1") and ev["emerges"].startswith("2026-10-2"), r["events"]


def test_ayanamsa_changes_longitude_and_state_is_not_shared():
    a = V.get_positions(STATE, ON, ["Saturn"])["planets"][0]["longitude"]
    other = BirthState(date_of_birth="1980-01-01", time_of_birth="01:00", latitude=28.6, longitude=77.2, timezone="Asia/Kolkata",
                       options={"ayanamsa": "RAMAN"})
    b = V.get_positions(other, ON, ["Saturn"])["planets"][0]["longitude"]
    assert abs(a - b) > 0.5          # different ayanamsa really changes longitude
    assert V.get_positions(STATE, ON, ["Saturn"])["planets"][0]["longitude"] == a   # and switching back restores it


# ------------------------------------------------------------------------------------------ real: every MCP tool
TOOL_CASES = {
    "get_transit_positions": {"on_date": ON, "planets": ["Saturn", "Jupiter"]},
    "get_gochara": {"on_date": ON},
    "get_transit_vs_natal": {"on_date": ON},
    "get_planet_ingresses": {"planet": "Mars", "start": "2026-11-19", "end": "2027-03-01"},
    "get_next_sign_change": {"planet": "Jupiter", "on_date": ON},
    "get_planet_stations": {"planet": "Mercury", "start": "2026-11-19", "end": "2027-06-01"},
    "search_transit_to_natal": {"transit_planet": "Mars", "natal_point": "Moon", "start": "2026-11-19", "end": "2027-12-31"},
    "search_planet_pair": {"planet_a": "Mars", "planet_b": "Venus", "start": "2026-11-19", "end": "2027-12-31"},
    "get_sade_sati": {"on_date": ON, "years_back": 3, "years_forward": 5},
    "get_daily_moon": {"on_date": ON},
    "get_eclipses": {"start": "2026-11-19", "end": "2027-12-31"},
    "get_sankrantis": {"start": "2026-11-19", "end": "2027-03-01"},
    "get_lagna_timeline": {"on_date": ON, "hours": 12},
    "get_varshaphal": {"age": 36},
    "get_monthly_chart": {"age": 36, "month": 3},
    "get_sahams": {},
    "get_ashtakavarga_transit": {"on_date": ON},
    "get_ashtakavarga_tables": {},
    "get_vedha": {"on_date": ON},
    "get_murthi_nirnaya": {"planet": "Jupiter", "start": "2026-11-19", "end": "2030-01-01"},
    "get_double_transit": {"on_date": ON, "reference": "moon"},
    "get_planet_conditions": {"on_date": ON},
    "get_panchanga": {"on_date": ON},
    "get_ayana_points": {"start": "2026-11-19", "end": "2027-12-31"},
    "get_planet_returns": {"planet": "Jupiter", "count": 1},
    "get_major_transits": {"start": "2026-11-19", "end": "2027-06-30", "planets": ["Saturn", "Rahu"]},
    "get_lords_transit": {"planets": ["Jupiter", "Sun", "Mercury"], "on_date": ON},
    "get_kp_transit": {"on_date": ON},
    "get_transit_nakshatras": {"on_date": ON},
    "get_sudarshana_transit": {"on_date": ON},
    "get_transit_divisional": {"on_date": ON, "factors": [9, 10, 12]},
    "get_lunar_phases": {"start": "2026-11-19", "end": "2027-01-01"},
    "get_tithi_dates": {"tithis": [11, 30], "start": "2026-11-19", "end": "2027-01-01"},
    "get_combustion_periods": {"planet": "Mercury", "start": "2026-11-19", "end": "2027-12-31"},
    "get_day_timings": {"on_date": ON},
    "get_transit_summary": {"on_date": ON, "dasha_lords": ["Jupiter", "Sun"]},
    "get_jaimini_transit": {"on_date": ON, "scheme": 8},
}


def _server():
    import transit_server
    return transit_server


def _call(name, args):
    return asyncio.run(_server().mcp.call_tool(name, {"state": STATE_DICT, **args}))


def test_every_registered_tool_has_a_case():
    names = {t.name for t in asyncio.run(_server().mcp.list_tools())}
    assert names == set(TOOL_CASES), f"untested: {sorted(names - set(TOOL_CASES))}; stale: {sorted(set(TOOL_CASES) - names)}"


def test_every_tool_runs_and_keeps_stdout_clean():
    buf = io.StringIO()
    failures = []
    with redirect_stdout(buf):
        for name, args in TOOL_CASES.items():
            try:
                _call(name, args)
            except Exception as exc:
                failures.append(f"{name}: {type(exc).__name__}: {str(exc)[:200]}")
    assert not failures, "\n".join(failures)
    assert buf.getvalue() == "", f"a tool wrote to stdout (would corrupt stdio transport): {buf.getvalue()[:200]!r}"


def test_tool_errors_are_clean():
    for name, args in [("get_next_sign_change", {"planet": "Pluto"}), ("get_planet_ingresses", {"kind": "weekday"}),
                       ("get_varshaphal", {"age": 500}), ("get_tithi_dates", {"tithis": [31]}),
                       ("get_transit_divisional", {"factors": [13]}), ("get_vedha", {"on_date": "not-a-date"})]:
        try:
            _call(name, args)
        except Exception as exc:
            assert "Traceback" not in str(exc) and str(exc), (name, exc)
        else:
            raise AssertionError(f"{name} {args} should have failed")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    bad = 0
    for n, f in tests:
        try:
            f()
            print("PASS", n, flush=True)
        except Exception as e:
            bad += 1
            print("FAIL", n, f"{type(e).__name__}: {str(e)[:300]}", flush=True)
    print(f"{len(tests) - bad}/{len(tests)} passed")
    sys.exit(1 if bad else 0)
