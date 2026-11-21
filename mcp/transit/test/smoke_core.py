"""Live smoke test of the transit engine core (needs PyJHora). Run: .venv/bin/python mcp/transit/test/smoke_core.py"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import transit_engine as e
from state import BirthState

state = BirthState(date_of_birth="1990-05-17", time_of_birth="06:30", latitude=19.0760, longitude=72.8777,
                   timezone="Asia/Kolkata", place_name="Mumbai, IN")

with e._calc(state) as (b, ctx):
    # 1. julian-day convention matches PyJHora
    dt = e.parse_when("2026-11-19T12:00:00", state)
    print("jd check", e.jd_from_local(dt), b.utils.julian_day_number((2026, 11, 19), (12, 0, 0)))
    # 2. natal
    nat = e._natal(state, b, ctx)
    print("natal lagna", e.describe_longitude(nat.lagna)["sign"], "moon", e.SIGNS[nat.moon_sign],
          {e.PLANETS[p]: e.SIGNS[int(l // 30)] for p, l in nat.lon.items()})
    # 3. Saturn ingress scan over 5 years
    a, z = e.jd_utc_from_local(state, ctx, e.parse_when("2026-01-01", state)), e.jd_utc_from_local(state, ctx, e.parse_when("2031-01-01", state))
    t0 = time.time()
    hits = []
    for s in range(12):
        hits += [(jd, d, s) for jd, d in e.find_crossings(e._longitude_target_fn(b, 6, s * 30.0), a, z,
                                                            e.SPEED_BOUND[6], 0.02, e.MAX_STEP[6])]
    hits.sort()
    for jd, d, s in hits:
        print("Saturn crosses", e.SIGNS[s], "boundary", "forward" if d > 0 else "back", e.fmt_dt(e.local_dt_from_utc(state, ctx, jd)))
    print(f"scan time {time.time() - t0:.1f}s")
    # 4. cross-check with PyJHora's own fast function
    t0 = time.time()
    r = e._quiet(b.drik.next_planet_entry_date, b.drik.ephemeris_planet_index(6) if False else 6, ctx.jd + 0, ctx.place)
    print("PyJHora next_planet_entry_date(Saturn) from birth ->", r, f"{time.time() - t0:.1f}s")
