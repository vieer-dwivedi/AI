"""Live smoke test of transit_calendar (needs PyJHora). Run: .venv/bin/python mcp/transit/test/smoke_calendar.py"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import transit_calendar as C
import transit_engine as E
from state import BirthState

state = BirthState(date_of_birth="1990-05-17", time_of_birth="06:30", latitude=19.0760, longitude=72.8777,
                   timezone="Asia/Kolkata", place_name="Mumbai, IN")

def run(label, fn, *a, **k):
    t = time.time()
    try:
        r = fn(state, *a, **k)
        r.pop("state", None)
        print(f"--- {label} ({time.time()-t:.1f}s)")
        print(json.dumps(r, indent=1, default=str)[:2200])
        return r
    except Exception as ex:
        print(f"--- {label} FAILED ({time.time()-t:.1f}s): {type(ex).__name__}: {ex}")

r = run("daily moon", C.get_daily_moon, "2026-11-19T12:00:00")
run("eclipses", C.get_eclipses, "2026-11-19", "2028-02-01")
run("sankranti", C.get_sankrantis, "2026-11-19", "2027-12-31")
run("lagna", C.get_lagna_timeline, "2026-11-19T00:00:00", 24)

# cross-check against PyJHora's own functions
with E._calc(state) as (b, ctx):
    jd = E.jd_from_local(E.parse_when("2026-11-19T12:00:00", state))
    good = set(E._quiet(b.drik.thaaraabalam, jd, ctx.place))
    birth_star = int(E._natal(state, b, ctx).lon[1] // E.NAK_LEN) + 1
    print("PyJHora thaaraabalam lists birth star", birth_star, "as good:", birth_star in good,
          "| ours:", r["tarabalam"]["favourable"])
    cb = set(E._quiet(b.drik.chandrabalam, ctx.jd, ctx.place))
    moon_sign_now = int(E._lon(b, jd - ctx.tz / 24, 1) // 30) + 1
    print("PyJHora chandrabalam good signs", sorted(cb), "now Moon sign", moon_sign_now, "good:", moon_sign_now in cb,
          "| ours:", r["chandrabalam"]["favourable"])
