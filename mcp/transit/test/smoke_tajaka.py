"""Live smoke test of transit_tajaka. Run: .venv/bin/python mcp/transit/test/smoke_tajaka.py"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import transit_tajaka as T
import transit_engine as E
from state import BirthState

state = BirthState(date_of_birth="1990-05-17", time_of_birth="06:30", latitude=19.0760, longitude=72.8777,
                   timezone="Asia/Kolkata", place_name="Mumbai, IN")

def run(label, fn, *a, **k):
    t = time.time()
    try:
        r = fn(state, *a, **k); r.pop("state", None)
        print(f"--- {label} ({time.time()-t:.1f}s)"); print(json.dumps(r, indent=1, default=str)[:2600]); return r
    except Exception as ex:
        import traceback; traceback.print_exc()
        print(f"--- {label} FAILED ({time.time()-t:.1f}s): {type(ex).__name__}: {ex}")

r = run("varshaphal age 36", T.get_varshaphal, 36)
run("monthly 3", T.get_monthly_chart, 36, None, 3)
run("sixty hour", T.get_monthly_chart, 36, None, 3, 2)
run("sahams natal", T.get_sahams)

# cross-check: our own solar-return finder vs PyJHora
with E._calc(state) as (b, ctx):
    nat = E._natal(state, b, ctx)
    target = nat.lon[0]
    a = E.jd_utc_from_local(state, ctx, E.parse_when("2026-05-10", state)); z = E.jd_utc_from_local(state, ctx, E.parse_when("2026-05-25", state))
    hits = E.find_crossings(E._longitude_target_fn(b, 0, target), a, z, E.SPEED_BOUND[0], 0.02, E.MAX_STEP[0])
    print("own solar return:", E.fmt_dt(E.local_dt_from_utc(state, ctx, hits[0][0])), "| PyJHora:", r["solar_return_local"])
