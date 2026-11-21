"""Live smoke test of transit_extra. Run: .venv/bin/python mcp/transit/test/smoke_extra.py [label ...]"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import transit_extra as X
from state import BirthState

state = BirthState(date_of_birth="1990-05-17", time_of_birth="06:30", latitude=19.0760, longitude=72.8777,
                   timezone="Asia/Kolkata", place_name="Mumbai, IN")
ON = "2026-11-19T18:00:00"
only = set(sys.argv[1:])
LIMIT = 1800

def run(label, fn, *a, **k):
    if only and label not in only:
        return
    t = time.time()
    try:
        r = fn(state, *a, **k); r.pop("state", None)
        print(f"--- {label} ({time.time()-t:.1f}s)"); print(json.dumps(r, indent=1, default=str)[:LIMIT])
    except Exception as ex:
        import traceback; traceback.print_exc()
        print(f"--- {label} FAILED ({time.time()-t:.1f}s): {type(ex).__name__}: {ex}")

run("lords", X.get_lords_transit, ["Jupiter", "Sun"], ON)
run("kp", X.get_kp_transit, ON)
run("nakshatras", X.get_transit_nakshatras, ON)
run("sudarshana", X.get_sudarshana_transit, ON)
run("divisional", X.get_transit_divisional, ON, [9, 10])
run("phases", X.get_lunar_phases, "2026-11-19", "2027-01-10")
run("tithis", X.get_tithi_dates, [11, 26], "2026-11-19", "2027-01-10")
run("combustion", X.get_combustion_periods, "Venus", "2026-01-01", "2028-12-31")
run("timings", X.get_day_timings, ON)
run("summary", X.get_transit_summary, ON)
