"""Live smoke test of transit_classical. Run: .venv/bin/python mcp/transit/test/smoke_classical.py [name ...]"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import transit_classical as K
from state import BirthState

state = BirthState(date_of_birth="1990-05-17", time_of_birth="06:30", latitude=19.0760, longitude=72.8777,
                   timezone="Asia/Kolkata", place_name="Mumbai, IN")
ON = "2026-11-19T18:00:00"
only = set(sys.argv[1:])

def run(label, fn, *a, **k):
    if only and label not in only:
        return
    t = time.time()
    try:
        r = fn(state, *a, **k); r.pop("state", None)
        print(f"--- {label} ({time.time()-t:.1f}s)"); print(json.dumps(r, indent=1, default=str)[:1800])
    except Exception as ex:
        import traceback; traceback.print_exc()
        print(f"--- {label} FAILED ({time.time()-t:.1f}s): {type(ex).__name__}: {ex}")

run("ashtakavarga_transit", K.get_ashtakavarga_transit, ON)
run("ashtakavarga_tables", K.get_ashtakavarga_tables)
run("vedha", K.get_vedha, ON)
run("murthi", K.get_murthi_nirnaya, "Saturn")
run("double_transit", K.get_double_transit, ON)
run("planet_conditions", K.get_planet_conditions, ON)
run("panchanga", K.get_panchanga, ON)
run("ayana", K.get_ayana_points)
run("returns", K.get_planet_returns, "Saturn", 2)
run("major", K.get_major_transits)
