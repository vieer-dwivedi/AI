"""Live smoke test of transit_events (needs PyJHora). Run: .venv/bin/python mcp/transit/test/smoke_events.py"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import transit_events as T
from state import BirthState

state = BirthState(date_of_birth="1990-05-17", time_of_birth="06:30", latitude=19.0760, longitude=72.8777,
                   timezone="Asia/Kolkata", place_name="Mumbai, IN")

def run(label, fn, *a, **k):
    t = time.time()
    try:
        r = fn(state, *a, **k)
        print(f"--- {label} ({time.time()-t:.1f}s)")
        print(json.dumps(r, indent=1, default=str)[:1500])
    except Exception as ex:
        print(f"--- {label} FAILED ({time.time()-t:.1f}s): {type(ex).__name__}: {ex}")

run("positions", T.get_positions, "2026-11-19T12:00:00", ["Saturn", "Jupiter"])
run("gochara", T.get_gochara, "2026-11-19")
run("ingress Saturn sign", T.get_ingresses, "Saturn", "sign", "2026-11-19", "2031-01-01")
run("ingress Moon nakshatra", T.get_ingresses, "Moon", "nakshatra", "2026-11-19", "2026-11-22")
run("next sign change Jupiter", T.get_next_sign_change, "Jupiter", "2026-11-19")
run("stations Jupiter", T.get_stations, "Jupiter", "2026-11-19", "2029-11-19")
run("Saturn->natal Moon", T.search_transit_to_natal, "Saturn", "Moon", None, "2026-11-19", "2032-01-01")
run("Jupiter-Saturn pair", T.search_planet_pair, "Jupiter", "Saturn", None, "2026-11-19", "2046-01-01")
run("sade sati", T.get_sade_sati, "2026-11-19")
run("transit vs natal", T.get_transit_vs_natal, "2026-11-19", 3.0)
