import sys, json
from pathlib import Path
sys.path.insert(0, "mcp/dasha")
import dasha_engine as e
from state import BirthState
from datetime import datetime

st = BirthState(date_of_birth="1990-05-15", time_of_birth="10:30", latitude=19.07, longitude=72.87, timezone="Asia/Kolkata")
m = e.get_mahadashas(st)
print("keys:", list(m.keys()))
rows = m.get("mahadashas") or m.get("periods") or m
print(json.dumps(rows[:2] if isinstance(rows, list) else rows, indent=1, default=str)[:1500])
sp = e.get_sub_periods(st, maha="1", levels_below=1)
print("sub keys:", list(sp.keys()))
print(json.dumps(sp, indent=1, default=str)[:2500])
