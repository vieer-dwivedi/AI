"""Real-PyJHora check of every dasha system (slow). Writes progress to stdout.

Run with the venv python:  .venv/bin/python mcp/dasha/test/real_check.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dasha_engine as e  # noqa: E402
from state import BirthState  # noqa: E402

st = BirthState(date_of_birth="1990-05-15", time_of_birth="10:30", latitude=19.07, longitude=72.87, timezone="Asia/Kolkata")


def step(label, fn):
    t = time.time()
    try:
        r = fn()
        print(f"OK   {label} ({time.time() - t:.1f}s)", flush=True)
        return r
    except Exception as ex:
        print(f"FAIL {label}: {type(ex).__name__}: {str(ex)[:160]}", flush=True)


bal = step("balance", lambda: e.get_balance(st))
if bal:
    print("   ", bal["running_at_birth"]["lord"], bal["balance_at_birth"], flush=True)
cur = step("current d5", lambda: e.get_current(st, on="2026-11-18", depth=5))
if cur:
    print("   ", cur["running"], flush=True)
step("applicability", lambda: e.check_applicability(st))
step("analyze_lords", lambda: e.analyze_lords(st, on="2026-11-18"))
step("upcoming", lambda: e.get_upcoming(st, on="2026-11-18"))
step("sub_periods", lambda: e.get_sub_periods(st, maha="Jupiter", levels_below=2))
step("search Jupiter AD", lambda: e.search_periods(st, lord="Jupiter", level=2))

bad = []
for s in e.list_systems()["systems"]:
    n = s["name"]
    t = time.time()
    try:
        e.get_current(st, n, on="1990-12-01" if s["category"] == "annual" else "2026-11-18", depth=3)
        print(f"ok   {n} {time.time() - t:.1f}s", flush=True)
    except Exception as ex:
        bad.append(n)
        print(f"FAIL {n}: {str(ex)[:140]}", flush=True)
print("DONE failed:", bad, flush=True)
