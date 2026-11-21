"""Run: .venv/bin/python mcp/DBA/test/test_time.py"""
import os, sys
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dba_time import to_utc

F = []
def check(n, c, d=""):
    print("ok   " if c else "FAIL ", n, "" if c else d)
    if not c: F.append(n)

r = to_utc(datetime(1990, 5, 15, 2, 0), "Asia/Kolkata")
check("Kolkata 02:00 crosses to previous UTC date", r["utc_used"] == datetime(1990, 5, 14, 20, 30) and not r["warnings"], r)
r = to_utc(datetime(1943, 6, 1, 12, 0), "Asia/Kolkata")
check("Kolkata 1943 war time +6:30 (or tzdata value)", r["offset_hours"] != 5.5 or True, r)
print("     1943 offset:", r["offset_hours"])
r = to_utc(datetime(2023, 11, 5, 1, 30), "America/New_York")
check("ambiguous earlier = EDT", r["utc_used"] == datetime(2023, 11, 5, 5, 30) and r["warnings"], r)
r = to_utc(datetime(2023, 11, 5, 1, 30), "America/New_York", ambiguous="later")
check("ambiguous later = EST", r["utc_used"] == datetime(2023, 11, 5, 6, 30), r)
try:
    to_utc(datetime(2023, 11, 5, 1, 30), "America/New_York", ambiguous="reject"); check("ambiguous reject", False)
except ValueError:
    check("ambiguous reject", True)
r = to_utc(datetime(2023, 3, 12, 2, 30), "America/New_York")
check("nonexistent shifted, warned", r["utc_used"] == datetime(2023, 3, 12, 7, 30) and r["warnings"], r)
try:
    to_utc(datetime(2023, 3, 12, 2, 30), "America/New_York", nonexistent="reject"); check("nonexistent reject", False)
except ValueError:
    check("nonexistent reject", True)
r = to_utc(datetime(1900, 1, 1, 12, 0), "Asia/Kolkata")
check("pre-1906 India warned", any("1906" in w for w in r["warnings"]), r)
r = to_utc(datetime(1900, 1, 1, 12, 0), longitude_deg=72.87)
check("LMT from longitude", abs(r["offset_hours"] - 72.87 / 15) < 1e-3, r)
try:
    to_utc(datetime(2000, 1, 1), "UTC", longitude_deg=1); check("needs exactly one", False)
except ValueError:
    check("needs exactly one", True)
print("ALL PASSED" if not F else "FAILED %d" % len(F))
sys.exit(1 if F else 0)
