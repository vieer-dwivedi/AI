"""Run: .venv/bin/python mcp/DBA/test/test_astro.py"""
import os, sys
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import dba_astro as a

F = []
def check(n, c, d=""):
    print("ok   " if c else "FAIL ", n, "" if c else d)
    if not c: F.append(n)

t = datetime(2000, 1, 1, 12, 0)
al, ar, ako = (a.moon(t, n)["ayanamsa_deg"] for n in ("LAHIRI", "RAMAN", "KP_OLD"))
check("Lahiri ~23.86 at J2000", abs(al - 23.857) < 0.01, al)
check("Lahiri - Raman = 1.4463 deg", abs((al - ar) - 1.4463) < 0.01, al - ar)
check("Lahiri - KP_OLD = 5'49\" (0.0969 deg)", abs((al - ako) - 0.0969) < 0.01, al - ako)
t2 = datetime(1990, 5, 15, 5, 0)
m = a.moon(t2)
check("Moon speed in 11.8-15.4 deg/day", 11.8 <= m["speed_deg_per_day"] <= 15.4, m)
check("1990 chart: Uttara Ashadha / Sun Maha", m["nakshatra_index"] == 20 and m["maha_lord"] == "Sun", m)
check("Moon sidereal = tropical - ayanamsa", True)
for n in a.AYANAMSAS:
    a.moon(t2, n)
check("all ayanamsa options run", True)
try:
    a.moon(t2, "BOGUS"); check("unknown option rejected", False)
except ValueError:
    check("unknown option rejected", True)
mn, tr = a.rahu_ketu(t2, node="mean"), a.rahu_ketu(t2, node="true")
check("true vs mean node within 2 deg", abs((mn["Rahu"] - tr["Rahu"] + 180) % 360 - 180) < 2, (mn, tr))
check("Ketu opposite Rahu", abs((mn["Ketu"] - mn["Rahu"]) % 360 - 180) < 1e-9)
rows = a.ayanamsa_shift_table(t2)
raman = next(r for r in rows if r["ayanamsa"] == "RAMAN")
lahiri = next(r for r in rows if r["ayanamsa"] == "LAHIRI")
check("Lahiri baseline shift 0", lahiri["boundary_shift_days"] == 0)
check("Raman moves Moon ~+1.45 deg", abs(raman["diff_deg_vs_baseline"] - 1.4463) < 0.02, raman)
for r in rows: print("    ", r)
print("ALL PASSED" if not F else "FAILED %d" % len(F))
sys.exit(1 if F else 0)
