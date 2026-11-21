"""Run: .venv/bin/python mcp/DBA/test/test_resolution.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dba_resolution import resolution_report as rr

F = []
def check(n, c, d=""):
    print("ok   " if c else "FAIL ", n, "" if c else d)
    if not c: F.append(n)

r = rr(14, 13.2, birth_sigma_min=15)                      # Bharani: Venus
check("Venus 15 min = 75.3 d", abs(r["levels"][0]["boundary_shift_days"]["birth_time"] - 75.3) < 0.05, r["levels"][0])
r = rr(100, 13.2, birth_sigma_min=15)                     # Pushya: Saturn
check("Saturn 15 min = 71.6 d", abs(r["levels"][0]["boundary_shift_days"]["birth_time"] - 71.6) < 0.05, r["levels"][0])
r = rr(14, 13.2, birth_sigma_min=15)
check("Moon mid-nakshatra: no flip", not r["first_maha_lord_flips_within_sigma"] and r["stable_for_minutes"] > 15)
r = rr(13.3, 13.2, birth_sigma_min=15)                    # 2 arcmin before boundary
check("Moon near boundary flips", r["first_maha_lord_flips_within_sigma"] and r["nearest_boundary"] == "leaves_later", r)
r = rr(14, 13.2, t_days=0, birth_sigma_min=15)
check("Deha is below resolution, Maha is not", r["levels"][5]["below_resolution"] and not r["levels"][0]["below_resolution"])
r = rr(14, 13.2, t_days=365.25 * 30, year_modes=("julian", "savana"))
check("year-length spread grows with age (30 y)", r["levels"][0]["boundary_shift_days"]["year_length"] > 100, r["levels"][0])
r = rr(266.0, 13.2, birth_sigma_min=0, moon_deg_alternatives=(266.0 - 1.4463,))
check("Lahiri vs Raman-size gap moves Maha by years", r["levels"][0]["boundary_shift_days"]["ayanamsa"] > 365, r["levels"][0])
print("ALL PASSED" if not F else "FAILED %d" % len(F))
sys.exit(1 if F else 0)
