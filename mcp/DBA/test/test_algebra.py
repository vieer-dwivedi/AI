"""Run: .venv/bin/python mcp/DBA/test/test_algebra.py"""
import os, sys
from fractions import Fraction as Fr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import dba_algebra as al
import dba_vimshottari as v
from dba_core import Period

F = []
def check(n, c, d=""):
    print("ok   " if c else "FAIL ", n, "" if c else d)
    if not c: F.append(n)

W = al.Window
a = [W(Fr(0), Fr(5)), W(Fr(10), Fr(20))]
b = [W(Fr(3), Fr(12)), W(Fr(30), Fr(40))]
check("union", [(w.start, w.end) for w in al.union(a, b)] == [(0, 20), (30, 40)])
check("intersect", [(w.start, w.end) for w in al.intersect(a, b)] == [(3, 5), (10, 12)])
check("gaps", [(w.start, w.end) for w in al.gaps(a, 0, 25)] == [(5, 10), (20, 25)])
check("coverage", al.coverage_fraction(a, 0, 25) == Fr(15, 25))
check("merge touching", len(al.merge_adjacent([W(Fr(0), Fr(5)), W(Fr(5), Fr(9))])) == 1)
check("empty intersect", al.intersect(a, [W(Fr(21), Fr(29))]) == [])

yd = v.year_days_from("julian")
span = 120 * yd
S = {"Sun", "Moon", "Mars"}
# Moon at 0 deg: Maha cycle starts exactly at birth, so [0, 120y) is one full cycle
r = al.find_periods(0, lambda p: p.path[0] in S and p.path[1] in S, 2, 0, span)
cov = al.coverage_fraction(r["periods"], 0, span)
check("exact coverage = (23/120)^2 over a full cycle", cov == Fr(23, 120) ** 2 and not r["truncated"], cov)
r1 = al.find_periods(0, lambda p: p.path[0] in S, 1, 0, span)
check("Maha-level coverage = 23/120", al.coverage_fraction(r1["periods"], 0, span) == Fr(23, 120))
# either lord qualifies: 1 - (97/120)^2 at Antar level
r2 = al.find_periods(0, lambda p: p.path[0] in S or p.path[1] in S, 2, 0, span)
check("either-lord coverage = 1-(97/120)^2", al.coverage_fraction(r2["periods"], 0, span) == 1 - Fr(97, 120) ** 2)
# truncation
r3 = al.find_periods(0, lambda p: True, 3, 0, span, max_results=50)
check("truncated flag and cap", r3["truncated"] and len(r3["periods"]) == 50)
# pruning: narrow window gives few periods and matches ladder
t = 40 * yd
r4 = al.find_periods(0, lambda p: True, 6, t, t + Fr(1, 10**6))
lad = v.ladder_at(0, t)
check("one-instant search returns the ladder's Deha", len(r4["periods"]) == 1 and r4["periods"][0] == lad[-1])
check("sources kept", al.merge_adjacent(r["periods"])[0].sources[0][0] in S)
print("ALL PASSED" if not F else "FAILED %d" % len(F))
sys.exit(1 if F else 0)
