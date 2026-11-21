"""Run: .venv/bin/python mcp/DBA/test/test_bands.py"""
import os, sys
from fractions import Fraction as Fr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import dba_bands as bd
import dba_algebra as alg
import dba_vimshottari as v

F = []
def check(n, c, d=""):
    print("ok   " if c else "FAIL ", n, "" if c else d)
    if not c: F.append(n)

yd = Fr(36525, 100)
HI = 120 * yd
S = {"Sun", "Moon", "Mars"}
pred = lambda p: p.path[0] in S and p.path[1] in S
moon, speed = Fr(100), Fr(13)

sc0 = bd.make_scenarios(moon, speed)
check("no uncertainty: 1 scenario", len(sc0) == 1)
b0 = bd.window_bands(pred, 2, 0, HI, sc0)
base = alg.find_periods(moon, pred, 2, 0, HI)["periods"]
check("no uncertainty: certain == possible == base windows",
      [(w.start, w.end) for w in b0["possible"]] == [(w.start, w.end) for w in b0["certain"]] ==
      [(w.start, w.end) for w in alg.merge_adjacent(base)])
sc = bd.make_scenarios(moon, speed, sigma_min=15, year_modes=["julian", "savana"])
check("3 shifts x 2 year modes = 6 scenarios", len(sc) == 6)
b = bd.window_bands(pred, 2, 0, HI, sc)
nom = [w for lab, w in zip([s["label"] for s in sc], b["per_scenario"]) if "birth +0.0" in lab and "julian" in lab][0]
cn = alg.coverage_fraction(nom, 0, HI)
check("certain <= nominal <= possible", b["coverage_certain"] <= cn <= b["coverage_possible"], (b["coverage_certain"], cn, b["coverage_possible"]))
check("certain is a subset of nominal", alg.intersect(b["certain"], nom) and
      alg.coverage_fraction(alg.intersect(b["certain"], nom), 0, HI) == b["coverage_certain"])
# an alternative ayanamsa (Moon 1.45 deg away) moves boundaries by months to years
sc2 = bd.make_scenarios(moon, speed, 0, [moon - Fr(145, 100)])
b2 = bd.window_bands(pred, 2, 0, HI, sc2)
check("alternative ayanamsa widens possible and narrows certain", b2["coverage_possible"] > b2["coverage_certain"]
      and b2["coverage_possible"] > b0["coverage_possible"] and b2["coverage_certain"] < b0["coverage_certain"], b2["coverage_possible"])
# shift algebra: +15 min moves a boundary by about speed-based days, in the right direction
d = Fr(15, 1440)
sh = bd.scenario_windows({"moon": moon + speed * d, "shift_days": d, "year_mode": "julian"}, pred, 2, 0, HI)
check("shifted scenario windows are valid and ordered", all(w.start < w.end for w in sh) and all(a.end <= b_.start for a, b_ in zip(sh, sh[1:])))
# robust_hit
w = b["per_scenario"]
mid_certain = (b["certain"][0].start + b["certain"][0].end) / 2
check("robust_hit: inside certain window = all", bd.robust_hit(w, mid_certain, mid_certain + 1) == "all")
gap_pt = (b["possible"][0].end + b["possible"][1].start) / 2
check("robust_hit: point in a gap of possible = none", bd.robust_hit(w, gap_pt, gap_pt + Fr(1, 10)) == "none")
fringe = alg.intersect(b["possible"], alg.gaps(b["certain"], 0, HI))
m = (fringe[0].start + fringe[0].end) / 2
check("robust_hit: in possible but not certain = some", len(fringe) > 0 and bd.robust_hit(w, m, m + Fr(1, 100)) == "some", fringe[:1])
print("ALL PASSED" if not F else "FAILED %d: %s" % (len(F), F))
sys.exit(1 if F else 0)
