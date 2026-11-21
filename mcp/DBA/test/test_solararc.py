"""Tests for the true_solar_arc year mode (dba_solararc.py). Run from the project root:
   .venv/bin/python mcp/DBA/test/test_solararc.py
"""
import os
import sys
from datetime import datetime
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import swisseph as swe
import dba_vimshottari as v
import dba_algebra as alg
import dba_solararc as sa
from dba_core import is_contiguous

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL "), name, "" if cond else detail)
    if not cond:
        FAILS.append(name)


def tropical_sun(utc, days):
    jd = sa._jd(utc) + float(days)
    return swe.calc_ut(jd, swe.SUN, swe.FLG_SWIEPH | swe.FLG_TROPICAL)[0][0]


BIRTH = datetime(1990, 5, 15, 5, 0, 0)          # 10:30 IST
MOON = Fraction("269.8424")

# 1. bare string still needs the birth instant
try:
    v.year_days_from("true_solar_arc")
    check("bare true_solar_arc string raises", False)
except NotImplementedError:
    check("bare true_solar_arc string raises", True)
try:
    sa.resolve_year_mode("true_solar_arc")
    check("resolve without birth raises", False)
except ValueError:
    check("resolve without birth raises", True)

tm = sa.resolve_year_mode("true_solar_arc", BIRTH)
check("resolve fixed mode gives Fraction", sa.resolve_year_mode("julian") == Fraction(1461, 4))
check("map is recognised", v.is_map(tm) and not v.is_map(Fraction(1)))
check("tau 0 = birth", tm.days(0) == 0)

# 2. the map really is the Sun's arc
for tau in (1, 5, 18, -3, Fraction(57, 100) + 4):
    d = tm.days(tau)
    err = (tropical_sun(BIRTH, d) - (tm.lon0 + 360 * float(tau))) % 360
    err = min(err, 360 - err)
    check("Sun arc at tau=%s (err %.2e deg)" % (tau, err), err < 1e-6)
check("one revolution is near 365.2422 d", abs(float(tm.days(1)) - 365.2422) < 0.02, float(tm.days(1)))
check("monotone in tau", all(tm.days(Fraction(k, 7)) < tm.days(Fraction(k + 1, 7)) for k in range(-20, 120)))
check("additive in arc: 2 x 1 revolution lands on the same longitude",
      abs(((tropical_sun(BIRTH, tm.days(2)) - tm.lon0 + 180) % 360) - 180) < 1e-6)

# 3. structure: contiguity, exact sums, boundary identity across parents
mahas = v.mahadashas(MOON, tm)
check("nine Mahas, contiguous", len(mahas) == 9 and is_contiguous(mahas))
check("Maha tau lengths are the lord's years", all(m.tau_end - m.tau_start == v.YEARS[m.lord] for m in mahas))
check("first Maha starts before birth by the elapsed share", mahas[0].start < 0 < mahas[0].end)
ok_sum = ok_cont = True
for m in mahas:
    ch = v.children(m)
    ok_cont &= is_contiguous(ch) and ch[0].start == m.start and ch[-1].end == m.end
    ok_sum &= sum(c.tau_end - c.tau_start for c in ch) == m.tau_end - m.tau_start
check("children contiguous and end exactly at the parent end (all Mahas)", ok_cont)
check("children tau lengths sum exactly to the parent", ok_sum)
deep = mahas[0]
for _ in range(5):
    ch = v.children(deep)
    deep = ch[4]
check("six levels deep, path length 6, still a map period", deep.level == 6 and deep.tmap is tm)
_c = v.children(mahas[2])[3]
check("tau lengths follow parent x child/120 exactly",
      _c.tau_end - _c.tau_start == Fraction(v.YEARS[mahas[2].lord] * v.YEARS[_c.lord], 120))

# 4. not proportional in days (that is the point), but within the Sun-speed band
ch = v.children(mahas[2])
prop = [float(mahas[2].duration) * v.YEARS[c.lord] / 120 for c in ch]
dev = [abs(float(c.duration) - p) for c, p in zip(ch, prop)]
check("Antar lengths differ from the proportional days by some hours to days, never more than 5 d (max %.2f d)" % max(dev),
      0.05 < max(dev) < 5, dev)
check("whole Maha is within 0.8% of the mean-year length",
      all(abs(float(m.duration) / (v.YEARS[m.lord] * 365.24219) - 1) < 0.008 for m in mahas[1:]))

# 5. ladder_at and next_changes work with a map and agree with the periods
lad = v.ladder_at(MOON, Fraction(9000), tm, 6)
check("ladder at day 9000 has 6 levels and nests", len(lad) == 6 and all(a.start <= b.start and b.end <= a.end for a, b in zip(lad, lad[1:])))
check("boundary belongs to the new period (map mode)",
      v.ladder_at(MOON, mahas[1].start, tm, 1)[0].path == mahas[1].path)
check("one instant before the boundary is the old Maha",
      v.ladder_at(MOON, mahas[1].start - Fraction(1, 86400), tm, 1)[0].path == mahas[0].path)

# 6. find_periods with a map
r = alg.find_periods(MOON, lambda p: p.path[0] == "Venus" and p.path[1] == "Sun", 2, Fraction(0), 120 * Fraction("365.24219"), tm)
check("find_periods finds Venus-Sun Antar in a 120-year range with a map", len(r["periods"]) >= 1 and not r["truncated"])

# 7. sidereal variant: longer year by about 20.4 minutes... (sidereal 365.2564 d)
ts = sa.resolve_year_mode("true_solar_arc_sidereal", BIRTH)
check("sidereal year longer than tropical (%.4f vs %.4f)" % (float(ts.days(1)), float(tm.days(1))),
      0.005 < float(ts.days(1) - tm.days(1)) < 0.02)
check("sidereal name", str(ts) == "true_solar_arc_sidereal" and ts.sidereal)

# 8. fixed modes unchanged: Periods carry no map
fx = v.mahadashas(MOON, "julian")
check("fixed mode Period has no map", fx[0].tmap is None and v.children(fx[0])[0].tmap is None)
check("fixed mode equality ignores tau fields", fx[0] == fx[0].__class__(fx[0].path, fx[0].start, fx[0].end, Fraction(1), Fraction(2)))

# 9. different birth instants give different but equally valid maps (a one-case check hides bugs)
for b in (datetime(1955, 7, 1, 3, 0), datetime(1978, 10, 20, 17, 25), datetime(2004, 3, 9, 0, 40)):
    m2 = sa.SunArcMap(b)
    d = m2.days(19)
    err = (tropical_sun(b, d) - (m2.lon0 + 360 * 19)) % 360
    err = min(err, 360 - err)
    check("arc check for %s (err %.1e)" % (b.date(), err), err < 1e-6 and abs(float(d) - 19 * 365.2422) < 3)

print("\nALL PASSED" if not FAILS else "\nFAILED: %s" % FAILS)
sys.exit(0 if not FAILS else 1)
