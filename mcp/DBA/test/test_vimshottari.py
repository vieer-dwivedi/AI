"""Tests for dba_core + dba_vimshottari. Run from the project root:
   .venv/bin/python mcp/DBA/test/test_vimshottari.py
"""
import os
import sys
from datetime import datetime
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dba_core import Period, find_running, is_contiguous, render, to_fraction
import dba_vimshottari as v

JULIAN = Fraction(1461, 4)
FAILS = []


def check(name, cond, detail=""):
    if cond:
        print("ok   ", name)
    else:
        print("FAIL ", name, detail)
        FAILS.append(name)


# 1. table
check("years sum to 120", sum(v.YEARS.values()) == 120)
check("nine distinct lords", len(set(v.LORDS)) == 9 and set(v.LORDS) == set(v.YEARS))

# 2. year options
check("julian = 365.25", v.year_days_from("julian") == JULIAN)
check("savana = 360", v.year_days_from("savana") == 360)
check("fixed:365.2425", v.year_days_from("fixed:365.2425") == Fraction(3652425, 10000))
check("numeric string", v.year_days_from("365.25") == JULIAN)
try:
    v.year_days_from("true_solar_arc")
    check("true_solar_arc raises", False)
except NotImplementedError:
    check("true_solar_arc raises", True)

# 3. balance at birth
nak, i, frac = v.moon_start(0)
check("0 deg Aries: Ashwini, Ketu, nothing elapsed", (nak, v.LORDS[i], frac) == (0, "Ketu", 0))
nak, i, frac = v.moon_start(Fraction(40, 3))
check("13d20m: Bharani, Venus", (nak, v.LORDS[i], frac) == (1, "Venus", 0))
nak, i, frac = v.moon_start(Fraction(20, 3))
check("6d40m: Ashwini half elapsed", (nak, v.LORDS[i], frac) == (0, "Ketu", Fraction(1, 2)))
check("360 wraps to 0", v.moon_start(360)[0] == 0 and v.moon_start(-Fraction(1, 10**6))[0] == 26)
ms = v.mahadashas(Fraction(20, 3), "julian")
check("balance 3.5 y of Ketu", ms[0].end == Fraction(7, 2) * JULIAN and ms[0].start == -Fraction(7, 2) * JULIAN)

# 4. structure over many Moon positions and both year lengths
bad = []
for yd in ("julian", "savana", "fixed:365.2425"):
    ydf = v.year_days_from(yd)
    for k in range(0, 27 * 4):
        moon = Fraction(k, 4) * v.NAK_SPAN / 1  # 4 sample points per nakshatra
        ms = v.mahadashas(moon, yd)
        if not is_contiguous(ms):
            bad.append(("maha contiguity", yd, k))
        if ms[-1].end - ms[0].start != 120 * ydf:
            bad.append(("120-year cycle", yd, k))
        # first Maha already partly elapsed: birth (0) lies inside it
        if not ms[0].contains(0):
            bad.append(("birth inside first maha", yd, k))
        frontier = list(ms)
        for level in range(1, 4):               # Antar, Pratyantar, Sookshma
            nxt = []
            for p in frontier:
                ch = v.children(p)
                if ch[0].lord != p.lord:
                    bad.append(("first child lord", yd, k, p.path))
                if ch[0].start != p.start or ch[-1].end != p.end:
                    bad.append(("children span parent", yd, k, p.path))
                if not is_contiguous(ch):
                    bad.append(("children contiguous", yd, k, p.path))
                if sum(c.duration for c in ch) != p.duration:
                    bad.append(("children sum", yd, k, p.path))
                nxt.extend(ch)
            frontier = nxt if level < 3 else []
            if len(frontier) > 6000:
                frontier = frontier[:6000]
check("structure (3 year lengths x 108 moon points x 3 levels)", not bad, bad[:3])

# 5. exact deep level duration: Ketu/Ketu/Ketu/Ketu/Ketu/Ketu
ms = v.mahadashas(0, "julian")
p = ms[0]
for _ in range(5):
    p = v.children(p)[0]
check("Deha of Ketu-in-Ketu-... exact", p.duration == Fraction(7) * JULIAN * Fraction(7, 120) ** 5 and p.level == 6)

# 6. 360 vs 365.25: Rahu Maha differs by 94.5 days, 120 years by 630 days
j = {q.lord: q for q in v.mahadashas(0, "julian")}
s = {q.lord: q for q in v.mahadashas(0, "savana")}
check("Rahu Maha 94.5 d longer in julian", j["Rahu"].duration - s["Rahu"].duration == Fraction(189, 2))
check("120 years differ by 630 d", 120 * (JULIAN - 360) == 630)
# two epochs a decade or more apart: the end of Maha #4 and Maha #7 move by different amounts
j_end4 = [q for q in v.mahadashas(0, "julian")][3].end
s_end4 = [q for q in v.mahadashas(0, "savana")][3].end
j_end7 = [q for q in v.mahadashas(0, "julian")][6].end
s_end7 = [q for q in v.mahadashas(0, "savana")][6].end
check("drift grows between epochs", (j_end7 - s_end7) > (j_end4 - s_end4) > 0)

# 7. ladder and boundaries (start inclusive)
moon = Fraction("269.842238")        # chart in progress.md V1: Sun Maha, 0.2382 elapsed
nak, i, frac = v.moon_start(moon)
check("V1 chart: Moon in Uttara Ashadha (index 20), Sun Maha",
      nak == 20 and v.LORDS[i] == "Sun" and abs(float(frac) - 0.23817) < 1e-4, (nak, v.LORDS[i], float(frac)))
bal = (1 - frac) * 6
check("V1 chart: balance about 4.571 y", abs(float(bal) - 4.5710) < 2e-4, float(bal))
ms = v.mahadashas(moon, "julian")
end0 = ms[0].end
lad = v.ladder_at(moon, end0)
check("at Maha boundary the NEW Maha runs", lad[0].lord == "Moon" and lad[0].start == end0 and lad[-1].start == end0)
lad_before = v.ladder_at(moon, end0 - Fraction(1, 10**9))
check("just before the boundary the OLD Maha runs", lad_before[0].lord == "Sun")
lad = v.ladder_at(moon, 0)
check("ladder at birth has 6 levels, all contain birth", len(lad) == 6 and all(x.contains(0) for x in lad))
check("ladder path lengths 1..6", [x.level for x in lad] == [1, 2, 3, 4, 5, 6])
far = v.ladder_at(moon, 130 * JULIAN)
check("ladder 130 years after birth (second cycle)", far[0].contains(130 * JULIAN))
check("next_changes returns ends", v.next_changes(lad) == [x.end for x in lad])
check("find_running agrees with ladder_at", find_running(v.mahadashas(moon), 100) == v.ladder_at(moon, 100, depth=1)[0])

# 8. rendering
birth = datetime(1990, 5, 15, 10, 30, 0)
check("render 0", render(birth, 0) == birth)
check("render half second rounds up", render(birth, Fraction(1, 2 * 86400)) == datetime(1990, 5, 15, 10, 30, 1))
check("render one day", render(birth, 1) == datetime(1990, 5, 16, 10, 30, 0))

print()
print("FAILED: %d" % len(FAILS) if FAILS else "ALL PASSED")
sys.exit(1 if FAILS else 0)
