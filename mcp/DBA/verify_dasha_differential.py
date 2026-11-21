import sys, io, contextlib
from datetime import datetime, timedelta, timezone
from fractions import Fraction
sys.path.insert(0, "mcp/dasha")
import swisseph as swe
import dasha_engine as e
from state import BirthState

st = BirthState(date_of_birth="1990-05-15", time_of_birth="10:30", latitude=19.07, longitude=72.87, timezone="Asia/Kolkata")
LORDS = ["Ketu","Venus","Sun","Moon","Mars","Rahu","Jupiter","Saturn","Mercury"]
YEARS = [7,20,6,10,7,18,16,19,17]
NAK = 360/27

# ---- independent Moon longitude (Swiss Ephemeris, Lahiri, with and without nutation convention)
jd_ut = swe.julday(1990, 5, 15, 5.0, swe.GREG_CAL)  # 10:30 IST = 05:00 UT
swe.set_sid_mode(swe.SIDM_LAHIRI)
flags = swe.FLG_SWIEPH | swe.FLG_SIDEREAL | swe.FLG_SPEED
moon_sid = swe.calc_ut(jd_ut, swe.MOON, flags)[0][0]
ay_ut = swe.get_ayanamsa_ut(jd_ut)
moon_trop = swe.calc_ut(jd_ut, swe.MOON, swe.FLG_SWIEPH)[0][0]
moon_alt = (moon_trop - ay_ut) % 360
print("Moon sidereal (swe sid mode): %.6f   tropical-ayanamsa(no nutation): %.6f   diff arcsec: %.2f" % (moon_sid, moon_alt, (moon_sid-moon_alt)*3600))

def ladder(moon):
    nak = int(moon // NAK)
    frac = (moon - nak*NAK)/NAK
    idx = nak % 9
    return idx, frac

birth = datetime(1990, 5, 15, 5, 0, 0)  # UTC naive

def mds(moon, yd):
    idx, frac = ladder(moon)
    y = YEARS[idx]
    start = birth - timedelta(days=frac*y*yd)
    out = []
    t = start
    for k in range(9):
        i = (idx+k) % 9
        end = t + timedelta(days=YEARS[i]*yd)
        out.append((LORDS[i], t, end))
        t = end
    return out

# ---- PyJHora via existing dasha engine (local times, IST = UTC+5:30)
pj = e.get_mahadashas(st)["mahadashas"]
def parse(s): return datetime.fromisoformat(s) - timedelta(hours=5, minutes=30)

for label, moon, yd in (("365.25, swe-sid Moon", moon_sid, 365.25),
                        ("365.24219, swe-sid Moon", moon_sid, 365.24219),
                        ("365.25636, swe-sid Moon", moon_sid, 365.256364)):
    ind = mds(moon, yd)
    print("\n==", label)
    worst = 0
    for (lord, s, en), row in list(zip(ind, pj))[:9]:
        ps, pe = parse(row["start"]), parse(row["end"])
        ds = (s-ps).total_seconds()/86400
        de = (en-pe).total_seconds()/86400
        worst = max(worst, abs(ds), abs(de))
        print(f"{lord:8s} PyJHora {row['lord'][:6]:6s} start diff {ds:+8.3f} d   end diff {de:+8.3f} d")
    print("worst abs diff (days):", round(worst,3))

# implied year length in PyJHora per MD
print("\nImplied year length in PyJHora MDs (days/year):")
for row in pj[:9]:
    print(row["lord"][:7], row["duration_years"], round(row["duration_days"]/row["duration_years"], 5))

# ---- sub-period proportionality inside first MD (Sun)
sp = e.get_sub_periods(st, maha="1", levels_below=1)
parent = sp["parent"]["duration_days"]
tot = 0
print("\nAD durations in Sun MD (PyJHora) vs exact Sun x lord / 120 x MD length:")
for ch in sp["sub_periods"]:
    lord = ch["lord"][:-1] if not ch["lord"][-1].isalpha() else ch["lord"]
    y = dict(zip(LORDS, YEARS))[lord]
    expect = parent * y / 120
    tot += ch["duration_days"]
    print(f"{lord:8s} {ch['duration_days']:9.3f}  expect {expect:9.3f}  diff {ch['duration_days']-expect:+.3f}")
print("sum of ADs", round(tot,3), " parent", parent)
