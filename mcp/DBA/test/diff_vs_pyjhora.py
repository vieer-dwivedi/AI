"""Differential test: DBA exact Vimshottari vs PyJHora (via mcp/dasha), 50 random charts, two fixed year modes.
Run from the project root:  .venv/bin/python mcp/DBA/test/diff_vs_pyjhora.py
Moon longitude comes from Swiss Ephemeris (Lahiri, sidereal); both engines use the same fixed year.
Expected: Maha boundaries agree to ~0.1 d (Moon-longitude noise, < 1 arcsec), Antar lengths of the first Maha to ~0.005 d.
"""
import os
import random
import sys
from datetime import datetime
from fractions import Fraction
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "dasha"))
import swisseph as swe
import dasha_engine as e
from state import BirthState
import dba_vimshottari as v
from dba_core import render

MODES = {"MEAN_TROPICAL_YEAR": "fixed:365.24219", "MEAN_SIDEREAL_YEAR": "fixed:365.256364"}
swe.set_sid_mode(swe.SIDM_LAHIRI)
rng = random.Random(20261120)
tz = ZoneInfo("Asia/Kolkata")
N = 50

def moon_sidereal(local):
    utc = local.astimezone(ZoneInfo("UTC"))
    jd = swe.julday(utc.year, utc.month, utc.day, utc.hour + utc.minute / 60.0 + utc.second / 3600.0, swe.GREG_CAL)
    return swe.calc_ut(jd, swe.MOON, swe.FLG_SWIEPH | swe.FLG_SIDEREAL)[0][0]

worst_md = {m: 0.0 for m in MODES}
worst_ad = {m: 0.0 for m in MODES}
lord_mismatch = 0
errors = 0
for n in range(N):
    y, mth, d = rng.randint(1950, 2015), rng.randint(1, 12), rng.randint(1, 28)
    hh, mm = rng.randint(0, 23), rng.randint(0, 59)
    lat, lon = rng.uniform(8, 32), rng.uniform(70, 90)
    local = datetime(y, mth, d, hh, mm, tzinfo=tz)
    moon = moon_sidereal(local)
    birth = local.replace(tzinfo=None)
    for pj_mode, mode in MODES.items():
        try:
            st = BirthState(date_of_birth="%04d-%02d-%02d" % (y, mth, d), time_of_birth="%02d:%02d" % (hh, mm),
                            latitude=lat, longitude=lon, timezone="Asia/Kolkata", options={"year_type": pj_mode})
            pj = e.get_mahadashas(st)["mahadashas"][:9]
            ours = v.mahadashas(moon, mode)
            if not pj[0]["lord"].startswith(ours[0].lord):
                lord_mismatch += 1
                continue
            for p, row in zip(ours, pj):
                ps, pe = datetime.fromisoformat(row["start"]), datetime.fromisoformat(row["end"])
                worst_md[pj_mode] = max(worst_md[pj_mode],
                                        abs((render(birth, p.start) - ps).total_seconds()) / 86400,
                                        abs((render(birth, p.end) - pe).total_seconds()) / 86400)
            sp = e.get_sub_periods(st, maha="1", levels_below=1)["sub_periods"]
            for c, row in zip(v.children(ours[0]), sp):
                dur = row["duration_days"]
                worst_ad[pj_mode] = max(worst_ad[pj_mode], abs(float(c.duration) - dur))
        except Exception as ex:
            errors += 1
            print("chart %d %s ERROR %r" % (n, pj_mode, ex)[:300])

print("charts: %d  modes: %d  first-lord mismatches: %d  errors: %d" % (N, len(MODES), lord_mismatch, errors))
for m in MODES:
    print("%-20s MD boundary max diff %.4f d   first-Maha Antar max diff %.4f d" % (m, worst_md[m], worst_ad[m]))
ok = lord_mismatch == 0 and errors == 0 and all(x < 0.5 for x in worst_md.values()) and all(x < 0.05 for x in worst_ad.values())
print("PASS" if ok else "CHECK")
sys.exit(0 if ok else 1)
