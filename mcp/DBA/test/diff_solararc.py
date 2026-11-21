"""Differential test: DBA true_solar_arc modes vs PyJHora's Sun-return modes (via mcp/dasha), random charts.
Run from the project root:  .venv/bin/python mcp/DBA/test/diff_solararc.py
  true_solar_arc          <->  TRUE_TROPICAL_YEAR   (PyJHora default)
  true_solar_arc_sidereal <->  TRUE_SIDEREAL_YEAR
Compares all nine Maha starts/ends and every Antar start/end of the first and fourth Maha.
Differences expected only from float round-off, ephemeris noise and PyJHora's chaining / clamping.
"""
import os
import random
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "dasha"))
import swisseph as swe
import dasha_engine as e
from state import BirthState
import dba_vimshottari as v
import dba_solararc as sa
from dba_core import render

MODES = {"TRUE_TROPICAL_YEAR": "true_solar_arc", "TRUE_SIDEREAL_YEAR": "true_solar_arc_sidereal"}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 25
rng = random.Random(20261121)
tz = ZoneInfo("Asia/Kolkata")
swe.set_sid_mode(swe.SIDM_LAHIRI)

worst_md = {m: 0.0 for m in MODES}
worst_ad = {m: 0.0 for m in MODES}
lord_mismatch = errors = compared = 0
for n in range(N):
    y, mth, d = rng.randint(1950, 2015), rng.randint(1, 12), rng.randint(1, 28)
    hh, mm = rng.randint(0, 23), rng.randint(0, 59)
    lat, lon = rng.uniform(8, 32), rng.uniform(70, 90)
    local = datetime(y, mth, d, hh, mm, tzinfo=tz)
    utc = local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    jd = sa._jd(utc)
    moon = swe.calc_ut(jd, swe.MOON, swe.FLG_SWIEPH | swe.FLG_SIDEREAL)[0][0]
    birth = local.replace(tzinfo=None)
    for pj_mode, mode in MODES.items():
        try:
            st = BirthState(date_of_birth="%04d-%02d-%02d" % (y, mth, d), time_of_birth="%02d:%02d" % (hh, mm),
                            latitude=lat, longitude=lon, timezone="Asia/Kolkata", options={"year_type": pj_mode})
            pj = e.get_mahadashas(st)["mahadashas"][:9]
            tm = sa.resolve_year_mode(mode, utc)
            ours = v.mahadashas(moon, tm)
            if not pj[0]["lord"].startswith(ours[0].lord):
                lord_mismatch += 1
                continue
            for p, row in zip(ours, pj):
                ps, pe = datetime.fromisoformat(row["start"]), datetime.fromisoformat(row["end"])
                worst_md[pj_mode] = max(worst_md[pj_mode],
                                        abs((render(birth, p.start) - ps).total_seconds()) / 86400,
                                        abs((render(birth, p.end) - pe).total_seconds()) / 86400)
            for idx in (0, 3):
                sp = e.get_sub_periods(st, maha=str(idx + 1), levels_below=1)["sub_periods"]
                for c, row in zip(v.children(ours[idx]), sp):
                    ps, pe = datetime.fromisoformat(row["start"]), datetime.fromisoformat(row["end"])
                    worst_ad[pj_mode] = max(worst_ad[pj_mode],
                                            abs((render(birth, c.start) - ps).total_seconds()) / 86400,
                                            abs((render(birth, c.end) - pe).total_seconds()) / 86400)
            compared += 1
        except Exception as ex:
            errors += 1
            print("chart %d %s ERROR %r" % (n, pj_mode, ex)[:300])

print("charts: %d  comparisons: %d  first-lord mismatches: %d  errors: %d" % (N, compared, lord_mismatch, errors))
for m in MODES:
    print("%-20s Maha boundary max diff %.4f d   Antar boundary max diff (Maha 1 and 4) %.4f d" % (m, worst_md[m], worst_ad[m]))
ok = lord_mismatch == 0 and errors == 0 and all(x < 0.5 for x in worst_md.values()) and all(x < 0.5 for x in worst_ad.values())
print("PASS" if ok else "CHECK")
sys.exit(0 if ok else 1)
