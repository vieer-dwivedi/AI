"""Differential test on historical dates and non-India places: DBA exact Vimshottari vs PyJHora (via mcp/dasha).
Run from the project root:  .venv/bin/python mcp/DBA/test/diff_historical.py
Both use the same fixed year (MEAN_TROPICAL_YEAR = 365.24219 and MEAN_SIDEREAL_YEAR = 365.256364), the same IANA
timezone for the birth moment, Lahiri ayanamsa. The Moon comes from Swiss Ephemeris through dba_time + dba_astro.
What is checked: first Maha lord, the nine Maha boundaries (local clock, birth offset kept), first-Maha Antar lengths.
Also printed: the UTC offset each side used, so a time-zone disagreement is visible.
"""
import os
import sys
from datetime import datetime
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "dasha"))
import dasha_engine as e
from state import BirthState
import dba_astro as astro
import dba_vimshottari as v
from dba_core import render
from dba_time import to_utc

MODES = {"MEAN_TROPICAL_YEAR": "fixed:365.24219", "MEAN_SIDEREAL_YEAR": "fixed:365.256364"}
# (label, date, time, IANA tz, lat, lon)
CHARTS = [
    ("1943 India war time", "1943-06-01", "12:00", "Asia/Kolkata", 19.07, 72.87),
    ("1930 India", "1930-11-12", "05:45", "Asia/Kolkata", 22.57, 88.36),
    ("1905 India pre-1906", "1905-03-10", "08:20", "Asia/Kolkata", 28.61, 77.21),
    ("1969 London BST all year", "1969-02-10", "09:30", "Europe/London", 51.51, -0.13),
    ("1975 New York DST", "1975-07-04", "14:10", "America/New_York", 40.71, -74.01),
    ("1988 Sydney DST", "1988-01-20", "22:05", "Australia/Sydney", -33.87, 151.21),
    ("1960 Tokyo", "1960-09-30", "03:15", "Asia/Tokyo", 35.68, 139.69),
    ("1999 Los Angeles winter", "1999-12-15", "07:40", "America/Los_Angeles", 34.05, -118.24),
    ("1951 Nairobi", "1951-04-03", "16:25", "Africa/Nairobi", -1.29, 36.82),
    ("2012 Kathmandu +5:45", "2012-08-08", "11:11", "Asia/Kathmandu", 27.72, 85.32),
]

worst_md = {m: 0.0 for m in MODES}
worst_ad = {m: 0.0 for m in MODES}
problems = 0
print("%-26s %-19s %9s %9s %8s %8s" % ("chart", "year_type", "MD max d", "AD max d", "off own", "off PJ"))
for label, date, time, tzn, lat, lon in CHARTS:
    local = datetime.fromisoformat(date + "T" + time + ":00")
    ti = to_utc(local, tz_name=tzn)
    moon = astro.moon(ti["utc_used"], "LAHIRI")["longitude_deg"]
    for pj_mode, mode in MODES.items():
        try:
            st = BirthState(date_of_birth=date, time_of_birth=time, latitude=lat, longitude=lon, timezone=tzn,
                            options={"year_type": pj_mode})
            pj = e.get_mahadashas(st)["mahadashas"][:9]
            ours = v.mahadashas(moon, mode)
            off_pj = st.timezone_offset()
            if not pj[0]["lord"].startswith(ours[0].lord):
                problems += 1
                print("%-26s %-19s FIRST LORD MISMATCH own %s PyJHora %s" % (label, pj_mode, ours[0].lord, pj[0]["lord"]))
                continue
            md = 0.0
            for p, row in zip(ours, pj):
                ps, pe = datetime.fromisoformat(row["start"]), datetime.fromisoformat(row["end"])
                md = max(md, abs((render(local, p.start) - ps).total_seconds()) / 86400,
                         abs((render(local, p.end) - pe).total_seconds()) / 86400)
            sp = e.get_sub_periods(st, maha="1", levels_below=1)["sub_periods"]
            ad = max(abs(float(c.duration) - row["duration_days"]) for c, row in zip(v.children(ours[0]), sp))
            worst_md[pj_mode], worst_ad[pj_mode] = max(worst_md[pj_mode], md), max(worst_ad[pj_mode], ad)
            flag = "" if md < 0.5 and ad < 0.05 and abs(ti["offset_hours"] - off_pj) < 1e-9 else "   <-- CHECK"
            problems += bool(flag)
            print("%-26s %-19s %9.4f %9.4f %8.2f %8.2f%s" % (label, pj_mode, md, ad, ti["offset_hours"], off_pj, flag))
            for w in ti["warnings"]:
                print("    own warning:", w)
        except Exception as ex:
            problems += 1
            print("%-26s %-19s ERROR %r" % (label, pj_mode, ex)[:300])
print("problems: %d" % problems)
for m in MODES:
    print("%-20s worst MD %.4f d, worst AD %.4f d" % (m, worst_md[m], worst_ad[m]))
sys.exit(1 if problems else 0)
