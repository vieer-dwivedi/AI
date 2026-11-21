import sys, io, contextlib, traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
sys.path.insert(0, "mcp/dasha")
import swisseph as swe
import dasha_engine as e
from state import BirthState

LORDS = ["Ketu","Venus","Sun","Moon","Mars","Rahu","Jupiter","Saturn","Mercury"]
YEARS = [7,20,6,10,7,18,16,19,17]
NAK = 360/27
YD = {"MEAN_TROPICAL_YEAR": 365.24219, "GREGORIAN_YEAR": 365.2425, "MEAN_SIDEREAL_YEAR": 365.256364}
# (label, date, time, lat, lon)  all Asia/Kolkata; spread over seasons and eras
CHARTS = [
    ("1990-05-15 Mumbai", "1990-05-15", "10:30", 19.07, 72.87),
    ("1962-01-03 Delhi", "1962-01-03", "06:15", 28.61, 77.21),
    ("1955-07-20 Chennai", "1955-07-20", "21:40", 13.08, 80.27),
    ("1978-10-08 Kolkata", "1978-10-08", "14:05", 22.57, 88.36),
    ("2004-03-21 Mumbai", "2004-03-21", "03:50", 19.07, 72.87),
]
swe.set_sid_mode(swe.SIDM_LAHIRI)

def own(date, time, tzname, yd):
    local = datetime.fromisoformat(date + "T" + time + ":00").replace(tzinfo=ZoneInfo(tzname))
    utc = local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    jd = swe.julday(utc.year, utc.month, utc.day, utc.hour + utc.minute/60.0, swe.GREG_CAL)
    moon = swe.calc_ut(jd, swe.MOON, swe.FLG_SWIEPH | swe.FLG_SIDEREAL)[0][0]
    nak = int(moon // NAK); frac = (moon - nak*NAK)/NAK; i0 = nak % 9
    birth_local = local.replace(tzinfo=None)
    start = birth_local - timedelta(days=frac*YEARS[i0]*yd)
    out, t = [], start
    for k in range(9):
        i = (i0 + k) % 9
        end = t + timedelta(days=YEARS[i]*yd)
        out.append((LORDS[i], t, end)); t = end
    return out, i0, frac

print("%-20s %-20s %10s %10s" % ("chart", "year_type", "MD maxdiff", "AD maxdev"))
for label, date, time, lat, lon in CHARTS:
    for mode, yd in YD.items():
        try:
            st = BirthState(date_of_birth=date, time_of_birth=time, latitude=lat, longitude=lon,
                            timezone="Asia/Kolkata", options={"year_type": mode})
            pj = e.get_mahadashas(st)["mahadashas"]
            ow, i0, frac = own(date, time, "Asia/Kolkata", yd)
            worst = 0.0
            for (lord, s, en), row in zip(ow, pj[:9]):
                ps, pe = datetime.fromisoformat(row["start"]), datetime.fromisoformat(row["end"])
                worst = max(worst, abs((s-ps).total_seconds()/86400), abs((en-pe).total_seconds()/86400))
            first_lord_ok = pj[0]["lord"].startswith(LORDS[i0])
            sp = e.get_sub_periods(st, maha="1", levels_below=1)
            parent = sp["parent"]["duration_days"]
            dev = 0.0
            for ch in sp["sub_periods"]:
                lord = next(l for l in LORDS if ch["lord"].startswith(l))
                dev = max(dev, abs(ch["duration_days"] - parent*YEARS[LORDS.index(lord)]/120))
            print("%-20s %-20s %10.4f %10.4f   first lord match: %s (moon frac %.4f)" % (label, mode, worst, dev, first_lord_ok, frac))
        except Exception as ex:
            print("%-20s %-20s ERROR %s" % (label, mode, repr(ex)[:200]))
            traceback.print_exc(limit=2)
