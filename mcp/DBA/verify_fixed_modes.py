import sys, io, contextlib
from datetime import datetime, timedelta
sys.path.insert(0, "mcp/dasha")
import swisseph as swe
import dasha_engine as e
from state import BirthState
with contextlib.redirect_stdout(io.StringIO()):
    from jhora import const

st = BirthState(date_of_birth="1990-05-15", time_of_birth="10:30", latitude=19.07, longitude=72.87, timezone="Asia/Kolkata")
LORDS = ["Ketu","Venus","Sun","Moon","Mars","Rahu","Jupiter","Saturn","Mercury"]
YEARS = [7,20,6,10,7,18,16,19,17]
NAK = 360/27
jd_ut = swe.julday(1990, 5, 15, 5.0, swe.GREG_CAL)
swe.set_sid_mode(swe.SIDM_LAHIRI)
moon = swe.calc_ut(jd_ut, swe.MOON, swe.FLG_SWIEPH | swe.FLG_SIDEREAL)[0][0]
nak = int(moon // NAK); frac = (moon - nak*NAK)/NAK; idx0 = nak % 9
birth = datetime(1990, 5, 15, 5, 0, 0)

def own(yd):
    start = birth - timedelta(days=frac*YEARS[idx0]*yd)
    out, t = [], start
    for k in range(9):
        i = (idx0+k) % 9
        end = t + timedelta(days=YEARS[i]*yd)
        out.append((LORDS[i], t, end)); t = end
    return out

def parse(s): return datetime.fromisoformat(s) - timedelta(hours=5, minutes=30)

for mode, yd in (("MEAN_TROPICAL_YEAR", 365.24219), ("GREGORIAN_YEAR", 365.2425), ("MEAN_SIDEREAL_YEAR", 365.256364), ("TRUE_TROPICAL_YEAR (default)", None)):
    const.dhasa_year_duration_default = getattr(const.DHASA_YEAR_DURATION, mode.split(" ")[0])
    pj = e.get_mahadashas(st)["mahadashas"]
    print("\n==", mode)
    if yd is not None:
        worst = 0
        for (lord, s, en), row in zip(own(yd), pj[:9]):
            ds = (s - parse(row["start"])).total_seconds()/86400
            de = (en - parse(row["end"])).total_seconds()/86400
            worst = max(worst, abs(ds), abs(de))
        print("worst abs MD boundary diff (days) vs own fixed yd=%s: %.4f" % (yd, worst))
    sp = e.get_sub_periods(st, maha="1", levels_below=1)
    parent = sp["parent"]["duration_days"]
    dev = []
    for ch in sp["sub_periods"]:
        lord = ch["lord"]
        lord = next(l for l in LORDS if lord.startswith(l))
        dev.append(ch["duration_days"] - parent * dict(zip(LORDS, YEARS))[lord] / 120)
    print("max AD deviation from exact proportion (days): %.4f" % max(abs(x) for x in dev))
