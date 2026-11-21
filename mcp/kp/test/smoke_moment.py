"""Live printout of the KP moment features (no assertions). Run: .venv/bin/python mcp/kp/test/smoke_moment.py"""
import os, sys, time
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from state import KPState, MomentState
import kp_core as core
import kp_chart as ch
import kp_moment as mo
import kp_dasha as da

BIRTH = dict(date_of_birth="1990-05-15", time_of_birth="10:30", place_name="Mumbai, IN", latitude=19.07,
             longitude=72.87, timezone="Asia/Kolkata")


def main():
    st = KPState(**BIRTH)
    dt = st.local_datetime
    with core.calc(st) as ctx:
        jd = ctx.jd_from_local(dt)
        natal = ch.chart_at(ctx, jd, st.latitude, st.longitude, dt)

        print("== ruling planets (birth moment)")
        rp = mo.ruling_planets(ctx, dt, st.latitude, st.longitude, True, natal)
        print(rp["day"], [(e["role"], e["planet"]) for e in rp["entries"]], rp["appearing_most"])

        print("== horary 100, 'start'")
        for pt in mo.LAGNA_POINTS:
            c, info = mo.cast_horary(ctx, 100, datetime(2026, 11, 20, 15, 0), 19.07, 72.87, pt)
            print(pt, info["division"]["start_dms"], info["division"]["sign"], info["division"]["sub_lord"],
                  "cusp_time", info["cusp_time_local"], info["lagna_check"], info["actual_ascendant_at_question"])
        for n in (1, 249, 7, 244):
            c, info = mo.cast_horary(ctx, n, datetime(2026, 11, 20, 15, 0), 19.07, 72.87, "middle")
            print("number", n, info["lagna_check"], info["cusp_time_local"])
        c, info = mo.cast_horary(ctx, 100, datetime(2026, 11, 20, 15, 0), 19.07, 72.87, "start")
        rph = mo.ruling_planets(ctx, datetime(2026, 11, 20, 15, 0), 19.07, 72.87, True, None, c)
        jh = mo.judge_horary(c, rph, event="marriage")
        print("judge_horary", jh["verdict"], jh["csl"]["sub_lord"], jh["ruling_planet_check"], jh["moon_sub_lord_check"])

        print("== transit snapshot with natal overlay")
        ts = mo.transit_snapshot(ctx, datetime(2026, 11, 20, 15, 0), natal)
        for r in ts["planets"]:
            print(r["planet"], r["dms"], r["sign"], r["star_lord"], r["sub_lord"], "natal H", r["natal_house"], r["retrograde"])

        print("== boundary changes: Moon sub, 2026-11-20..22")
        t0 = time.time()
        b = mo.find_boundary_changes(ctx, "Moon", datetime(2026, 11, 20), datetime(2026, 11, 22), "sub")
        print(b["change_count"], b["at_start"]["lord"], b["changes"][:3], round(time.time() - t0, 2), "s")
        for lv in ("sign", "star", "sub", "sub_sub"):
            t0 = time.time()
            b = mo.find_boundary_changes(ctx, "Sun", datetime(2026, 11, 20), datetime(2026, 12, 20), lv)
            print("Sun", lv, b["change_count"], round(time.time() - t0, 2), "s")
        b = mo.find_boundary_changes(ctx, "Saturn", datetime(2026, 1, 1), datetime(2027, 12, 31), "sub")
        print("Saturn sub 2y", b["change_count"], [(c["time"], c["from"], c["to"], c["retrograde"]) for c in b["changes"][:5]])
        b = mo.find_boundary_changes(ctx, "Rahu", datetime(2026, 1, 1), datetime(2028, 12, 31), "star")
        print("Rahu star 3y", [(c["time"], c["from"], c["to"]) for c in b["changes"]])

        print("== lagna changes +-30 min (birth)")
        lc = mo.lagna_changes(ctx, dt, st.latitude, st.longitude, 30, "sub")
        for r in lc["stretches"]:
            print(r["start"], r["end"], r["sign"], r["star_lord"], r["sub_lord"], r["minutes"])

        print("== triggers: Sun/Moon over Venus/Saturn star, Nov 2026")
        tg = mo.find_transit_triggers(ctx, ["Sun"], {"Venus": ["dasha"], "Saturn": ["bhukti"]}, datetime(2026, 11, 20),
                                      datetime(2027, 1, 31), ["star"])
        print(tg["count"], [(t["start"], t["end"], t["lord"]) for t in tg["triggers"]])

        print("== rectify (1 event)")
        t0 = time.time()
        rc = mo.rectify(ctx, dt, st.latitude, st.longitude, [{"date": "2018-02-10", "event": "marriage"}], 20, 60)
        print(rc["best_score"], rc["max_score_possible"], len(rc["best_ranges"]), round(time.time() - t0, 2), "s")
        for r in rc["best_ranges"][:3]:
            print(r["start"], r["end"], r["score"], r["asc_sub_lord"], r["events"][0]["csl"], r["events"][0]["verdict"])


main()
