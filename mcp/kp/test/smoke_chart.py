"""Quick live printout of a KP chart (no assertions). Run: .venv/bin/python mcp/kp/test/smoke_chart.py"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from state import KPState
import kp_core as core
import kp_chart as ch

BIRTH = dict(date_of_birth="1990-05-15", time_of_birth="10:30", place_name="Mumbai, IN", latitude=19.07,
             longitude=72.87, timezone="Asia/Kolkata")

def main():
    st = KPState(**BIRTH)
    with core.calc(st) as ctx:
        dt = st.local_datetime
        jd = ctx.jd_from_local(dt)
        chart = ch.chart_at(ctx, jd, st.latitude, st.longitude, dt)
        print("jd", round(jd, 5), "ayanamsa", round(ctx.ayanamsa(jd), 5), "system", chart.houses["system"])
        print("ASC", chart.asc, ch.kp_lords(chart.asc)["sign"], ch.kp_lords(chart.asc)["star_lord"], ch.kp_lords(chart.asc)["sub_lord"])
        for r in ch.cusp_rows(chart):
            print("cusp", r["house"], r["dms"], r["sign"], r["sign_lord"], r["star_lord"], r["sub_lord"], r["occupants"])
        for r in ch.planet_rows(chart):
            print(f'{r["planet"]:8s} {r["dms"]:>10s} {r["sign"]:11s} H{r["house"]:<2d} star={r["star_lord"]:8s} sub={r["sub_lord"]:8s} retro={r["retrograde"]}')
        for r in ch.planet_significations(chart):
            print(r["planet"], r["levels"], "->", r["signifies"], "no_planet_in_star" if r["no_planet_in_its_star"] else "")
        j = ch.judge_event(chart, event="marriage")
        print("MARRIAGE", j["csl"]["sub_lord"], j["csl"]["sub_lord_signifies"], j["verdict"], j["favourable_signified_by_csl"], j["denial_signified_by_csl"])
        print("fortuna", ch.fortuna(chart)["dms"], ch.fortuna(chart)["sign"])
        print("agents", chart.agents)

main()
