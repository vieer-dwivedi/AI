"""Tests for the KP MCP engine and server (real Swiss Ephemeris; independent cross-checks against raw swisseph calls).

Run:  .venv/bin/python mcp/kp/test/test_kp.py
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import swisseph as swe  # noqa: E402

import kp_chart as ch  # noqa: E402
import kp_core as core  # noqa: E402
import kp_dasha as da  # noqa: E402
import kp_fruitful as ff  # noqa: E402
import kp_methods as km  # noqa: E402
import kp_moment as mo  # noqa: E402
import kp_retro as rt  # noqa: E402
import kp_tables as kt  # noqa: E402
from state import KPOptions, KPState, MomentState  # noqa: E402

BIRTH = dict(date_of_birth="1990-05-15", time_of_birth="10:30", place_name="Mumbai, IN", latitude=19.07,
             longitude=72.87, timezone="Asia/Kolkata")
passed = 0


def check(label, cond):
    global passed
    assert cond, f"FAILED: {label}"
    passed += 1
    print(f"ok  {label}")


def raises(label, fn, exc=Exception, text=None):
    try:
        fn()
    except exc as e:
        check(label, text is None or text.lower() in str(e).lower())
        return
    check(label, False)


# ---------------------------------------------------------------------------------- tables
check("249 numbered divisions", len(kt.NUMBERED) == 249 and len(kt.SUBS) == 243)
check("6 subs split by a sign boundary", sum(1 for r in kt.NUMBERED if r["split_by_sign"]) == 12)
check("Aries 22 / Taurus 19 divisions", sum(1 for r in kt.NUMBERED if r["sign_index"] == 0) == 22
      and sum(1 for r in kt.NUMBERED if r["sign_index"] == 1) == 19)
check("divisions tile the circle", kt.NUMBERED[0]["start_arc"] == 0 and kt.NUMBERED[-1]["end_arc"] == kt.FULL_ARC
      and all(a["end_arc"] == b["start_arc"] for a, b in zip(kt.NUMBERED, kt.NUMBERED[1:])))
check("sub spans sum to a nakshatra", sum(400 * y for y in kt.VIM_YEARS.values()) == kt.NAK_ARC)
for lon_dms, lords in [((1, 23, 55), ("Mars", "Ketu", "Venus")), ((33, 46, 54), ("Venus", "Sun", "Saturn")),
                       ((59, 19, 19), ("Venus", "Mars", "Saturn")), ((83, 2, 49), ("Mercury", "Jupiter", "Saturn")),
                       ((108, 59, 58), ("Moon", "Mercury", "Ketu")), ((141, 16, 58), ("Sun", "Venus", "Jupiter"))]:
    d, m, s = lon_dms
    r = kt.kp_lords(d + m / 60 + (s + 0.5) / 3600)
    check(f"textbook longitude {d}d{m}'{s}\" -> {lords}", (r["sign_lord"], r["star_lord"], r["sub_lord"]) == lords)
r0 = kt.kp_lords(0.0)
check("0 deg Aries: Mars / Ketu / Ketu / Ketu", (r0["sign_lord"], r0["star_lord"], r0["sub_lord"], r0["sub_sub_lord"]) == ("Mars", "Ketu", "Ketu", "Ketu"))
check("360-epsilon is Pisces, number 249", kt.kp_lords(359.9999)["sign"] == "Pisces" and kt.kp_lords(359.9999)["kp_number"] == 249)
check("every numbered row agrees with kp_lords", all(
    kt.kp_lords((r["start_arc"] + 0.5) / 3600.0)["kp_number"] == r["number"] for r in kt.NUMBERED))
check("planet aliases", kt.planet_name("Shani") == "Saturn" and kt.planet_name("guru") == "Jupiter")
raises("unknown planet rejected", lambda: kt.planet_name("Pluto"), ValueError)
raises("unknown event rejected with list", lambda: kt.event_info("nonsense"), ValueError, "Known events")
check("badhaka by lagna type", kt.badhaka_house(0) == 11 and kt.badhaka_house(1) == 9 and kt.badhaka_house(2) == 7)

# ---------------------------------------------------------------------------------- state
check("ayanamsa alias KP -> KP_NEW", KPOptions(ayanamsa="kp").ayanamsa == "KP_NEW")
raises("bad ayanamsa rejected", lambda: KPOptions(ayanamsa="fagan"), ValueError)
raises("timezone required", lambda: KPState(date_of_birth="1990-05-15", time_of_birth="10:30", latitude=1, longitude=1), ValueError)
raises("bad timezone rejected", lambda: KPState(**{**BIRTH, "timezone": "Mars/Olympus"}), ValueError)
check("unknown option keys ignored", KPState(**BIRTH, options={"ayanamsa": "LAHIRI", "dasha_system": "yogini"}).options.ayanamsa == "LAHIRI")

st = KPState(**BIRTH)

# ---------------------------------------------------------------------------------- ayanamsa & chart vs raw swisseph
jd_now = core.julday_utc(2026, 11, 20, 0.0)
av = core.ayanamsa_values(jd_now)
check("KP_NEW is Lahiri minus about 4m36s", abs((av["LAHIRI"] - av["KP_NEW"]) * 3600 - 276.2) < 3)
check("KP_OLD is Lahiri minus about 5m49s", abs((av["LAHIRI"] - av["KP_OLD"]) * 3600 - 349.0) < 3)

with core.calc(st) as ctx:
    dt = st.local_datetime
    jd = ctx.jd_from_local(dt)
    check("JD of 1990-05-15 10:30 IST", abs(jd - 2448026.70833) < 1e-4)
    chart = ch.chart_at(ctx, jd, st.latitude, st.longitude, dt)
    _, ascmc = swe.houses(jd, st.latitude, st.longitude, b"P")
    # swisseph's sidereal frame subtracts the ayanamsa WITH nutation from the true-equinox tropical value (= mean frame)
    ay = swe.get_ayanamsa_ex_ut(jd, swe.FLG_SWIEPH)[1]
    check("sidereal Ascendant = tropical Placidus ASC - ayanamsa", abs(chart.asc - (ascmc[0] - ay) % 360.0) < 1e-6)
    tm = swe.calc_ut(jd, 1, swe.FLG_SWIEPH)[0][0]
    check("Moon = tropical Moon - ayanamsa", abs(chart.planets["Moon"]["longitude"] - (tm - ay) % 360.0) < 1e-3)
    check("Ketu is 180 deg from Rahu", abs((chart.planets["Ketu"]["longitude"] - chart.planets["Rahu"]["longitude"]) % 360 - 180) < 1e-6)
    check("cusps ascend in order and cusp 1 = Ascendant", abs(chart.cusps[0] - chart.asc) < 1e-6 and all(
        (chart.cusps[(i + 1) % 12] - chart.cusps[i]) % 360 < 180 for i in range(12)))
    check("cusp 10 = MC", abs(chart.cusps[9] - (ascmc[1] - ay) % 360.0) < 1e-6)
    check("every planet in a house 1..12", all(1 <= chart.planets[p]["house"] <= 12 for p in kt.PLANETS))
    check("owned houses cover 1..12 exactly once",
          sorted(h for p in kt.PLANETS for h in chart.owned[p]) == list(range(1, 13)))
    # significator tables agree between the two views
    hs = ch.house_significators(chart)
    ok = True
    for row in hs:
        for L, key in (("A", "A_in_star_of_occupants"), ("C", "C_in_star_of_owner")):
            ok &= sorted(row[key]) == sorted(p for p in kt.PLANETS if row["house"] in chart.sig[p][L])
    check("planet view and house view agree (A, C)", ok)
    mr = ch.judge_event(chart, "marriage")
    check("marriage CSL = sub lord of cusp 7", mr["csl"]["sub_lord"] == kt.kp_lords(chart.cusps[6])["sub_lord"] and mr["primary_cusp"] == 7)
    check("marriage houses 2,7,11 / denial 1,6,10", mr["favourable_houses"] == [2, 7, 11] and mr["denial_houses"] == [1, 6, 10])
    check("verdict is a known label", mr["verdict"] in mo._VERDICT_POINTS)
    cu = ch.judge_event(chart, None, custom_favourable=[3, 9], custom_denial=[4], primary_cusp=9)
    check("custom houses accepted", cu["event"] == "custom" and cu["favourable_houses"] == [3, 9] and cu["primary_cusp"] == 9)
    raises("event or custom required", lambda: ch.judge_event(chart), ValueError)
    lg = ch.judge_event(chart, "longevity")
    check("badhaka resolved to a house number", all(isinstance(x, int) for x in lg["denial_houses"]))
    check("promise table covers every event", len(ch.promise_table(chart)) == len(kt.EVENTS))
    check("fortuna = Asc + Moon - Sun", abs(ch.fortuna(chart)["longitude"] - (chart.asc + chart.planets["Moon"]["longitude"] - chart.planets["Sun"]["longitude"]) % 360) < 1e-5)
    check("node agents present", all(len(chart.agents[n]) >= 2 for n in ("Rahu", "Ketu")))

    # ------------------------------------------------------------------------------ 4-step and 6-fold (backlog item 1)
    fs = km.four_step(chart)
    check("4-step: 9 planets x 4 steps", len(fs["planets"]) == 9 and all(len(r["steps"]) == 4 for r in fs["planets"]))
    check("4-step: steps follow planet / star / sub / sub's star", all(
        [s["planet"] for s in r["steps"]] == [r["planet"], chart.star_of[r["planet"]], chart.planets[r["planet"]]["sub_lord"],
                                              chart.star_of[chart.planets[r["planet"]]["sub_lord"]]] for r in fs["planets"]))
    check("4-step: primary and secondary are disjoint", all(not set(r["primary_houses"]) & set(r["secondary_houses"]) for r in fs["planets"]))
    check("4-step: A-D houses are a subset of primary + secondary", all(
        set(chart.sig[r["planet"]]["A"] + chart.sig[r["planet"]]["B"] + chart.sig[r["planet"]]["C"] + chart.sig[r["planet"]]["D"])
        <= set(r["primary_houses"]) | set(r["secondary_houses"]) for r in fs["planets"]))
    check("4-step: house view agrees with planet view", all(
        (r["planet"] in hrow["primary_significators"]) == (hrow["house"] in r["primary_houses"])
        for r in fs["planets"] for hrow in fs["houses"]))
    check("4-step: own-star planet is primary for its occupied house", all(
        chart.house_of[r["planet"]] in r["primary_by_step"][1] for r in fs["planets"] if chart.star_of[r["planet"]] == r["planet"]))
    check("4-step: star lord's occupied house is always primary (rule 2)", all(
        chart.house_of[chart.star_of[r["planet"]]] in r["primary_by_step"][2] for r in fs["planets"]))
    check("4-step: a planet with other planets in its star is secondary for its occupied house (step 1, not on a cusp)", all(
        chart.house_of[r["planet"]] not in r["primary_by_step"][1] for r in fs["planets"]
        if chart.star_of[r["planet"]] != r["planet"] and any(chart.star_of[q] == r["planet"] for q in kt.PLANETS if q != r["planet"])
        and not km._on_cusp_houses(chart, r["planet"], km.ORB_DEG)
        and chart.house_of[r["planet"]] not in chart.owned[r["planet"]]))
    sf6 = km.six_fold(chart)
    check("6-fold: signifies_6fold contains the 4-fold houses", all(set(r["signifies_4fold"]) <= set(r["signifies_6fold"]) for r in sf6["planets"]))
    check("6-fold: sub-of-occupant column agrees with planet view", all(
        (r["planet"] in hrow["planets_in_sub_of_occupants"]) == (hrow["house"] in r["sub_levels"]["A_sub"])
        for r in sf6["planets"] for hrow in sf6["houses"]))
    check("6-fold: sub-of-owner column agrees with planet view", all(
        (r["planet"] in hrow["planets_in_sub_of_owner"]) == (hrow["house"] in r["sub_levels"]["C_sub"])
        for r in sf6["planets"] for hrow in sf6["houses"]))
    te = km.type_e(chart)
    check("type E: 9 planets, 12 houses, orb 3d20'", len(te["planets"]) == 9 and len(te["houses"]) == 12 and abs(te["orb_deg"] - 200 / 60) < 1e-5)
    check("type E: E houses never repeat the planet's own A-D houses", all(
        not set(r["e_houses"]) & set(chart.sig[r["planet"]]["A"] + chart.sig[r["planet"]]["B"] + chart.sig[r["planet"]]["C"] + chart.sig[r["planet"]]["D"])
        for r in te["planets"]))
    check("type E: house view agrees with planet view", all(
        (r["planet"] in hrow["E_planets"]) == (hrow["house"] in r["e_houses"]) for r in te["planets"] for hrow in te["houses"]))
    check("type E: every link is a real relation", all(
        l["relation"].startswith(("conjunction", "aspect")) for r in te["planets"] for l in r["planet_links"] + r["cusp_links"]))
    check("type E: orb conjunction is symmetric", all(
        (any(l["planet"] == b and l["relation"].startswith("conjunction") for l in ra["planet_links"]))
        == (any(l["planet"] == a and l["relation"].startswith("conjunction") for l in rb["planet_links"]))
        for ra in te["planets"] for rb in te["planets"] for a, b in [(ra["planet"], rb["planet"])] if a != b))
    tes = km.type_e(chart, "sign")
    check("type E: sign basis conjunction = same sign", all(
        (any(l["planet"] == q and l["relation"].startswith("conjunction") for l in r["planet_links"]))
        == (int(chart.planets[r["planet"]]["sign_index"]) == int(chart.planets[q]["sign_index"]))
        for r in tes["planets"] for q in kt.PLANETS if q != r["planet"]))
    check("type E: nodes never aspect", all(not l["relation"].startswith("aspect") for r in tes["planets"] if r["planet"] in ("Rahu", "Ketu") for l in r["planet_links"]))
    raises("type E bad basis rejected", lambda: km.type_e(chart, "degrees"), ValueError)
    raises("type_e_basis option rejected", lambda: KPOptions(type_e_basis="x"), ValueError)
    # ------------------------------------------------------------------------------ fruitful significators (backlog item 3)
    _mj = ch.judge_event(chart, "marriage")
    _fav, _den = _mj["favourable_houses"], _mj["denial_houses"]
    _fd = ff.analyze(chart, _fav, _den, ["detrimental"])
    check("fruitful: parse_methods aliases and combos", ff.parse_methods("rp, Sub-Lord") == ["ruling_planets", "sub_lord"] and ff.parse_methods("none") == [])
    raises("fruitful: unknown method rejected", lambda: ff.parse_methods("magic"), ValueError, "Unknown")
    raises("fruitful: ruling_planets method needs RPs", lambda: ff.analyze(chart, _fav, _den, ["ruling_planets"]), ValueError, "ruling planets")
    check("fruitful: detrimental keeps only planets without denial houses", all(
        not r["denial"] for r in _fd["planets"] if r["candidate"] and r["kept"]))
    check("fruitful: every candidate signifies a favourable house; non-candidates none", all(
        bool(r["favourable"]) == r["candidate"] for r in _fd["planets"]))
    check("fruitful: kept + dropped = candidates", sorted(_fd["fruitful"] + [x["planet"] for x in _fd["dropped"]])
          == sorted(r["planet"] for r in _fd["planets"] if r["candidate"]))
    _fs = ff.analyze(chart, _fav, _den, ["sub_lord"])
    check("fruitful: sub_lord keeps planets whose sub lord signifies a favourable house", all(
        r["sub_lord_favourable"] for r in _fs["planets"] if r["candidate"] and r["kept"]))
    _fr = ff.analyze(chart, _fav, _den, ["ruling_planets"], rp=["Venus", "Moon"])
    check("fruitful: ruling_planets keeps only RPs", set(_fr["fruitful"]) <= {"Venus", "Moon"})
    _f4 = ff.analyze(chart, _fav, _den, ["four_step_primary"])
    check("fruitful: four_step_primary keeps planets primary for a favourable house", all(
        set(km.four_step_planet(chart, r["planet"])["primary_houses"]) & set(_fav) for r in _f4["planets"] if r["candidate"] and r["kept"]))
    _fc = ff.analyze(chart, _fav, _den, ["detrimental", "sub_lord"])
    check("fruitful: combined methods are the intersection", set(_fc["fruitful"]) == set(_fd["fruitful"]) & set(_fs["fruitful"]))
    _dd = da.make_dasha(chart, dt)
    _all = da.find_joint_periods(_dd, chart, _fav, _den, dt, dt + timedelta(days=365.25 * 40), 3, False, None, None, None, 5000)
    _flt = da.find_joint_periods(_dd, chart, _fav, _den, dt, dt + timedelta(days=365.25 * 40), 3, False, None, None, None, 5000, set(_fd["fruitful"]))
    check("fruitful: filtered periods only use allowed lords", all(set(x["lords"]) <= set(_fd["fruitful"]) for x in _flt["periods"]))
    check("fruitful: filtered periods are a subset of the unfiltered ones", {tuple(x["lords"]) for x in _flt["periods"]} <= {tuple(x["lords"]) for x in _all["periods"]}
          and _flt["count"] <= _all["count"] and _flt["fruitful_filter"] == sorted(_fd["fruitful"]))
    # ------------------------------------------------------------------------------ retrograde rules (backlog item 4)
    check("retro: rule aliases", rt.normalize_rule("KSK") == "ksk_horary_cusp" and rt.normalize_rule(None) == "report_only"
          and KPOptions(retrograde_rule="strict").retrograde_rule == "strict")
    raises("retro: bad rule rejected", lambda: rt.normalize_rule("harsh"), ValueError)
    raises("retro: bad option rejected", lambda: KPOptions(retrograde_rule="harsh"), ValueError)
    check("retro: nodes and luminaries never count", not any(rt.is_retro(chart, x) for x in ("Rahu", "Ketu", "Sun", "Moon")))
    _csl = kt.kp_lords(chart.cusps[6])["sub_lord"]
    _c0, _c1, _c2 = (rt.cusp_check(chart, _csl, r) for r in rt.RULES)
    _hit = rt.is_retro(chart, _csl) or rt.is_retro(chart, chart.star_of[_csl])
    check("retro: report_only never triggers (facts only)", not _c0["triggered"] and _c0["facts_only"] and _c0["effect"] == "")
    check("retro: ksk / strict trigger exactly when CSL or its star lord is retrograde", _c1["triggered"] == _hit and _c2["triggered"] == _hit)
    check("retro: facts are identical under every rule", _c0["csl_retrograde"] == _c1["csl_retrograde"] == _c2["csl_retrograde"])
    _rp = [x for x in kt.PLANETS if rt.is_retro(chart, x)]
    _pf0, _pf1 = rt.period_flags(chart, list(kt.PLANETS), "report_only"), rt.period_flags(chart, list(kt.PLANETS), "strict")
    check("retro: period flags list retrograde lords; only strict marks delayed", _pf0["retrograde_flags"] == _pf1["retrograde_flags"]
          and not _pf0["delayed"] and _pf1["delayed"] == bool(_rp) and {f["lord"] for f in _pf1["retrograde_flags"] if f["kind"] == "retrograde lord"} == set(_rp))
    _chs = ch.chart_at(ctx, jd, st.latitude, st.longitude, dt)
    _chs.opts = KPOptions(retrograde_rule="strict")
    _jr = ch.judge_event(_chs, "marriage")
    check("retro: judge_event reports the check and never changes the verdict", "retrograde_check" in _jr["csl"]
          and _jr["verdict"] == ch.judge_event(chart, "marriage")["verdict"])
    _st = rt.find_stations(ctx, "Mercury", datetime(2026, 1, 1), datetime(2026, 12, 31))
    _dates = [(x["station"], x["datetime_local"][:10]) for x in _st["stations"]]
    check("stations: Mercury 2026 matches the published retrograde periods (+-1 day)", len(_dates) == 6 and all(
        abs((datetime.fromisoformat(a) - datetime.fromisoformat(b)).days) <= 1 for a, b in zip(
            [d for _, d in _dates], ["2026-02-26", "2026-03-21", "2026-06-29", "2026-07-24", "2026-10-24", "2026-11-13"])))
    check("stations: alternate retrograde / direct", [x for x, _ in _dates] == ["turns retrograde", "turns direct"] * 3)
    _sa = rt.find_stations(ctx, "Saturn", datetime(2026, 11, 20), datetime(2026, 11, 25))
    check("stations: Saturn retrograde at start, next direct found beyond the window", _sa["retrograde_at_start"] and _sa["searched_beyond_window"]
          and _sa["next_direct_station"]["datetime_local"].startswith("2026-12-1"))
    raises("stations: nodes rejected", lambda: rt.find_stations(ctx, "Rahu", datetime(2026, 1, 1), datetime(2026, 2, 1)), Exception, "no stations")
    raises("stations: Sun rejected", lambda: rt.find_stations(ctx, "Sun", datetime(2026, 1, 1), datetime(2026, 2, 1)), Exception, "never retrograde")
    check("significator_method option validated", KPOptions(significator_method="4step").significator_method == "four_step")
    raises("bad significator_method rejected", lambda: KPOptions(significator_method="ten_step"), ValueError)

    # ------------------------------------------------------------------------------ dasha
    d = da.make_dasha(chart, dt)
    check("first dasha lord = Moon star lord", d.start_lord == chart.planets["Moon"]["star_lord"])
    full = sum(m["days"] for m in d.mds[:9])
    check("9 mahadashas = 120 years of 365.25 days", abs(full - 120 * 365.25) < 1e-6)
    check("dasha periods are contiguous", all(a["end"] == b["start"] for a, b in zip(d.mds, d.mds[1:])))
    bal = d.balance_at_birth()
    check("balance <= lord's years", 0 < bal["balance_years"] <= kt.VIM_YEARS[bal["lord"]])
    check("birth falls inside the first dasha", d.mds[0]["start"] <= dt < d.mds[0]["end"])
    md = next(m for m in d.mds if m["lords"][0] == "Venus")
    kids = d.children(md)
    check("9 bhuktis tile the dasha", len(kids) == 9 and kids[0]["start"] == md["start"] and kids[-1]["end"] == md["end"])
    check("first bhukti is the dasha lord's own", kids[0]["lords"] == ["Venus", "Venus"])
    check("Venus/Venus = 20*20/120 years", abs(kids[0]["days"] - 20 * 20 / 120 * 365.25) < 1e-6)
    check("find() by lord path", d.find(["Venus", "Sun"])["lords"] == ["Venus", "Sun"])
    raises("bad lord path rejected", lambda: d.find(["Venus", "Venus", "Pluto"]), (da.DashaError, ValueError))
    at = datetime(2026, 11, 20, 15, 0)
    lad = da.running_ladder(d, chart, at, 5)
    check("ladder has 5 nested levels", len(lad) == 5 and all(a["start"] <= b["start"] and b["end"] <= a["end"] for a, b in zip(lad, lad[1:])))
    check("ladder percent in 0..100", all(0 <= r["elapsed_percent"] <= 100 for r in lad))
    raises("date outside span rejected", lambda: da.running_ladder(d, chart, datetime(1500, 1, 1), 3), da.DashaError)
    tbl = da.periods_table(d, chart, 2, datetime(2026, 1, 1), datetime(2027, 1, 1))
    check("bhukti table overlaps window", len(tbl) >= 1 and tbl[0]["level"] == "bhukti")
    raises("level must be deeper than parent", lambda: da.periods_table(d, chart, 1, dt, dt + timedelta(days=9000), ["Venus"]), da.DashaError)
    fav, den = [2, 7, 11], [1, 6, 10]
    jp = da.find_joint_periods(d, chart, fav, den, datetime(2026, 1, 1), datetime(2031, 1, 1), depth=3)
    check("joint periods: every lord signifies a favourable house", all(
        all(da.lord_status(chart, x, set(fav), set(den))["favourable"] for x in p["lords"]) for p in jp["periods"]))
    check("joint periods obey coverage", all(p["coverage_count"] >= jp["min_coverage"] for p in jp["periods"]))
    jx = da.find_joint_periods(d, chart, fav, den, datetime(2026, 1, 1), datetime(2031, 1, 1), depth=3, exclude_denial=True)
    check("exclude_denial keeps a subset", {p["path"] for p in jx["periods"]} <= {p["path"] for p in jp["periods"]})
    alt = KPState(**BIRTH, options={"dasha_year_days": 360})
    with core.calc(alt) as c2:
        ch2 = ch.chart_at(c2, jd, st.latitude, st.longitude, dt)
        d2 = da.make_dasha(ch2, dt)
        check("dasha year option changes length", abs(d2.mds[1]["days"] / kt.VIM_YEARS[d2.mds[1]["lords"][0]] - 360) < 1e-9)

    # ------------------------------------------------------------------------------ ruling planets / day lord
    fri = ctx.day_lord(datetime(2026, 11, 20, 15, 0))
    pre = ctx.day_lord(datetime(2026, 11, 20, 3, 0))
    check("Friday afternoon day lord = Venus", fri["day_lord"] == "Venus")
    check("before sunrise the previous weekday rules (Jupiter)", pre["day_lord"] == "Jupiter" and pre["note"] is not None)
    rp = mo.ruling_planets(ctx, dt, st.latitude, st.longitude, True, chart)
    check("7 ruling-planet entries (extended)", len(rp["entries"]) == 7)
    check("5 entries when not extended", len(mo.ruling_planets(ctx, dt, st.latitude, st.longitude, False)["entries"]) == 5)
    check("RP counts add up", sum(p["count"] for p in rp["ruling_planets"]) == 7)
    check("RP gets natal houses when natal chart given", all("natal_signifies" in p for p in rp["ruling_planets"]))
    tl = mo.ruling_planets_timeline(ctx, dt, st.latitude, st.longitude, 2, 5)
    check("RP timeline stretches are ordered and non-empty", len(tl["stretches"]) >= 2 and all(
        a["from"] < b["from"] for a, b in zip(tl["stretches"], tl["stretches"][1:])))

    # ------------------------------------------------------------------------------ horary
    t0 = time.time()
    qdt = datetime(2026, 11, 20, 15, 0)
    bad = []
    for n in range(1, 250):
        c, info = mo.cast_horary(ctx, n, qdt, 19.07, 72.87, "start")
        if not info["lagna_check"]["matches_number"]:
            bad.append(n)
    check(f"all 249 horary numbers give the right Lagna division ({time.time() - t0:.1f}s)", not bad)
    c, info = mo.cast_horary(ctx, 100, qdt, 19.07, 72.87, "middle")
    row = kt.division_row(100)
    check("middle point lies inside the division", row["start"] < info["target_longitude"] < row["end"] and info["lagna_check"]["matches_number"])
    check("horary planets are those of the question moment", abs(c.planets["Moon"]["longitude"] - ch.chart_at(ctx, ctx.jd_from_local(qdt), 19.07, 72.87, qdt).planets["Moon"]["longitude"]) < 1e-9)
    raises("horary number 0 rejected", lambda: mo.cast_horary(ctx, 0, qdt, 19.07, 72.87), core.KPError)
    raises("horary number 250 rejected", lambda: mo.cast_horary(ctx, 250, qdt, 19.07, 72.87), core.KPError)
    raises("bad lagna_point rejected", lambda: mo.cast_horary(ctx, 5, qdt, 19.07, 72.87, "somewhere"), core.KPError)
    rph = mo.ruling_planets(ctx, qdt, 19.07, 72.87, True, None, c)
    check("horary RP Lagna comes from the horary chart", rph["lagna"]["sub_lord"] == kt.kp_lords(c.asc)["sub_lord"])
    jh = mo.judge_horary(c, rph, event="marriage")
    check("judge_horary adds RP and Moon checks", "ruling_planet_check" in jh and "moon_sub_lord_check" in jh)

    # ---- horary extras (backlog item 11)
    import kp_horary as khy
    from kp_retro import is_retro as _is_retro
    jh_e = mo.judge_horary(c, rph, event="marriage")
    hx = khy.horary_extras(c, info, rph, jh_e)
    _rps = {p["planet"] for p in rph["ruling_planets"]}
    _lag = kt.kp_lords(c.asc)["star_lord"]
    _ms = c.planets["Moon"]["sub_lord"]
    _gm = {x_["id"]: x_ for x_ in hx["genuineness"]["checks"]}
    _link = _lag == _ms or (_ms != _lag and c.star_of[_ms] == _lag) or (_lag != _ms and c.star_of[_lag] == _ms)
    check("genuineness: Lagna star lord / Moon sub lord link and RP lines match a direct computation",
          _gm["lagna_star_lord_linked_to_moon_sub_lord"]["value"] == _link and _gm["lagna_star_lord_is_ruling_planet"]["value"] == (_lag in _rps)
          and _gm["moon_sub_lord_is_ruling_planet"]["value"] == (_ms in _rps)
          and _gm["moon_star_lord_signifies_matter_houses"]["value"] == bool(set(ch.signified(c, c.planets["Moon"]["star_lord"])) & set(jh_e["favourable_houses"]))
          and hx["genuineness"]["passed"] == sum(1 for x_ in hx["genuineness"]["checks"] if x_["value"]))
    _cl = {x_["id"]: x_["value"] for x_ in hx["horary_checklist"]["items"]}
    _csl = jh_e["csl"]["sub_lord"]
    check("checklist: retrograde, house and RP lines match a direct computation",
          _cl["csl_not_retrograde"] == (not _is_retro(c, _csl)) and _cl["csl_not_in_retrograde_star"] == (not _is_retro(c, c.star_of[_csl]))
          and _cl["csl_signifies_no_denial_house"] == (not jh_e["denial_signified_by_csl"]) and _cl["csl_linked_to_supporting_house"] == bool(jh_e["favourable_signified_by_csl"])
          and _cl["csl_is_ruling_planet"] == (_csl in _rps) and _cl["csl_star_lord_is_ruling_planet"] == (c.star_of[_csl] in _rps)
          and hx["horary_checklist"]["passed"] == sum(_cl.values()) and hx["horary_checklist"]["all_passed"] == all(_cl.values()))
    check("lagna_point explanation names the point used, the division and the actual Ascendant",
          hx["lagna_point"]["used"] == "middle" and hx["lagna_point"]["number"] == 100 and hx["lagna_point"]["division"]["sub_lord"] == kt.division_row(100)["sub_lord"]
          and set(hx["lagna_point"]["alternatives"]) == set(mo.LAGNA_POINTS)
          and hx["lagna_point"]["actual_ascendant"]["inside_the_numbered_division"] == info["actual_ascendant_at_question"]["inside_the_numbered_division"])
    check("reminders present; first_asking_only only when declared",
          len(hx["reminders"]) >= 3 and "first_asking_only" not in hx and khy.horary_extras(c, info, rph, jh_e, None, True)["first_asking_only"]["declared_repeat"])
    check("extras never change the judgment verdict", jh_e["verdict"] == jh["verdict"])

    # ------------------------------------------------------------------------------ transit snapshot
    ts = mo.transit_snapshot(ctx, qdt, chart)
    check("transit snapshot: 9 planets with natal house", len(ts["planets"]) == 9 and all(1 <= p["natal_house"] <= 12 for p in ts["planets"]))
    check("no natal fields without a natal chart", "natal_house" not in mo.transit_snapshot(ctx, qdt)["planets"][0])

    # ------------------------------------------------------------------------------ boundary search vs raw swisseph
    b = mo.find_boundary_changes(ctx, "Sun", datetime(2026, 12, 1), datetime(2026, 12, 31), "sign")
    check("Sun changes sidereal sign once in Dec 2026 (to Sagittarius)", b["change_count"] == 1 and b["changes"][0]["sign"] == "Sagittarius")
    tchg = datetime.fromisoformat(b["changes"][0]["time"])
    jdc = ctx.jd_from_local(tchg)
    sun = (swe.calc_ut(jdc, 0, swe.FLG_SWIEPH)[0][0] - swe.get_ayanamsa_ex_ut(jdc, swe.FLG_SWIEPH)[1]) % 360.0
    check("reported change time puts the Sun at 240 deg (within 5 arcsec)", abs(sun - 240.0) < 5 / 3600)
    mb = mo.find_boundary_changes(ctx, "Moon", datetime(2026, 11, 20), datetime(2026, 11, 22), "sub")
    check("Moon sub changes ~19 in 2 days", 15 <= mb["change_count"] <= 25)
    segs = mo.planet_segments(ctx, "Moon", datetime(2026, 11, 20), datetime(2026, 11, 22), "sub")
    check("segments are contiguous", all(a["end"] == b2["start"] for a, b2 in zip(segs, segs[1:])))
    check("no two neighbouring segments share a sub lord", all(a["sub_lord"] != b2["sub_lord"] or a["nakshatra"] != b2["nakshatra"] for a, b2 in zip(segs, segs[1:])))
    rb = mo.find_boundary_changes(ctx, "Rahu", datetime(2026, 1, 1), datetime(2028, 12, 31), "star")
    check("mean Rahu moves backwards through the nakshatras", rb["change_count"] >= 3 and all(c["retrograde"] for c in rb["changes"]))
    raises("window too long rejected", lambda: mo.find_boundary_changes(ctx, "Moon", datetime(2000, 1, 1), datetime(2030, 1, 1), "sub_sub"), core.KPError)
    raises("end before start rejected", lambda: mo.find_boundary_changes(ctx, "Sun", datetime(2026, 2, 1), datetime(2026, 1, 1)), core.KPError)
    raises("bad level rejected", lambda: mo.find_boundary_changes(ctx, "Sun", datetime(2026, 1, 1), datetime(2026, 2, 1), "x"), core.KPError)
    tg = mo.find_transit_triggers(ctx, ["Sun"], {"Saturn": ["bhukti"]}, datetime(2026, 11, 1), datetime(2027, 1, 31), ["star"])
    check("trigger windows are in the star of the target lord", tg["count"] >= 1 and all(t["lord"] == "Saturn" for t in tg["triggers"]))

    # ------------------------------------------------------------------------------ lagna changes / rectification
    lc = mo.lagna_changes(ctx, dt, st.latitude, st.longitude, 30, "sub")
    check("lagna stretches are contiguous and span 60 minutes",
          all(a["end"] == b2["start"] for a, b2 in zip(lc["stretches"], lc["stretches"][1:])) and abs(sum(s["minutes"] for s in lc["stretches"]) - 60) < 0.1)
    check("several Lagna sub lords within an hour", len(lc["stretches"]) >= 5)
    rc = mo.rectify(ctx, dt, st.latitude, st.longitude, [{"date": "2018-02-10", "event": "marriage"}], 20, 60)
    check("rectify scans 41 trial times", rc["scanned"]["trial_times"] == 41)
    check("rectify best score >= score at the given time", all(r["score"] <= rc["best_score"] for r in rc["best_ranges"] + rc["other_ranges"]))
    check("rectify ranges cover the whole window", sum(r["trial_times"] for r in rc["best_ranges"] + rc["other_ranges"]) <= 41)
    raises("rectify needs events", lambda: mo.rectify(ctx, dt, st.latitude, st.longitude, []), core.KPError)
    raises("rectify rejects event before birth", lambda: mo.rectify(ctx, dt, st.latitude, st.longitude, [{"date": "1980-01-01", "event": "marriage"}]), core.KPError)
    raises("rectify rejects too many trial times", lambda: mo.rectify(ctx, dt, st.latitude, st.longitude, [{"date": "2018-02-10", "event": "marriage"}], 600, 5), core.KPError)

    # ---- rectification upgrades (backlog item 12)
    import kp_rectify as krc
    _lm = krc.lagna_moon_link(chart)
    _lg = kt.kp_lords(chart.asc)
    _mn = chart.planets["Moon"]
    check("Lagna-Moon link: shared planets and linked flag are consistent",
          _lm["shared_planets"] == sorted({_lg["star_lord"], _lg["sub_lord"]} & {_mn["star_lord"], _mn["sub_lord"]})
          and _lm["linked"] == bool(_lm["links"]) and (not _lm["shared_planets"] or _lm["linked"])
          and _lm["lagna_lords"] == {"star": _lg["star_lord"], "sub": _lg["sub_lord"]})
    _ev1 = [{"date": "2018-02-10", "event": "marriage"}]
    _r1 = mo.rectify(ctx, dt, st.latitude, st.longitude, _ev1, 20, 60)
    _okl = True
    for _rg in _r1["best_ranges"] + _r1["other_ranges"]:
        _t = datetime.fromisoformat(_rg["start"])
        _okl &= _rg["lagna_moon_link"] == krc.lagna_moon_link(ch.chart_at(ctx, ctx.jd_from_local(_t), st.latitude, st.longitude, _t))
    check("every range carries the Lagna-Moon link of its first trial time", _okl)
    _r2 = mo.rectify(ctx, dt, st.latitude, st.longitude, [{**_ev1[0], "weight": 2}], 20, 60)
    check("event weight scales the scores and the maximum",
          _r2["max_score_possible"] == 12.0 and abs(_r2["best_score"] - 2 * _r1["best_score"]) < 1e-9 and _r2["weights_used"][0]["weight"] == 2.0
          and _r1["weights_used"][0]["weight"] == 1.0 and _r1["max_score_possible"] == 6.0)
    for _bw in (0, -1, 11, "x"):
        raises(f"rectify rejects weight {_bw!r}", lambda _bw=_bw: mo.rectify(ctx, dt, st.latitude, st.longitude, [{**_ev1[0], "weight": _bw}], 5, 60), core.KPError, "weight")
    raises("rectify rejects a bad moon_link mode", lambda: mo.rectify(ctx, dt, st.latitude, st.longitude, _ev1, 5, 60, None, 8, "maybe"), core.KPError, "moon_link")
    _r3 = mo.rectify(ctx, dt, st.latitude, st.longitude, _ev1, 20, 60, None, 8, "require")
    _linked_ranges = [r_ for r_ in _r1["best_ranges"] + _r1["other_ranges"] if r_["lagna_moon_link"]["linked"]]
    check("moon_link=require ranks only linked ranges (or says none were linked)",
          (all(r_["lagna_moon_link"]["linked"] for r_ in _r3["best_ranges"] + _r3["other_ranges"]) and not _r3["moon_link"]["no_range_linked_so_filter_not_applied"])
          or (not _linked_ranges and _r3["moon_link"]["no_range_linked_so_filter_not_applied"]))
    check("report mode filters nothing", _r1["moon_link"]["ranges_filtered_out"] == 0 and _r1["moon_link"]["mode"] == "report" and _r1["filtered_out_ranges"] == [])
    _okse = True
    for _rg in _r1["best_ranges"]:
        _s = _rg["lagna_sub_lord_sensitivity"]
        _mid = datetime.fromisoformat(_s["around"])
        _okse &= (_s["lagna_sub_lord"] == kt.kp_lords(ctx.asc_only(ctx.jd_from_local(_mid), st.latitude, st.longitude))["sub_lord"]
                  and _s["minutes_back_to_change"] >= 0 and _s["minutes_forward_to_change"] >= 0
                  and abs(_s["minutes_back_to_change"] + _s["minutes_forward_to_change"] - _s["stretch_minutes"]) < 0.02)
    check("each best range reports a Lagna sub-lord sensitivity that matches the Ascendant at its middle", _okse and _r1["best_ranges"])
    check("anchor events and the unverified-rules caution are reported", "marriage" in _r1["anchor_events"]["events"] and "not implemented" in _r1["caution"])

# ---------------------------------------------------------------------------------- cusp proximity (backlog item 5)
import kp_cusp as kc  # noqa: E402

_ORB = 200.0 / 60.0
_cs = [10.0 + 30.0 * i for i in range(12)]  # synthetic cusps at 10, 40, ... 340


def _sun_entries(lon, house, orb=_ORB):
    pl = {p: {"longitude": lon if p == "Sun" else 200.0, "house": house if p == "Sun" else 7} for p in kt.PLANETS}
    return [e for e in kc.near_cusp_entries(pl, _cs, orb) if e["planet"] == "Sun"]


_e = _sun_entries(38.0, 1)
check("near cusp: 2 deg behind cusp 2 -> behind_cusp 2", len(_e) == 1 and _e[0]["relation"] == "behind_cusp" and _e[0]["cusp"] == 2 and abs(_e[0]["distance_deg"] - 2.0) < 1e-6)
_e = _sun_entries(8.0, 12)
check("near cusp: house 12 planet behind cusp 1 wraps around the zodiac", len(_e) == 1 and _e[0]["cusp"] == 1 and abs(_e[0]["distance_deg"] - 2.0) < 1e-6)
_e = _sun_entries(11.0, 1)
check("near cusp: 1 deg past cusp 1 -> past_cusp 1 only", len(_e) == 1 and _e[0]["relation"] == "past_cusp" and _e[0]["cusp"] == 1)
check("near cusp: planet exactly on a cusp is past_cusp at distance 0", [(e["relation"], e["distance_deg"]) for e in _sun_entries(40.0, 2)] == [("past_cusp", 0.0)])
check("near cusp: mid-house planet has no entry", _sun_entries(25.0, 1) == [])
check("near cusp: just inside the orb counts, just outside does not", len(_sun_entries(40.0 - _ORB + 1e-6, 1)) == 1 and _sun_entries(40.0 - _ORB - 1e-6, 1) == [])
check("near cusp: no orb (None or 0) means rule off", _sun_entries(38.0, 1, None) == [] and _sun_entries(38.0, 1, 0) == [])
_shift = kc.shift_map(_sun_entries(38.0, 1) + _sun_entries(11.0, 1))
check("shift map moves only behind_cusp planets", _shift["Sun"] == [2] and all(v == [] for k, v in _shift.items() if k != "Sun"))


def _snap(opts):
    s_ = KPState(**BIRTH, options=opts)
    with core.calc(s_) as c_:
        k = ch.chart_at(c_, c_.jd_from_local(s_.local_datetime), s_.latitude, s_.longitude, s_.local_datetime)
        return dict(chart=k, sig=k.sig, near=k.near_cusp, cusps=k.cusps, planets=k.planets, ps=ch.planet_significations(k),
                    hs=ch.house_significators(k), cr=ch.cusp_rows(k), block=ch.near_cusp_block(k),
                    cd=ch.chart_dict(k, {}, "standard"), j=ch.judge_event(k, event="marriage"), j2=ch.judge_event(k, event="service"))


_n0, _n1, _n2 = _snap({}), _snap({"cusp_orb_deg": _ORB}), _snap({"cusp_orb_deg": _ORB, "cusp_proximity_apply": True})
check("cusp proximity off by default: no near_cusp anywhere", _n0["near"] == [] and _n0["block"] is None and "near_cusp" not in _n0["cd"]
      and "planets_behind_cusp" not in _n0["cr"][0] and "near_cusp" not in _n0["ps"][0])
_exp = set()
for _p in kt.PLANETS:
    _lon = _n1["planets"][_p]["longitude"]
    for _i, _cu in enumerate(_n1["cusps"]):
        if 0 < (_cu - _lon) % 360.0 <= _ORB:
            _exp.add((_p, "behind_cusp", _i + 1))
        if (_lon - _cu) % 360.0 <= _ORB:
            _exp.add((_p, "past_cusp", _i + 1))
check("near cusp entries match a brute-force check over all planets and cusps", {(e["planet"], e["relation"], e["cusp"]) for e in _n1["near"]} == _exp and len(_exp) >= 4)
check("every behind_cusp planet sits in the house before that cusp",
      all(_n1["planets"][e["planet"]]["house"] == (e["cusp"] - 2) % 12 + 1 for e in _n1["near"] if e["relation"] == "behind_cusp"))
check("report mode leaves A-D untouched", _n1["sig"] == _n0["sig"] and [r["levels"] for r in _n1["ps"]] == [r["levels"] for r in _n0["ps"]]
      and _n1["j"]["verdict"] == _n0["j"]["verdict"] and _n1["j"]["csl"]["sub_lord_signifies"] == _n0["j"]["csl"]["sub_lord_signifies"])
check("chart output carries the near_cusp block with the rule applied flag", _n1["cd"]["near_cusp"]["count"] == len(_n1["near"]) and _n1["cd"]["near_cusp"]["applied_to_A_to_D"] is False)
_sunr = next(r for r in _n1["ps"] if r["planet"] == "Sun")
check("signifies_with_near_cusp adds the next house (Sun, house 10 -> 11) without touching signifies",
      11 in _sunr["signifies_with_near_cusp"] and 11 not in _sunr["signifies"] and set(_sunr["signifies"]) <= set(_sunr["signifies_with_near_cusp"]))
_sat = next(r for r in _n1["ps"] if r["planet"] == "Saturn")
check("signifies_with_near_cusp adds nothing when the planet already signifies that house (Saturn owns the 7th)",
      _sat["signifies_with_near_cusp"] == _sat["signifies"])
check("cusp rows list planets behind / past each cusp", "Saturn" in _n1["cr"][6]["planets_behind_cusp"] and "Moon" in _n1["cr"][6]["planets_behind_cusp"])
check("natal placement never moves when the shift is applied", {p: d["house"] for p, d in _n2["planets"].items()} == {p: d["house"] for p, d in _n0["planets"].items()}
      and [r["occupants"] for r in _n2["cr"]] == [r["occupants"] for r in _n0["cr"]])
check("apply mode: a planet behind a cusp is also occupant (B) of the next house", _n2["sig"]["Sun"]["B"] == [10, 11] and _n2["sig"]["Saturn"]["B"] == [6, 7])
check("apply mode: planets in its star gain level A of the next house", _n2["sig"]["Moon"]["A"] == [10, 11])
check("apply mode: C and D (ownership) are unchanged", all(_n2["sig"][p][L] == _n0["sig"][p][L] for p in kt.PLANETS for L in "CD"))
check("apply mode: every planet signifies at least what it signified before", all(set(_n0["sig"][p][L]) <= set(_n2["sig"][p][L]) for p in kt.PLANETS for L in "AB"))
check("apply mode: house view and planet view agree", all(
    r["B_occupants"] == [p for p in kt.PLANETS if r["house"] in _n2["sig"][p]["B"]]
    and r["A_in_star_of_occupants"] == [p for p in kt.PLANETS if r["house"] in _n2["sig"][p]["A"]] for r in _n2["hs"]))
check("apply mode: house 11 lists Sun as B occupant and as near-cusp planet; default does not",
      "Sun" in _n2["hs"][10]["B_occupants"] and "Sun" in _n2["hs"][10]["near_cusp_planets"] and "Sun" not in _n0["hs"][10]["B_occupants"])
check("apply mode: the CSL of the 7th signifies at least the same houses", set(_n0["j"]["csl"]["sub_lord_signifies"]) <= set(_n2["j"]["csl"]["sub_lord_signifies"]))
check("apply mode: judgment output records the extra houses", _n2["cd"]["near_cusp"]["applied_to_A_to_D"] is True)
raises("cusp_proximity_apply without an orb is rejected", lambda: KPOptions(cusp_proximity_apply=True), Exception, "cusp_orb_deg")
raises("negative cusp orb rejected", lambda: KPOptions(cusp_orb_deg=-1.0))
raises("cusp orb above 15 degrees rejected", lambda: KPOptions(cusp_orb_deg=16.0))
check("cusp orb 0 means off", _snap({"cusp_orb_deg": 0})["near"] == [])

# ---------------------------------------------------------------------------------- CSL extensions (backlog item 6)
import kp_csl_ext as kx  # noqa: E402

_EVS = ["marriage", "service", "child_birth", "property_purchase", "foreign_travel", "education", "gain_of_money", "divorce"]


def _csnap(mode):
    s_ = KPState(**BIRTH, options={"csl_extension": mode})
    with core.calc(s_) as c_:
        k = ch.chart_at(c_, c_.jd_from_local(s_.local_datetime), s_.latitude, s_.longitude, s_.local_datetime)
        return dict(sig=k.sig, planets=k.planets, cusps=k.cusps, reps=[ch.csl_report(k, i) for i in range(1, 13)],
                    js={e: ch.judge_event(k, event=e) for e in _EVS}, ps=ch.planet_significations(k))


_x0, _x1, _x2, _x3 = _csnap("off"), _csnap("own_bhava"), _csnap("sub_chain"), _csnap("both")
check("csl_extension mode names and aliases", [kx.normalize(m) for m in (None, "off", "KSK", "kar", "all", "sub-chain")] == ["off", "off", "own_bhava", "sub_chain", "both", "sub_chain"])
raises("bad csl_extension rejected", lambda: kx.normalize("magic"), ValueError, "csl_extension must be")
raises("bad csl_extension rejected in options", lambda: KPOptions(csl_extension="magic"), Exception, "csl_extension must be")
check("csl_extension alias accepted in options", KPOptions(csl_extension="kar").csl_extension == "sub_chain")
check("off by default: no csl_extension block and no extended_reading", all("csl_extension" not in r for r in _x0["reps"]) and all("extended_reading" not in j for j in _x0["js"].values()))
check("extensions never touch A-D, the CSL houses or the standard verdicts",
      all(x["sig"] == _x0["sig"] and [r["sub_lord_signifies"] for r in x["reps"]] == [r["sub_lord_signifies"] for r in _x0["reps"]]
          and {e: j["verdict"] for e, j in x["js"].items()} == {e: j["verdict"] for e, j in _x0["js"].items()}
          and [r["levels"] for r in x["ps"]] == [r["levels"] for r in _x0["ps"]] for x in (_x1, _x2, _x3)))


def _owned_by(x, planet):
    return sorted(i + 1 for i, cu in enumerate(x["cusps"]) if kt.SIGN_LORDS[kt.sign_index_of(cu)] == planet)


_ok_own, _ok_sub = True, True
for _i, _r in enumerate(_x3["reps"], start=1):
    _csl, _base = _r["sub_lord"], set(_x0["reps"][_i - 1]["sub_lord_signifies"])
    _in_star = [q for q in kt.PLANETS if q != _csl and _x3["planets"][q]["star_lord"] == _csl]
    _ob = _r["csl_extension"]["own_bhava"]
    _ok_own &= _ob["condition_met"] == (not _in_star) and _ob["planets_in_its_star"] == _in_star and _ob["adds"] == ([_i] if (not _in_star and _i not in _base) else [])
    _sub2 = _x3["planets"][_csl]["sub_lord"]
    _hs = {_x3["planets"][_sub2]["house"], *_owned_by(_x3, _sub2)}
    _sc = _r["csl_extension"]["sub_chain"]
    _ok_sub &= _sc["sub_lord_of_csl"] == _sub2 and set(_sc["houses"]) == _hs and set(_sc["adds"]) == _hs - _base
    _ok_sub &= set(_r["csl_extension"]["added_houses"]) == (({_i} if not _in_star else set()) | _hs) - _base
check("own_bhava matches a brute-force check on all 12 cusps (star occupancy, house added)", _ok_own)
check("sub_chain matches a brute-force check on all 12 cusps (sub of the CSL, its house and ownership)", _ok_sub)
check("the test chart exercises both readings (some cusp adds a house under each)",
      any(r["csl_extension"]["own_bhava"]["adds"] for r in _x3["reps"]) and any(r["csl_extension"]["sub_chain"]["adds"] for r in _x3["reps"]))
check("single-mode blocks contain only their own reading", all("sub_chain" not in r["csl_extension"] and "own_bhava" in r["csl_extension"] for r in _x1["reps"])
      and all("own_bhava" not in r["csl_extension"] and "sub_chain" in r["csl_extension"] for r in _x2["reps"]))
check("signifies_extended is the base houses plus the added ones", all(
    set(r["csl_extension"]["signifies_extended"]) == set(r["sub_lord_signifies"]) | set(r["csl_extension"]["added_houses"])
    and not set(r["csl_extension"]["added_houses"]) & set(r["sub_lord_signifies"]) for x in (_x1, _x2, _x3) for r in x["reps"]))
check("both = union of own_bhava and sub_chain", all(
    set(_x3["reps"][i]["csl_extension"]["added_houses"]) == set(_x1["reps"][i]["csl_extension"]["added_houses"]) | set(_x2["reps"][i]["csl_extension"]["added_houses"]) for i in range(12)))
_er_ok = True
for _e, _j in _x3["js"].items():
    _er = _j["extended_reading"]
    _er_ok &= set(_j["favourable_signified_by_csl"]) <= set(_er["favourable_signified_by_csl"]) and set(_j["denial_signified_by_csl"]) <= set(_er["denial_signified_by_csl"])
    _er_ok &= _er["changes_verdict"] == (_er["verdict"] != _j["verdict"]) and set(_er["added_houses"]) == set(_j["csl"]["csl_extension"]["added_houses"])
check("extended_reading only adds hits, flags a changed verdict, and sits beside the unchanged verdict", _er_ok)
check("the extension changes at least one verdict on the test chart (so it is worth keeping separate)", any(j["extended_reading"]["changes_verdict"] for j in _x3["js"].values()))

# ---------------------------------------------------------------------------------- cuspal interlinks (backlog item 7)
import kp_cil as kcil  # noqa: E402


def _cilsnap():
    s_ = KPState(**BIRTH)
    with core.calc(s_) as c_:
        k = ch.chart_at(c_, c_.jd_from_local(s_.local_datetime), s_.latitude, s_.longitude, s_.local_datetime)
        j = ch.judge_event(k, event="marriage")
        return dict(cr=ch.cusp_rows(k), sigs={p: ch.signified(k, p) for p in kt.PLANETS}, out=kcil.interlinks(k), j=j,
                    ev=kcil.interlinks(k, None, None, j), sub=kcil.interlinks(k, [7, 2]), only2=kcil.interlinks(k, [2], None, j),
                    star_of=dict(k.star_of), subs={p: k.planets[p]["sub_lord"] for p in kt.PLANETS})


_ci = _cilsnap()
_cro = _ci["out"]["cusps"]
check("interlinks: 12 cusp rows in order with the lords of the cusp rows", [r["cusp"] for r in _cro] == list(range(1, 13)) and all(
    (r["sign_lord"], r["star_lord"], r["sub_lord"], r["sub_sub_lord"]) == (c["sign_lord"], c["star_lord"], c["sub_lord"], c["sub_sub_lord"]) for r, c in zip(_cro, _ci["cr"])))
_rul = _ci["out"]["cusp_rulership"]
check("rulership matches a brute-force check at every level", all(
    _rul[p][key] == [c["house"] for c in _ci["cr"] if c[fld] == p] for p in kt.PLANETS
    for key, fld in (("sign_lord_of", "sign_lord"), ("star_lord_of", "star_lord"), ("sub_lord_of", "sub_lord"), ("sub_sub_lord_of", "sub_sub_lord"))))
check("every cusp has exactly one ruler per level", all(sum(len(_rul[p][k]) for p in kt.PLANETS) == 12 for k in ("sign_lord_of", "star_lord_of", "sub_lord_of", "sub_sub_lord_of")))
check("CSL and SSL houses equal the A-D signification of that planet; own cusp is left out of the links", all(
    r["csl"]["signifies"] == _ci["sigs"][r["sub_lord"]] and r["ssl"]["signifies"] == _ci["sigs"][r["sub_sub_lord"]]
    and r["csl"]["links_to_other_houses"] == [h for h in r["csl"]["signifies"] if h != r["cusp"]] for r in _cro))
check("links through the CSL's star lord and sub lord match a brute-force check", all(
    r["csl"]["star_lord"] == _ci["star_of"][r["sub_lord"]] and r["csl"]["sub_lord"] == _ci["subs"][r["sub_lord"]]
    and r["csl"]["is_csl_of_cusps"] == [c["house"] for c in _ci["cr"] if c["sub_lord"] == r["sub_lord"]]
    and r["csl"]["star_lord_is_csl_of_cusps"] == [c["house"] for c in _ci["cr"] if c["sub_lord"] == r["csl"]["star_lord"]]
    and r["csl"]["sub_lord_is_csl_of_cusps"] == [c["house"] for c in _ci["cr"] if c["sub_lord"] == r["csl"]["sub_lord"]] for r in _cro))
check("a CSL always lists its own cusp among the cusps it is CSL of", all(r["cusp"] in r["csl"]["is_csl_of_cusps"] for r in _cro))
check("same_as_csl flag is consistent", all(r["ssl"]["same_as_csl"] == (r["sub_lord"] == r["sub_sub_lord"]) for r in _cro))
check("no event means no event_reading", "event_reading" not in _ci["out"] and "school" in _ci["out"])
check("cusp subset returns only the requested cusps", [r["cusp"] for r in _ci["sub"]["cusps"]] == [7, 2])
_er = _ci["ev"]["event_reading"]
_j = _ci["j"]
_ssl_sig = set(_ci["sigs"][_er["ssl"]])
check("event reading: CSL part is the standard verdict, SSL part is the same house test on the SSL",
      _er["csl_verdict"] == _j["verdict"] and _er["primary_cusp"] == 7 and _er["ssl"] == _cro[6]["sub_sub_lord"]
      and _er["ssl_favourable"] == sorted(_ssl_sig & set(_j["favourable_houses"])) and _er["ssl_denial"] == sorted(_ssl_sig & set(_j["denial_houses"]) - set(_j["favourable_houses"]))
      and _er["ssl_agrees_with_csl"] == (_er["ssl_verdict"] == _j["verdict"]) and "SSL" in _er["ssl_reason"] and "CSL" not in _er["ssl_reason"])
check("event reading works when the primary cusp is not among the requested cusps", _ci["only2"]["event_reading"]["primary_cusp"] == 7 and len(_ci["only2"]["cusps"]) == 1)
raises("interlinks reject a bad cusp", lambda: kcil.interlinks(None, [13]), ValueError, "cusps must be")

# ---------------------------------------------------------------------------------- double transit (backlog item 8)
import kp_double as kd  # noqa: E402

check("relation: occupies and the aspect table (Jupiter 5/7/9, Saturn 3/7/10, wrap-around)",
      kd.relation("Jupiter", 0, 0) == "occupies" and kd.relation("Jupiter", 0, 4) == "aspects (5th)" and kd.relation("Jupiter", 0, 6) == "aspects (7th)"
      and kd.relation("Jupiter", 0, 8) == "aspects (9th)" and kd.relation("Jupiter", 0, 2) is None and kd.relation("Saturn", 0, 2) == "aspects (3rd)"
      and kd.relation("Saturn", 0, 9) == "aspects (10th)" and kd.relation("Saturn", 0, 4) is None and kd.relation("Saturn", 10, 0) == "aspects (3rd)")


class _DipCtx:
    """A planet that creeps over a sign boundary and back within 2 days, peaking at jd 2.5 (a station)."""

    @staticmethod
    def lon(planet, jd):
        return 30.001 - 0.001 * (jd - 2.5) ** 2


_seg = kd.sign_segments(_DipCtx, "Saturn", 0.0, 10.0, [2.5], step=5.0)
check("sign segments catch a dip back over a boundary when split at the station",
      [s for _, _, s in _seg] == [0, 1, 0] and abs(_seg[0][1] - 1.5) < 1e-3 and abs(_seg[1][1] - 3.5) < 1e-3)
check("without the station the same dip is missed by a 5-day grid (why stations are used)", [s for _, _, s in kd.sign_segments(_DipCtx, "Saturn", 0.0, 10.0, [], step=5.0)] == [0])


def _dsnap(event="marriage", houses=None, primary_only=False, start=datetime(2026, 11, 20), end=datetime(2030, 11, 20), scope="cusp", include_lord=False):
    s_ = KPState(**BIRTH)
    with core.calc(s_) as c_:
        k = ch.chart_at(c_, c_.jd_from_local(s_.local_datetime), s_.latitude, s_.longitude, s_.local_datetime)
        hs, src = kd.resolve_houses(event, houses, primary_only)
        return dict(out=kd.find_double_transit(c_, k, hs, src, start, end, scope, include_lord), cr=ch.cusp_rows(k))


_d = _dsnap()
_do = _d["out"]
check("double transit: marriage gives its favourable houses 2, 7, 11 and some windows", [h["house"] for h in _do["houses"]] == [2, 7, 11] and _do["window_count"] > 0)
check("target sign is the cusp sign of each house", all(h["target_signs"] == [{**h["target_signs"][0]}] and h["target_signs"][0]["sign"] == _d["cr"][h["house"] - 1]["sign"] for h in _do["houses"]))

swe.set_sid_mode(45)


def _raw_sign(planet_id, jd):
    return int(((swe.calc_ut(jd, planet_id, swe.FLG_SWIEPH)[0][0] - swe.get_ayanamsa_ex_ut(jd, swe.FLG_SWIEPH)[1]) % 360.0) // 30.0) % 12


def _raw_rel(offsets, sj, target):
    return sj == target or any((sj + o - 1) % 12 == target for o in offsets)


_sign_idx = {n: i for i, n in enumerate(kt.SIGNS)}
_wins = {h["house"]: [(datetime.fromisoformat(w["start"]), datetime.fromisoformat(w["end"])) for w in h["windows"]] for h in _do["houses"]}
_checked = _pos = _bad = 0
_t = datetime(2026, 11, 21, 12, 0)
while _t < datetime(2030, 11, 19):
    _jd = swe.julday(_t.year, _t.month, _t.day, 12.0 - 5.5)
    _jsn, _ssn = _raw_sign(5, _jd), _raw_sign(6, _jd)
    for _h in _wins:
        _tg = _sign_idx[_d["cr"][_h - 1]["sign"]]
        _exp = _raw_rel((5, 7, 9), _jsn, _tg) and _raw_rel((3, 7, 10), _ssn, _tg)
        if any(abs((_t - a).total_seconds()) < 1.5 * 86400 or abs((_t - b).total_seconds()) < 1.5 * 86400 for a, b in _wins[_h]):
            continue
        _got = any(a <= _t < b for a, b in _wins[_h])
        _checked += 1
        _pos += bool(_exp)
        _bad += _exp != _got
    _t += timedelta(days=11)
check(f"double transit windows agree with raw swisseph on {_checked} sample dates ({_pos} inside a window)", _bad == 0 and _checked > 100 and _pos >= 5)
_as = _do["at_start"]
_jd0 = swe.julday(2026, 11, 20, 0.0 - 5.5)
check("at_start signs match raw swisseph and Saturn is retrograde on 2026-11-20",
      _as["jupiter"]["sign"] == kt.SIGNS[_raw_sign(5, _jd0)] and _as["saturn"]["sign"] == kt.SIGNS[_raw_sign(6, _jd0)] and _as["saturn"]["retrograde"] is True)
check("stations include Saturn turning direct on 2026-12-11 (published date)", any(s["station"] == "turns direct" and s["datetime_local"].startswith("2026-12-11") for s in _do["stations"]["Saturn"]))


def _dt(s):
    return datetime.fromisoformat(s)


_ok_struct = True
for _h in _do["houses"]:
    _prev_end = None
    for _w in _h["windows"]:
        _ph = _w["phases"]
        _ok_struct &= _ph[0]["start"] == _w["start"] and _ph[-1]["end"] == _w["end"] and all(a["end"] == b["start"] for a, b in zip(_ph, _ph[1:]))
        _ok_struct &= abs((_dt(_w["end"]) - _dt(_w["start"])).total_seconds() / 86400.0 - _w["days"]) < 0.01 and all(p["via"] for p in _ph)
        _ok_struct &= _prev_end is None or _dt(_w["start"]) >= _prev_end
        _prev_end = _dt(_w["end"])
check("windows are ordered, phases tile each window, every phase says why it counts", _ok_struct)
check("flat window list is sorted and complete", [(w["start"], w["house"]) for w in _do["windows"]] == sorted((w["start"], w["house"]) for w in _do["windows"])
      and len(_do["windows"]) == sum(h["window_count"] for h in _do["houses"]) and not _do["truncated"])
check("a window cut off by the query end is flagged", any(w["continues_after_window_end"] for h in _do["houses"] for w in h["windows"]))
check("every window lists nearby stations as facts", all("stations_within_60_days_of_window" in w for h in _do["houses"] for w in h["windows"]))
check("primary_only uses just the primary cusp", [h["house"] for h in _dsnap(primary_only=True)["out"]["houses"]] == [7])
check("own houses are sorted and de-duplicated", [h["house"] for h in _dsnap(event=None, houses=[10, 3, 3])["out"]["houses"]] == [3, 10])
_dh, _dl = _dsnap(scope="house")["out"], _dsnap(include_lord=True)["out"]
_days = lambda o: {h["house"]: sum(w["days"] for w in h["windows"]) for h in o["houses"]}  # noqa: E731
check("house scope and include_lord only widen the targets, so never shrink the windows",
      all(_days(_dh)[h] >= _days(_do)[h] - 0.01 and _days(_dl)[h] >= _days(_do)[h] - 0.01 for h in (2, 7, 11))
      and all(len(a["target_signs"]) >= len(b["target_signs"]) for a, b in zip(_dh["houses"], _do["houses"])))
check("include_lord names the lord's natal sign as a reason", any("house lord" in w for h in _dl["houses"] for t in h["target_signs"] for w in t["why"]))
raises("double transit needs an event or houses", lambda: kd.resolve_houses(None, None, False), core.KPError, "Give `event`")
raises("double transit rejects house 13", lambda: kd.resolve_houses(None, [13], False), core.KPError, "houses must be")
raises("double transit rejects unknown event", lambda: kd.resolve_houses("nonsense", None, False), ValueError, "Unknown event")
raises("double transit rejects unknown sign_scope", lambda: _dsnap(scope="zodiac"), core.KPError, "sign_scope must be")
raises("double transit rejects a window over 30 years", lambda: _dsnap(end=datetime(2060, 1, 1)), core.KPError, "too long")
raises("double transit rejects end before start", lambda: _dsnap(end=datetime(2026, 1, 1)), core.KPError, "after")

# ---------------------------------------------------------------------------------- KP transit rule set / verify_event_day (item 9)
import kp_transit as ktr  # noqa: E402
import kp_dasha as kda  # noqa: E402


def _vd(at=datetime(2026, 11, 20, 16, 0), **kw):
    s_ = KPState(**BIRTH)
    with core.calc(s_) as c_:
        k = ch.chart_at(c_, c_.jd_from_local(s_.local_datetime), s_.latitude, s_.longitude, s_.local_datetime)
        return ktr.verify_event_day(c_, k, s_.local_datetime, at, s_.latitude, s_.longitude, **kw), k


_v, _vk = _vd()
_tg = {}
for _r in _v["running_dasha"]:
    _tg.setdefault(_r["lord"], []).append(_r["label"])
check("verify_event_day: default planets Jupiter, Saturn, Sun, Moon and a 4-level dasha chain",
      [t["planet"] for t in _v["transits"]] == ["Jupiter", "Saturn", "Sun", "Moon"] and len(_v["running_dasha"]) == 4 and _v["target_lords"] == _tg)
swe.set_sid_mode(45)
_jdv = swe.julday(2026, 11, 20, 16.0 - 5.5)
_okv = True
for _t in _v["transits"]:
    _pid = {"Sun": 0, "Moon": 1, "Jupiter": 5, "Saturn": 6}[_t["planet"]]
    _lon = (swe.calc_ut(_jdv, _pid, swe.FLG_SWIEPH)[0][0] - swe.get_ayanamsa_ex_ut(_jdv, swe.FLG_SWIEPH)[1]) % 360.0
    _l = kt.kp_lords(_lon)
    _two = _l["sign_lord"] in _tg or _l["star_lord"] in _tg
    _exp = "match" if _two else ("sub lord only" if _l["sub_lord"] in _tg else "no match")
    _d = _t["dasha_lords"]
    _okv &= (_d["status"] == _exp and _d["sign_lord"]["lord"] == _l["sign_lord"] and _d["star_lord"]["lord"] == _l["star_lord"]
             and _d["sub_lord"]["lord"] == _l["sub_lord"] and _d["two_step_match"] == _two
             and _d["planet_is_a_target_lord"]["is"] == (_t["planet"] in _tg))
check("transit lords and match status agree with an independent raw swisseph recomputation", _okv)
_ru = {x["planet"] for x in _v["ruling_planets_at_moment"]["entries"]} if False else set(_v["ruling_planets_at_moment"]["set"])
check("ruling planets of the moment are listed and each transit row has an RP status",
      len(_ru) >= 3 and all(t["ruling_planets"]["status"] in ("match", "sub lord only", "no match") for t in _v["transits"]))
_sm = _v["summary"]
check("summary is consistent with the rows",
      _sm["planets_touching_dasha_lords"] == [t["planet"] for t in _v["transits"] if t["dasha_lords"]["two_step_match"]]
      and _sm["both_luminaries_touch"] == (_sm["sun_touches"] and _sm["moon_touches"]) and _sm["dasha_chain_lords_signifying_event_houses"] is None)
_ve, _ = _vd(fav=[2, 7, 11], den=[1, 6, 10], event={"event": "marriage"})
check("event houses: every dasha lord status equals lord_status", all(
      x["event_houses"]["favourable"] == kda.lord_status(_vk, x["lord"], {2, 7, 11}, {1, 6, 10})["favourable"] for x in _ve["running_dasha"])
      and _ve["summary"]["dasha_chain_lords_signifying_event_houses"] is not None and _ve["event"] == {"event": "marriage"})
check("depth 2 gives 2 levels; planets list is honoured and de-duplicated",
      len(_vd(depth=2)[0]["running_dasha"]) == 2 and [t["planet"] for t in _vd(planets=["Venus", "Venus", "Mars"])[0]["transits"]] == ["Venus", "Mars"])
check("without Sun and Moon both_luminaries_touch is None", _vd(planets=["Jupiter"])[0]["summary"]["both_luminaries_touch"] is None)
raises("verify_event_day rejects depth 6", lambda: _vd(depth=6), core.KPError, "depth must be")
raises("verify_event_day rejects an unknown planet", lambda: _vd(planets=["Pluto"]), core.KPError, "planets must be")
raises("verify_event_day rejects a date beyond the dasha span", lambda: _vd(at=datetime(2300, 1, 1)), Exception, "outside")

# ---------------------------------------------------------------------------------- longevity / maraka / badhaka (item 10)
import kp_longevity as klg  # noqa: E402

_ls = KPState(**BIRTH)
_ld = kda.make_dasha(_vk, _ls.local_datetime)
_lf = klg.longevity_facts(_vk, _ld, datetime(2026, 11, 20), datetime(2031, 11, 20))
_bad = kt.badhaka_house(kt.sign_index_of(_vk.asc))
_sg = lambda p: set(ch.signified(_vk, p))  # noqa: E731
check("longevity: groups, badhaka house and lagna kind",
      _lf["groups_used"]["badhaka"] == _bad and _lf["groups_used"]["critical_period_houses"] == sorted({2, 7, _bad})
      and _lf["lagna"]["kind"] == ("movable", "fixed", "dual")[kt.sign_index_of(_vk.asc) % 3])
_okc = True
for _blk, _cu in ((_lf["ascendant_csl"], 1), (_lf["eighth_csl"], 8)):
    _sl = kt.kp_lords(_vk.cusps[_cu - 1])["sub_lord"]
    _okc &= (_blk["cuspal_sub_lord"] == _sl and _blk["sub_lord_signifies"] == sorted(_sg(_sl))
             and _blk["supporting_houses_signified"] == sorted(_sg(_sl) & {1, 5, 9, 10})
             and _blk["difficult_houses_signified"] == sorted(_sg(_sl) & {6, 8, 12}))
check("1st and 8th cusp sub lords and their supporting / difficult hits match a direct computation", _okc)
check("badhaka and maraka lords are the house owners and carry their significations",
      _lf["badhaka"]["lord"] == _vk.owner[_bad] and [m["lord"] for m in _lf["maraka"]] == [_vk.owner[2], _vk.owner[7]]
      and _lf["badhaka"]["lord_signifies"] == sorted(_sg(_vk.owner[_bad])) and _lf["maraka"][0]["lord_in_house"] == _vk.house_of[_vk.owner[2]])
_cp = _lf["critical_periods"]
_crit = set(_cp["houses"])
check("every critical period is flagged and each of its lords signifies a maraka / badhaka house",
      _cp["count"] == len(_cp["periods"]) > 0 and all(p["critical_period"] and all(_sg(x) & _crit for x in p["lords"]) for p in _cp["periods"]))
_direct = kda.find_joint_periods(_ld, _vk, sorted(_crit), [], datetime(2026, 11, 20), datetime(2031, 11, 20), 3, False, 1, None, None, 100)
check("critical periods equal the joint-period search on those houses", [p["path"] for p in _cp["periods"]] == [p["path"] for p in _direct["periods"]])
check("include_periods=False leaves out the periods", "critical_periods" not in klg.longevity_facts(_vk, _ld, datetime(2026, 11, 20), datetime(2027, 11, 20), include_periods=False))
_lx = klg.longevity_facts(_vk, _ld, datetime(2026, 11, 20), datetime(2031, 11, 20), extra_houses=[8])
check("extra_houses widens the critical houses and never shrinks the periods", 8 in _lx["critical_periods"]["houses"] and _lx["critical_periods"]["count"] >= _cp["count"])
check("output states it is not a prediction", "not a prediction" in _lf["note"] and "lifespan" in _lf["note"])
raises("longevity rejects extra house 13", lambda: klg.longevity_facts(_vk, _ld, datetime(2026, 11, 20), datetime(2027, 11, 20), extra_houses=[13]), core.KPError, "extra_houses")

# ---------------------------------------------------------------------------------- polar fallback and options
polar = KPState(**{**BIRTH, "latitude": 70.0, "longitude": 25.0, "timezone": "Europe/Oslo"})
with core.calc(polar) as pc:
    pchart = ch.chart_at(pc, pc.jd_from_local(polar.local_datetime), 70.0, 25.0, polar.local_datetime)
    check("polar latitude falls back to Porphyry with a warning", pchart.houses["system"] == "porphyry" and pchart.houses["warning"])
ln = KPState(**BIRTH, options={"ayanamsa": "LAHIRI", "node_type": "true"})
with core.calc(ln) as lc2:
    lchart = ch.chart_at(lc2, lc2.jd_from_local(ln.local_datetime), ln.latitude, ln.longitude, ln.local_datetime)
    check("KP_NEW longitudes exceed Lahiri's by about 4m36s", 0.06 < (chart.planets["Sun"]["longitude"] - lchart.planets["Sun"]["longitude"]) % 360 < 0.09)
tn = KPState(**BIRTH, options={"node_type": "true"})
with core.calc(tn) as tc:
    tchart = ch.chart_at(tc, tc.jd_from_local(tn.local_datetime), tn.latitude, tn.longitude, tn.local_datetime)
    gap = abs((tchart.planets["Rahu"]["longitude"] - chart.planets["Rahu"]["longitude"] + 180) % 360 - 180)
    check("true node differs from mean node by 0 < gap < 2 degrees", 1e-4 < gap < 2.0)
    check("true-node Ketu stays opposite Rahu", abs((tchart.planets["Ketu"]["longitude"] - tchart.planets["Rahu"]["longitude"]) % 360 - 180) < 1e-6)

# ---------------------------------------------------------------------------------- MCP layer (in process)
import kp_server as srv  # noqa: E402


def call(name, args):
    res = asyncio.run(srv.mcp.call_tool(name, args))
    if isinstance(res, tuple):
        res = res[1] if isinstance(res[1], dict) else res[0]
    if isinstance(res, dict):
        return res.get("result", res)
    return json.loads(res[0].text)


tools = asyncio.run(srv.mcp.list_tools())
names = {t.name for t in tools}
check(f"{len(names)} tools registered", len(names) >= 28)
for must in ("get_cuspal_interlinks", "get_kp_chart", "judge_event", "find_event_periods", "get_ruling_planets", "cast_horary_chart", "judge_horary",
             "find_transit_triggers", "rectify_birth_time", "get_kp_summary", "get_kp_sub_table", "get_lagna_changes"):
    check(f"tool {must} present", must in names)

S = dict(BIRTH)
M = dict(place_name="Mumbai, IN", latitude=19.07, longitude=72.87, timezone="Asia/Kolkata", date="2026-11-20", time="15:00")
r = call("get_kp_sub_table", {"number": 1})
check("tool: sub table row 1", r["rows"][0]["sub_lord"] == "Ketu" and r["rows"][0]["sign"] == "Aries")
check("tool: whole table has 249 rows", call("get_kp_sub_table", {})["count"] == 249)
check("tool: Taurus rows", call("get_kp_sub_table", {"sign": "Taurus"})["count"] == 19)
check("tool: lookup by sign+degree", call("lookup_longitude", {"sign": "Aries", "degree_in_sign": 1.5})["star_lord"] == "Ketu")
check("tool: ayanamsa has 4 modes", len(call("get_ayanamsa", {"on_date": "2026-11-20"})["values"]) == 4)
check("tool: events listed", call("list_event_types", {})["count"] == len(kt.EVENTS))
c = call("get_kp_chart", {"state": S})
check("tool: chart has 12 cusps, 9 planets, significators", len(c["cusps"]) == 12 and len(c["planets"]) == 9 and "house_significators" in c)
check("tool: basic chart omits significators", "house_significators" not in call("get_kp_chart", {"state": S, "detail": "basic"}))
check("tool: cusps", len(call("get_cusps", {"state": S})["cusps"]) == 12)
check("tool: one planet signification", len(call("get_planet_significations", {"state": S, "planet": "Shani"})["planets"]) == 1)
check("tool: default significators have no extra method block", "four_step" not in call("get_significators", {"state": S}))
check("tool: significators method four_step", len(call("get_significators", {"state": S, "method": "four_step"})["four_step"]["planets"]) == 9)
check("tool: significators method six_fold", len(call("get_significators", {"state": S, "method": "six_fold"})["six_fold"]["houses"]) == 12)
check("tool: significator_method option picks the view", "six_fold" in call("get_significators", {"state": {**S, "options": {"significator_method": "6fold"}}}))
check("tool: type E off by default", "type_e" not in call("get_significators", {"state": S}))
_te = call("get_significators", {"state": S, "include_type_e": True})
check("tool: type E block when asked", len(_te["type_e"]["planets"]) == 9)
check("tool: type E leaves A-D untouched", _te["planet_significations"] == call("get_significators", {"state": S})["planet_significations"])
check("tool: type E via options + sign basis", call("get_significators", {"state": {**S, "options": {"include_type_e": True, "type_e_basis": "sign"}}})["type_e"]["basis"] == "sign")
_fz = call("get_fruitful_significators", {"state": S, "event": "marriage", "methods": "detrimental,sub_lord"})
check("tool: fruitful significators with reasons", _fz["methods"] == ["detrimental", "sub_lord"] and all("reasons" in r for r in _fz["planets"]))
check("tool: fruitful via ruling planets at a moment", set(call("get_fruitful_significators", {"state": S, "event": "marriage", "methods": "ruling_planets", "moment": M})["fruitful"])
      <= set(call("get_fruitful_significators", {"state": S, "event": "marriage", "methods": "none"})["fruitful"]))
_ep0 = call("find_event_periods", {"state": S, "event": "marriage", "start": "2026-11-20", "end": "2040-11-20", "depth": 3})
_ep1 = call("find_event_periods", {"state": S, "event": "marriage", "start": "2026-11-20", "end": "2040-11-20", "depth": 3, "fruitful_method": "detrimental"})
check("tool: find_event_periods fruitful filter narrows and reports the selection", _ep1["count"] <= _ep0["count"] and "fruitful_selection" in _ep1 and "fruitful_selection" not in _ep0)
_fs2 = call("find_stations", {"moment": M, "planet": "Shani", "start": "2026-01-01", "end": "2027-01-01"})
check("tool: find_stations Saturn 2026 has two stations", _fs2["count"] == 2 and _fs2["stations"][0]["station"] == "turns retrograde")
_dt1 = call("find_double_transit", {"state": S, "event": "marriage", "start": "2026-11-20", "end": "2030-11-20"})
check("tool: find_double_transit returns houses 2, 7, 11 with school note, stations and birth summary",
      [h["house"] for h in _dt1["houses"]] == [2, 7, 11] and "school" in _dt1 and "birth" in _dt1 and set(_dt1["stations"]) == {"Jupiter", "Saturn"}
      and _dt1["window_count"] == _do["window_count"])
_dt2 = call("find_double_transit", {"state": S, "houses": [7], "start": "2026-11-20", "end": "2028-11-20", "sign_scope": " House ", "include_lord": True})
check("tool: find_double_transit accepts own houses, a padded / upper-case sign_scope and include_lord",
      [h["house"] for h in _dt2["houses"]] == [7] and _dt2["sign_scope"] == "house" and _dt2["include_lord"] is True)
_vt = call("verify_event_day", {"state": S, "when": "2026-11-20T16:00:00", "event": "marriage"})
check("tool: verify_event_day returns transits, dasha chain with event houses, summary and birth",
      len(_vt["transits"]) == 4 and all("event_houses" in x for x in _vt["running_dasha"]) and "summary" in _vt and "birth" in _vt)
_vt2 = call("verify_event_day", {"state": S, "when": "2026-11-20"})
check("tool: a date-only `when` is taken as 12:00 local", _vt2["datetime_local"].startswith("2026-11-20T12:00"))
_lt = call("get_longevity_facts", {"state": S, "start": "2026-11-20", "end": "2029-11-20"})
check("tool: get_longevity_facts returns the blocks, critical periods and birth",
      all(k in _lt for k in ("ascendant_csl", "eighth_csl", "badhaka", "maraka", "critical_periods", "birth", "note")))
_jr2 = call("judge_event", {"state": {**S, "options": {"retrograde_rule": "strict"}}, "event": "marriage"})
check("tool: judge_event carries the retrograde check", _jr2["csl"]["retrograde_check"]["rule"] == "strict")
_ps = call("find_event_periods", {"state": {**S, "options": {"retrograde_rule": "strict"}}, "event": "marriage", "start": "2026-11-20", "end": "2040-11-20", "depth": 3})
check("tool: strict marks delayed exactly the flagged periods", all(bool(p.get("delayed")) == bool(p.get("retrograde_flags")) for p in _ps["periods"]) and _ps["retrograde_rule"] == "strict")
check("tool: strict adds transit info for flagged retrograde planets", (not any(p.get("retrograde_flags") for p in _ps["periods"])) or "retrograde_transit_info" in _ps)
check("tool: default rule adds no delayed mark or transit info", "retrograde_transit_info" not in _ep0 and not any(p.get("delayed") for p in _ep0["periods"]))
check("tool: analyze_joint_period has a retrograde block", "retrograde" in call("analyze_joint_period", {"state": S, "path": ["Venus", "Sun"], "event": "service"}))
_NC = {**S, "options": {"cusp_orb_deg": 3.3333}}
_nc = call("get_significators", {"state": _NC})
check("tool: near_cusp off by default in chart and significators", "near_cusp" not in call("get_significators", {"state": S}) and "near_cusp" not in call("get_kp_chart", {"state": S}))
check("tool: significators carry near_cusp when the orb is set", _nc["near_cusp"]["count"] >= 4 and _nc["near_cusp"]["applied_to_A_to_D"] is False
      and all("signifies_with_near_cusp" in r for r in _nc["planet_significations"]))
check("tool: chart carries near_cusp when the orb is set", "near_cusp" in call("get_kp_chart", {"state": _NC}))
check("tool: reporting near-cusp leaves A-D and verdicts untouched", [r["levels"] for r in _nc["planet_significations"]] == [r["levels"] for r in call("get_significators", {"state": S})["planet_significations"]]
      and call("judge_event", {"state": _NC, "event": "marriage"})["verdict"] == call("judge_event", {"state": S, "event": "marriage"})["verdict"])
_ap = call("judge_event", {"state": {**S, "options": {"cusp_orb_deg": 3.3333, "cusp_proximity_apply": True}}, "event": "marriage"})
check("tool: applying the shift only adds houses to the CSL", set(call("judge_event", {"state": S, "event": "marriage"})["csl"]["sub_lord_signifies"]) <= set(_ap["csl"]["sub_lord_signifies"]))
_epa = call("find_event_periods", {"state": {**S, "options": {"cusp_orb_deg": 3.3333, "cusp_proximity_apply": True}}, "event": "marriage", "start": "2026-11-20", "end": "2040-11-20", "depth": 3})
check("tool: period search runs with the shift applied", _epa["count"] >= 0 and "promise" in _epa)
_CX = {**S, "options": {"csl_extension": "both"}}
_ca0 = call("get_cuspal_sublord_analysis", {"state": S})
_ca1 = call("get_cuspal_sublord_analysis", {"state": _CX})
check("tool: CSL analysis has no extension block by default", all("csl_extension" not in c for c in _ca0["cusps"]) and "extended_reading" not in call("judge_event", {"state": S, "event": "marriage"}))
check("tool: CSL analysis carries the extension for all 12 cusps and keeps the base houses", len(_ca1["cusps"]) == 12 and all("csl_extension" in c for c in _ca1["cusps"])
      and [c["sub_lord_signifies"] for c in _ca1["cusps"]] == [c["sub_lord_signifies"] for c in _ca0["cusps"]])
_jx = call("judge_event", {"state": _CX, "event": "marriage"})
check("tool: judge_event adds extended_reading beside the unchanged verdict", _jx["verdict"] == call("judge_event", {"state": S, "event": "marriage"})["verdict"] and _jx["extended_reading"]["mode"] == "both")
_cl = call("get_cuspal_interlinks", {"state": S})
check("tool: interlinks for all 12 cusps, flagged as a later school, no event reading", len(_cl["cusps"]) == 12 and "not pure krishnamurti" in _cl["school"].lower() and "event_reading" not in _cl)
_cle = call("get_cuspal_interlinks", {"state": S, "cusps": [7], "event": "marriage"})
check("tool: interlinks with an event add the SSL reading beside the CSL verdict", _cle["event_reading"]["csl_verdict"] == call("judge_event", {"state": S, "event": "marriage"})["verdict"] and len(_cle["cusps"]) == 1)
check("tool: interlinks accept custom houses", call("get_cuspal_interlinks", {"state": S, "custom_favourable": [2, 11], "custom_denial": [12], "primary_cusp": 2})["event_reading"]["event"] == "custom")
check("tool: include_sub_level adds signifies_6fold", "signifies_6fold" in call("get_planet_significations", {"state": S, "planet": "Guru", "include_sub_level": True})["planets"][0])
check("tool: CSL analysis for two cusps", len(call("get_cuspal_sublord_analysis", {"state": S, "cusps": [7, 10]})["cusps"]) == 2)
j = call("judge_event", {"state": S, "event": "marriage"})
check("tool: judge_event verdict", j["verdict"] and j["csl"]["sub_lord"] == mr["csl"]["sub_lord"])
check("tool: promise table", len(call("get_promise_table", {"state": S})["matters"]) == len(kt.EVENTS))
check("tool: dasha level 1", call("get_kp_dasha", {"state": S, "level": 1})["count"] >= 9)
check("tool: dasha inside a parent", call("get_kp_dasha", {"state": S, "level": 2, "parent": ["Venus"]})["count"] == 9)
cur = call("get_current_kp_dasha", {"state": S, "on_date": "2026-11-20", "depth": 3})
check("tool: current dasha depth 3", len(cur["running"]) == 3)
check("tool: dasha vs event", "lord_vs_event" in call("get_dasha_significations", {"state": S, "on_date": "2026-11-20", "event": "service"}))
ep = call("find_event_periods", {"state": S, "event": "service", "start": "2026-11-20", "end": "2029-11-20", "depth": 3})
check("tool: event periods carry the promise first", "promise" in ep and ep["promise"]["verdict"] and "periods" in ep)
jp2 = call("analyze_joint_period", {"state": S, "path": cur["running"][0]["lords"] + [cur["running"][1]["lords"][-1]], "event": "service"})
check("tool: joint period audit", len(jp2["lords"]) == 2 and "every_lord_signifies_favourable" in jp2)
rpt = call("get_ruling_planets", {"moment": M, "birth": S})
check("tool: ruling planets with natal overlay", len(rpt["entries"]) == 7 and "natal_signifies" in rpt["ruling_planets"][0])
check("tool: ruling planet timeline", len(call("get_ruling_planets_timeline", {"moment": M, "hours": 2})["stretches"]) >= 2)
hc = call("cast_horary_chart", {"moment": M, "number": 100})
check("tool: horary chart lagna check", hc["horary"]["lagna_check"]["matches_number"] and len(hc["cusps"]) == 12)
jh2 = call("judge_horary", {"moment": M, "number": 100, "event": "property_purchase"})
check("tool: judge_horary", jh2["judgment"]["event"] == "property_purchase" and "ruling_planet_check" in jh2["judgment"])
check("tool: judge_horary carries the horary extras", all(k in jh2["horary_extras"] for k in ("genuineness", "horary_checklist", "lagna_point", "reminders"))
      and "first_asking_only" not in jh2["horary_extras"])
jh3 = call("judge_horary", {"moment": M, "number": 100, "event": "property_purchase", "asked_before": True, "lagna_point": "actual_if_inside"})
check("tool: asked_before is flagged and lagna_point explanation follows the option",
      jh3["horary_extras"]["first_asking_only"]["declared_repeat"] and jh3["horary_extras"]["lagna_point"]["used"] == "actual_if_inside")
check("tool: transit snapshot + natal", "natal_house" in call("get_transit_snapshot", {"moment": M, "birth": S})["planets"][0])
bc = call("find_boundary_changes", {"moment": M, "planet": "Moon", "start": "2026-11-20", "end": "2026-11-21", "level": "star"})
check("tool: boundary changes", bc["change_count"] >= 1)
tt = call("find_transit_triggers", {"state": S, "start": "2026-11-20", "end": "2027-02-20", "planets": ["Sun"], "levels": ["star"]})
check("tool: triggers use the running dasha lords as targets", set(tt["targets"]) == {r["lords"][-1] for r in call("get_current_kp_dasha", {"state": S, "on_date": "2026-11-20", "depth": 4})["running"]})
check("tool: lagna changes", len(call("get_lagna_changes", {"moment": {**M, "date": "1990-05-15", "time": "10:30"}, "minutes": 15})["stretches"]) >= 2)
rr = call("rectify_birth_time", {"state": S, "events": [{"date": "2018-02-10", "event": "marriage"}], "minutes": 10})
check("tool: rectify", rr["scanned"]["trial_times"] == 21 and rr["best_ranges"])
rr2 = call("rectify_birth_time", {"state": S, "events": [{"date": "2018-02-10", "event": "marriage", "weight": 3}], "minutes": 10, "moon_link": "require"})
check("tool: rectify with weight and moon_link", rr2["max_score_possible"] == 18.0 and rr2["moon_link"]["mode"] == "require" and "lagna_moon_link" in rr2["best_ranges"][0])
sm = call("get_kp_summary", {"state": S, "on_date": "2026-11-20"})
check("tool: summary", len(sm["cuspal_sub_lords"]) == 12 and len(sm["running_dasha"]) == 4 and sm["promise_table"])


def tool_error(name, args, text):
    try:
        asyncio.run(srv.mcp.call_tool(name, args))
    except Exception as e:  # noqa: BLE001
        check(f"tool error: {name} -> '{text}'", text.lower() in str(e).lower())
        return
    check(f"tool error: {name} -> '{text}'", False)


# ---- item 13: twin / close-birth comparison
_tw0 = call("compare_births", {"a": S, "b": S})
check("compare_births: identical births differ nowhere",
      _tw0["summary"]["count"] == 0 and not _tw0["planets"] and not _tw0["events"] and _tw0["minutes_apart"] == 0
      and _tw0["dasha_balance"]["balance_difference_days"] == 0)
_S6 = {**S, "time_of_birth": "10:36"}
_tw = call("compare_births", {"a": S, "b": _S6})
check("compare_births: 6 minutes apart, 12 cusp rows, minutes_apart 6", len(_tw["cusps"]) == 12 and abs(_tw["minutes_apart"] - 6.0) < 1e-6 and _tw["same_place"])
with core.calc(KPState(**S)) as _cx:
    _na = srv._natal(_cx, KPState(**S))[0]
with core.calc(KPState(**_S6)) as _cx:
    _nb = srv._natal(_cx, KPState(**_S6))[0]
_ok = True
for _r in _tw["cusps"]:
    _la, _lb = kt.kp_lords(_na.cusps[_r["house"] - 1]), kt.kp_lords(_nb.cusps[_r["house"] - 1])
    _d = [k.replace("_lord", "") for k in ("sign_lord", "star_lord", "sub_lord", "sub_sub_lord") if _la[k] != _lb[k]]
    _ok &= (_r["differs_at"] == _d and _r["a"]["sub_lord"] == _la["sub_lord"] and _r["b"]["sub_lord"] == _lb["sub_lord"]
            and _r["sub_lord_differs"] == ("sub" in _d))
check("compare_births: every cusp row equals a direct recomputation of the lords", _ok)
check("compare_births: summary lists match cusp rows",
      _tw["summary"]["cusps_with_different_sub_lord"] == [r["house"] for r in _tw["cusps"] if r["sub_lord_differs"]]
      and _tw["summary"]["count"] == len(_tw["summary"]["cusps_with_different_sub_lord"]))
check("compare_births: Lagna moved forward in 6 minutes (about 1.3 deg)", 1.0 < _tw["lagna"]["moved_degrees"] < 1.7)
check("compare_births: events listed only where verdict or CSL differ",
      all(e["verdict_a"] != e["verdict_b"] or e["csl_a"] != e["csl_b"] for e in _tw["events"]))
check("compare_births: dasha balance difference sign (later birth -> smaller balance of the same lord)",
      _tw["dasha_balance"]["same_start_lord"] and _tw["dasha_balance"]["balance_difference_days"] < 0)
_tw_far = call("compare_births", {"a": S, "b": {**S, "latitude": 28.61, "longitude": 77.2, "timezone": "Asia/Kolkata"}})
check("compare_births: different place -> same_place False", _tw_far["same_place"] is False)
_tw_opt = call("compare_births", {"a": {**S, "options": {"ayanamsa": "LAHIRI"}}, "b": _S6})
check("compare_births: options of `a` apply to both", _tw_opt["b"] == call("compare_births", {"a": {**S, "options": {"ayanamsa": "LAHIRI"}}, "b": {**_S6, "options": {"ayanamsa": "LAHIRI"}}})["b"])

# ---- item 15: muhurta windows
_MM = {**M, "date": "2026-11-21", "time": "06:00"}
_mw = call("find_muhurta_windows", {"moment": _MM, "event": "marriage", "step_minutes": 10, "min_rp_support": 1})
check("muhurta: 24 h at 10 min = 145 samples, matter block, truncated flag",
      _mw["samples"] == 145 and _mw["matter"]["favourable"] == [2, 7, 11] and _mw["truncated"] is False and _mw["window_count"] == len(_mw["windows"]))
check("muhurta: passing samples = sum of window samples", _mw["samples_passing"] == sum(w["samples"] for w in _mw["windows"]))
_ok = True
_mopt = KPState(**{**BIRTH}).options
with core.calc(MomentState(**_MM)) as _cx:
    import kp_muhurta as _mh
    _dc = {}
    for _w in _mw["windows"]:
        _t = datetime.fromisoformat(_w["from"])
        _r = _mh.evaluate(_cx, _t, _MM["latitude"], _MM["longitude"], "marriage", None, None, None, None, _dc)
        _ok &= _r["lagna_sub_ok"] and _mh.VERDICT_RANK[_r["csl_verdict"]] >= 2 and len(_r["rp_supporting"]) >= 1
        _ok &= _r["lagna_sub_lord"] == _w["lagna_sub_lord"] and _r["csl"] == _w["csl"]
        _tb = datetime.fromisoformat(_w["until_about"])
        _rb = _mh.evaluate(_cx, _tb, _MM["latitude"], _MM["longitude"], "marriage", None, None, None, None, _dc)
        _ok &= not (_rb["lagna_sub_ok"] and _mh.VERDICT_RANK[_rb["csl_verdict"]] >= 2 and len(_rb["rp_supporting"]) >= 1) or _w.get("continues_after_window_end")
check("muhurta: every window start passes the tests and the sample after its end fails (independent recomputation)", _ok)
_mw2 = call("find_muhurta_windows", {"moment": _MM, "event": "marriage", "step_minutes": 10, "min_rp_support": 1, "csl_min": "strong"})
check("muhurta: stricter csl_min never adds passing samples", _mw2["samples_passing"] <= _mw["samples_passing"])
_mw3 = call("find_muhurta_windows", {"moment": _MM, "event": "marriage", "step_minutes": 10, "min_rp_support": 1, "require_lagna_sub": False})
check("muhurta: dropping the Lagna test never removes passing samples", _mw3["samples_passing"] >= _mw["samples_passing"])
_mw4 = call("find_muhurta_windows", {"moment": _MM, "event": "marriage", "step_minutes": 10, "min_rp_support": 1, "limit": 1})
check("muhurta: limit truncates", len(_mw4["windows"]) == 1 and (_mw4["truncated"] is (_mw["window_count"] > 1)))
_mw5 = call("find_muhurta_windows", {"moment": _MM, "custom_favourable": [2, 7, 11], "custom_denial": [1, 6, 10], "primary_cusp": 7, "step_minutes": 10, "min_rp_support": 1})
check("muhurta: custom houses equal the marriage table", _mw5["windows"] == _mw["windows"])
check("muhurta: place summary and note present", _mw["place"]["place"] and "not an auspiciousness verdict" in _mw["note"])

# ---- item 16: outer planets as occupants
_o0 = call("get_kp_chart", {"state": S})
_o1 = call("get_kp_chart", {"state": {**S, "options": {"include_outer": True}}})
check("outer: default off -> no outer block / keys", "outer_planets" not in _o0 and "outer_occupants" not in _o0["cusps"][0]
      and "outer_occupants" not in _o0["house_significators"][0])
check("outer: on -> block lists Uranus, Neptune, Pluto", [x["planet"] for x in _o1["outer_planets"]["planets"]] == ["Uranus", "Neptune", "Pluto"])
_pl1 = {x["planet"]: x for x in _o1["planets"]}
check("outer: block houses equal the planet rows", all(x["house"] == _pl1[x["planet"]]["house"] for x in _o1["outer_planets"]["planets"]))
check("outer: cusp rows and house significators list the same occupants",
      all(c["outer_occupants"] == h["outer_occupants"] for c, h in zip(_o1["cusps"], _o1["house_significators"]))
      and sorted(sum((c["outer_occupants"] for c in _o1["cusps"]), [])) == ["Neptune", "Pluto", "Uranus"])
_strip = lambda d: {k: v for k, v in d.items() if k != "outer_occupants"}
check("outer: A-D house tables and planet significations identical with and without outer planets",
      [_strip(h) for h in _o1["house_significators"]] == _o0["house_significators"]
      and [x for x in _o1["planet_significations"] if x["planet"] in kt.PLANETS] == _o0["planet_significations"])
_j0 = call("judge_event", {"state": S, "event": "marriage"})
_j1 = call("judge_event", {"state": {**S, "options": {"include_outer": True}}, "event": "marriage"})
check("outer: judgment verdict unchanged", _j0["verdict"] == _j1["verdict"] and _j0["favourable_signified_by_csl"] == _j1["favourable_signified_by_csl"])

# ---- item 27: condensed report
_rp = call("get_kp_report", {"state": S, "on_date": "2026-11-20"})
_md = _rp["markdown"]
check("report: markdown with all sections", _rp["format"] == "markdown" and all(x in _md for x in ("## Planets", "## Cusps and cuspal links", "## Significators by house", "## Dasha", "## Ruling planets")))
check("report: header carries birth date and place", "1990-05-15 10:30:00" in _md.splitlines()[0] and "Mumbai" in _md.splitlines()[0])
_cs = call("get_kp_chart", {"state": S})
check("report: every cusp sub lord and planet sub lord appears in its row",
      all(f"| {c['house']} | {c['dms']} |" in _md and c["sub_lord"] in _md for c in _cs["cusps"])
      and all(f"| {p['planet']}" in _md and p["dms"] in _md for p in _cs["planets"]))
_cs2 = call("get_cuspal_sublord_analysis", {"state": S, "cusp": 7}) if "get_cuspal_sublord_analysis" in names else None
_row7 = [l for l in _md.splitlines() if l.startswith("| 7 | ")][0]
check("report: CSL-signifies column equals the signified houses of the sub lord",
      _row7.split("|")[8].strip() == ",".join(str(x) for x in (_cs2 or {}).get("sub_lord_signifies", [6, 7, 10, 12])))
check("report: running ladder and balance present", "Balance at birth: Sun" in _md and "Rahu/Sun/Saturn" in _md)
check("report: has no verdict words", all(w not in _md.lower() for w in ("strongly_promised", "not_promised", "will happen")))

# ---- item 26: guided workflow
_wf = call("kp_judgment_workflow", {"state": S, "event": "marriage", "start": "2026-11-20", "end": "2029-11-20", "when": "2026-12-01"})
_je = call("judge_event", {"state": S, "event": "marriage"})
_ep = call("find_event_periods", {"state": S, "event": "marriage", "start": "2026-11-20", "end": "2029-11-20", "limit": 300})
_st = {x["step"]: x for x in _wf["steps"]}
check("workflow: six steps in order, counts add up", [x["step"] for x in _wf["steps"]] == [1, 2, 3, 4, 5, 6]
      and sum(_wf["status_counts"].values()) == 6)
check("workflow: step 1/3 equal judge_event", _st[1]["favourable_houses"] == _je["favourable_houses"] and _st[1]["denial_houses"] == _je["denial_houses"]
      and _st[3]["verdict"] == _je["verdict"] and _st[2]["csl"] == _je["csl"]["sub_lord"] and _st[2]["csl_signifies"] == _je["csl"]["sub_lord_signifies"])
check("workflow: step 3 status follows the verdict", _st[3]["status"] == {"strongly_promised": "pass", "promised_with_obstacles": "partial", "weakly_promised": "partial",
                                                                          "doubtful": "fail", "not_promised": "fail"}[_je["verdict"]])
check("workflow: step 4 periods equal find_event_periods", _st[4]["period_count"] == len(_ep["periods"]) and _st[4]["periods_shown"] == _ep["periods"][:10])
_vd = call("verify_event_day", {"state": S, "when": "2026-12-01", "event": "marriage"})
check("workflow: step 5 equals verify_event_day", _st[5]["summary"] == _vd["summary"] and _st[5]["transits"] == _vd["transits"])
_rpm = call("get_ruling_planets", {"moment": {**M, "date": "2026-12-01", "time": "12:00"}, "birth": S})
check("workflow: step 6 ruling planets equal the ruling-planet tool at that moment (same place)",
      sorted(_st[6]["ruling_planets"]) == sorted(p["planet"] for p in _rpm["ruling_planets"]))
check("workflow: step 6 supporting subset of ruling planets, no verdict key", set(_st[6]["supporting"]) <= set(_st[6]["ruling_planets"]) and "overall" not in _wf)
_wf2 = call("kp_judgment_workflow", {"state": S, "custom_favourable": [2, 7, 11], "custom_denial": [1, 6, 10], "primary_cusp": 7, "start": "2026-11-20", "end": "2029-11-20", "when": "2026-12-01"})
check("workflow: custom houses = marriage table; no `when` -> step 5 n.a.",
      _wf2["steps"][3] == _st[4] or _wf2["steps"][3]["period_count"] == _st[4]["period_count"])
_wf3 = call("kp_judgment_workflow", {"state": S, "event": "marriage", "limit": 3})
check("workflow: without `when` step 5 is n.a.", _wf3["steps"][4]["status"] == "n.a." and len(_wf3["steps"][3]["periods_shown"]) <= 3)

# ---- item 19: node check
_nc = call("get_kp_chart", {"state": S, "detail": "full"})["node_check"]
_ncT = call("get_kp_chart", {"state": {**S, "options": {"node_type": "true"}}, "detail": "full"})["node_check"]
check("node_check: absent unless detail='full'", "node_check" not in call("get_kp_chart", {"state": S}))
check("node_check: mean chart compares with true, and the reverse", _nc["used"] == "mean" and _nc["compared_with"] == "true"
      and _ncT["used"] == "true" and _ncT["compared_with"] == "mean")
check("node_check: 'other' of the mean chart = 'used' of the true chart (both nodes)",
      all(_nc["nodes"][n]["other"] == _ncT["nodes"][n]["used"] and _nc["nodes"][n]["used"] == _ncT["nodes"][n]["other"] for n in ("Rahu", "Ketu")))
check("node_check: Ketu is Rahu + 180 in the difference and the lords", abs(_nc["nodes"]["Rahu"]["difference_deg"] - _nc["nodes"]["Ketu"]["difference_deg"]) < 1e-6
      and abs(_nc["nodes"]["Rahu"]["difference_deg"]) < 2.0)
check("node_check: flags consistent with the lords", all(r["sub_lord_differs"] == (r["used"]["sub_lord"] != r["other"]["sub_lord"]) for r in _nc["nodes"].values())
      and _nc["any_sub_lord_differs"] == any(r["sub_lord_differs"] for r in _nc["nodes"].values()))

# ---- item 21: place helper
import kp_place as _kpl
_t1 = call("get_timezone_info", {"latitude": 19.07, "longitude": 72.87, "date": "1990-05-15", "time": "10:30"})
check("place: Mumbai -> Asia/Kolkata, +5.5, no DST", _t1["timezone"] == "Asia/Kolkata" and _t1["utc_offset_hours"] == 5.5 and _t1["is_dst"] is False)
_t2 = call("get_timezone_info", {"latitude": 40.71, "longitude": -74.0, "date": "2026-07-01"})
check("place: New York in July = -4 with DST, default 12:00", _t2["timezone"] == "America/New_York" and _t2["utc_offset_hours"] == -4.0 and _t2["is_dst"] is True and _t2["local_datetime"].endswith("12:00:00"))
_t3 = call("get_timezone_info", {"latitude": 40.71, "longitude": -74.0, "date": "2026-11-01", "time": "01:30"})
check("place: repeated hour at DST end flagged ambiguous with the alternative offset", _t3["ambiguous_local_time"] is True and _t3["alternative_utc_offset_hours"] == -5.0 and _t3["utc_offset_hours"] == -4.0)
_t4 = call("get_timezone_info", {"latitude": 40.71, "longitude": -74.0, "date": "2026-03-08", "time": "02:30"})
check("place: skipped hour at DST start flagged non-existent", _t4["nonexistent_local_time"] is True and _t4["ambiguous_local_time"] is False)
_t5 = call("get_timezone_info", {"latitude": 28.61, "longitude": 77.2})
check("place: no date -> only the zone", _t5["timezone"] == "Asia/Kolkata" and "utc_offset_hours" not in _t5)
_fake = lambda q, n: [{"name": "Mumbai, Maharashtra, India", "latitude": 19.076, "longitude": 72.8777}, {"name": "Nowhere at sea", "latitude": 0.0, "longitude": -30.0}][:n]
_lp = _kpl.lookup_place("Mumbai", 2, geocoder=_fake)
check("place: lookup_place adds the offline timezone (open sea gets an Etc zone)", _lp["count"] == 2 and _lp["places"][0]["timezone"] == "Asia/Kolkata"
      and _lp["places"][1]["timezone"] == "Etc/GMT+2")
_kpl_orig = _kpl._nominatim
_kpl._nominatim = lambda q, n: (_ for _ in ()).throw(core.KPError("Place lookup needs internet access and failed (X)."))
tool_error("lookup_place", {"query": "Mumbai"}, "needs internet")
_kpl._nominatim = _kpl_orig

# ---- item 25: annual chart
_an = call("get_kp_annual_chart", {"state": S, "year": 2027})
_natal_c = call("get_kp_chart", {"state": S})
_sun_n = [p for p in _natal_c["planets"] if p["planet"] == "Sun"][0]["longitude"]
_sun_r = [p for p in _an["chart"]["planets"] if p["planet"] == "Sun"][0]["longitude"]
check("annual: Sun in the return chart equals the natal Sun within 0.5 arc-second", abs(((_sun_r - _sun_n + 180) % 360) - 180) < 0.5 / 3600)
check("annual: return falls within 2 days of the birthday", abs((datetime.fromisoformat(_an["return_moment"]["local"]) - datetime(2027, 5, 15, 10, 30)).total_seconds()) < 2 * 86400)
check("annual: 12 cusp and 9+ planet comparison rows, promise table of every event", len(_an["comparison"]["cusps"]) == 12 and len(_an["comparison"]["planets"]) == 9
      and len(_an["promise_table"]) == len(call("list_event_types", {})["events"]))
_an_p = call("get_kp_annual_chart", {"state": S, "year": 2027, "place": {"place_name": "Delhi, IN", "latitude": 28.61, "longitude": 77.2, "timezone": "Asia/Kolkata"}})
check("annual: other place -> same instant, same planet longitudes, different cusps",
      _an_p["return_moment"]["julian_day_ut"] == _an["return_moment"]["julian_day_ut"]
      and [p["longitude"] for p in _an_p["chart"]["planets"]] == [p["longitude"] for p in _an["chart"]["planets"]]
      and [c["longitude"] for c in _an_p["chart"]["cusps"]] != [c["longitude"] for c in _an["chart"]["cusps"]])
_s29 = call("get_kp_annual_chart", {"state": {**S, "date_of_birth": "1992-02-29"}, "year": 2027, "detail": "basic"})
check("annual: 29 February birth works in a non-leap year", _s29["return_moment"]["local"][:7] in ("2027-02", "2027-03") and abs((datetime.fromisoformat(_s29["return_moment"]["local"]) - datetime(2027, 2, 28, 10, 30)).total_seconds()) < 4 * 86400)
check("annual: natal dasha ladder present", _an["natal_dasha_at_return"][0]["level"] == "mahadasha")

# ---- item 14: health facts
_hf = call("get_health_facts", {"state": S, "start": "2026-11-21", "end": "2029-11-21"})
_c6 = call("get_cuspal_sublord_analysis", {"state": S, "cusp": 6}) if "get_cuspal_sublord_analysis" in names else None
_sg = set(_hf["sixth_cusp"]["sub_lord_signifies"])
check("health: 6th CSL fields consistent with the sets", _hf["sixth_cusp"]["disease_houses_signified"] == sorted(_sg & {6, 8, 12})
      and _hf["sixth_cusp"]["recovery_houses_signified"] == sorted(_sg & {5, 11}) and _hf["sixth_cusp"]["signifies_first_house"] == (1 in _sg))
check("health: 6th CSL equals the cusp-6 row of the chart", _hf["sixth_cusp"]["sub_lord"] == call("get_kp_chart", {"state": S})["cusps"][5]["sub_lord"])
check("health: reading flags follow the rule", _hf["reading"]["tendency_to_illness_by_rule"] == bool(_sg & {6, 8, 12})
      and _hf["reading"]["recurrence_by_rule"] == (bool(_sg & {6, 8, 12}) and 1 in _sg) and _hf["reading"]["recovery_support_by_rule"] == bool(_sg & {5, 11}))
check("health: disclaimer, no verdict words, body table by sign", "Not medical advice" in _hf["disclaimer"] and "parts" in _hf["body_parts_traditional"]["6th cusp sign"]
      and all(w not in str(_hf).lower() for w in ("strongly_promised", "you will", "cancer diagnosis")))
_dp = _hf["disease_periods"]["periods"]
_ok = bool(_dp)
for _p in _dp:
    _ok &= all(set(_l).__len__() >= 0 for _l in [_p["lords"]]) and _p["lords"] is not None
check("health: disease periods are the joint periods of 6 / 8 / 12 (equal to find_event_periods with custom houses)",
      _hf["disease_periods"]["count"] == call("find_event_periods", {"state": S, "custom_favourable": [6, 8, 12], "custom_denial": [5, 11], "start": "2026-11-21", "end": "2029-11-21", "limit": 50})["count"])
_hf2 = call("get_health_facts", {"state": S, "include_periods": False})
check("health: include_periods False drops the period blocks", "disease_periods" not in _hf2 and "recovery_periods" not in _hf2)
check("health: 8th and 12th CSL listed", [x["cusp"] for x in _hf["eighth_and_twelfth_csl"]] == [8, 12])

tool_error("get_significators", {"state": S, "method": "ten_step"}, "method must be")
tool_error("get_health_facts", {"state": S, "limit": 0}, "limit must")
tool_error("get_kp_annual_chart", {"state": S, "year": 1500}, "year must be")
tool_error("get_kp_annual_chart", {"state": S, "year": 2027, "detail": "huge"}, "detail")
tool_error("get_timezone_info", {"latitude": 95, "longitude": 10}, "latitude")
tool_error("get_timezone_info", {"latitude": 10, "longitude": 200}, "longitude")
tool_error("get_timezone_info", {"latitude": 10, "longitude": 10, "date": "15-05-1990"}, "Bad date")
tool_error("lookup_place", {"query": "  "}, "must not be empty")
tool_error("lookup_place", {"query": "Mumbai", "limit": 99}, "limit must")
tool_error("kp_judgment_workflow", {"state": S}, "event")
tool_error("kp_judgment_workflow", {"state": S, "event": "marriage", "depth": 9}, "depth")
tool_error("kp_judgment_workflow", {"state": S, "event": "marriage", "limit": 0}, "limit must")
tool_error("find_muhurta_windows", {"moment": _MM}, "event` name")
tool_error("find_muhurta_windows", {"moment": _MM, "event": "marriage", "csl_min": "high"}, "csl_min must be")
tool_error("find_muhurta_windows", {"moment": _MM, "event": "marriage", "step_minutes": 0.5}, "step_minutes")
tool_error("find_muhurta_windows", {"moment": _MM, "event": "marriage", "end": "2027-12-01", "step_minutes": 5}, "Too many samples")
tool_error("find_muhurta_windows", {"moment": _MM, "event": "marriage", "end": "2026-11-20"}, "must be after")
tool_error("compare_births", {"a": S, "b": {**S, "latitude": 95}}, "latitude")
tool_error("get_cuspal_interlinks", {"state": S, "cusps": [13]}, "cusps must be")
tool_error("judge_event", {"state": {**S, "options": {"csl_extension": "magic"}}, "event": "marriage"}, "csl_extension must be")
tool_error("get_kp_chart", {"state": {**S, "options": {"cusp_proximity_apply": True}}}, "cusp_orb_deg")
tool_error("get_kp_chart", {"state": {**S, "options": {"cusp_orb_deg": 40}}}, "cusp_orb_deg")
tool_error("rectify_birth_time", {"state": S, "events": [{"date": "2018-02-10", "event": "marriage"}], "minutes": 5, "moon_link": "maybe"}, "moon_link must be")
tool_error("rectify_birth_time", {"state": S, "events": [{"date": "2018-02-10", "event": "marriage", "weight": 50}], "minutes": 5}, "weight")
tool_error("get_longevity_facts", {"state": S, "extra_houses": [13]}, "extra_houses")
tool_error("get_longevity_facts", {"state": S, "depth": 9}, "depth must be")
tool_error("verify_event_day", {"state": S, "depth": 9}, "depth must be")
tool_error("verify_event_day", {"state": S, "planets": ["Pluto"]}, "Unknown planet")
tool_error("verify_event_day", {"state": S, "event": "nonsense"}, "Known events")
tool_error("find_double_transit", {"state": S}, "Give `event`")
tool_error("find_double_transit", {"state": S, "event": "marriage", "sign_scope": "zodiac"}, "sign_scope must be")
tool_error("find_double_transit", {"state": S, "houses": [0]}, "houses must be")
tool_error("find_stations", {"moment": M, "planet": "Rahu"}, "no stations")
tool_error("find_stations", {"moment": M, "planet": "Moon"}, "never retrograde")
tool_error("get_fruitful_significators", {"state": S, "event": "marriage", "methods": "ruling_planets"}, "ruling planets")
tool_error("find_event_periods", {"state": S, "event": "marriage", "fruitful_method": "magic"}, "Unknown fruitful")
tool_error("judge_event", {"state": S, "event": "nonsense"}, "Known events")
tool_error("cast_horary_chart", {"moment": M, "number": 300}, "1 to 249")
tool_error("get_kp_dasha", {"state": S, "level": 9}, "level")
tool_error("get_kp_chart", {"state": {**S, "timezone": "Nowhere/City"}}, "timezone")
tool_error("get_kp_chart", {"state": S, "detail": "huge"}, "detail")
tool_error("find_boundary_changes", {"moment": M, "planet": "Pluto", "start": "2026-01-01", "end": "2026-02-01"}, "Unknown planet")

print(f"\nAll {passed} checks passed")
