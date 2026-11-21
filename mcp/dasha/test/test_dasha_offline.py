"""Offline tests for the Dasha MCP (stubbed PyJHora; nothing needs to be installed).

Run:  python mcp/test_dasha_offline.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dasha_engine as eng  # noqa: E402
from state import BirthState  # noqa: E402

ORDER = [8, 5, 0, 1, 2, 7, 4, 6, 3]
YEARS = {8: 7, 5: 20, 0: 6, 1: 10, 2: 7, 7: 18, 4: 16, 6: 19, 3: 17}
NAMES = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]
BIRTH = datetime(1990, 5, 15, 10, 30)


def tup(dt):
    return (dt.year, dt.month, dt.day, dt.hour + dt.minute / 60 + dt.second / 3600)


def dt_of(t):
    return datetime(t[0], t[1], t[2]) + timedelta(hours=t[3])


def split(lord, start, end):
    """Vimsottari-style split of [start, end] starting at `lord`."""
    i = ORDER.index(lord)
    seq = ORDER[i:] + ORDER[:i]
    total, cur, rows = (end - start).total_seconds(), start, []
    for p in seq:
        nxt = cur + timedelta(seconds=total * YEARS[p] / 120)
        rows.append(((p,), NAMES[p], tup(cur), tup(nxt)))
        cur = nxt
    return rows


def roots(*_a, **_k):
    start = BIRTH - timedelta(days=365.2425 * 15)  # Venus, 15 years already elapsed at birth
    rows, cur = [], start
    for p in ORDER[1:] + ORDER[:1]:  # Venus first
        nxt = cur + timedelta(days=365.2425 * YEARS[p])
        rows.append(((p,), NAMES[p], tup(cur), tup(nxt)))
        cur = nxt
    return rows


def children(name, lords, start, parent_end=None, **_k):
    return split(lords[-1], dt_of(start), dt_of(parent_end))


stub = SimpleNamespace(
    const=SimpleNamespace(
        nakshathra_dhasas=["yogini"], karaka_dhasas=[], special_dhasas=[], rasi_dhasas=["chara"],
        DHASA_YEAR_DURATION={"SAVANA_YEAR": 5}, SAVANA_YEAR_METHOD={"MIDPOINT": 0},
        dhasa_year_duration_default=4,
    ),
    utils=SimpleNamespace(julian_day_number=lambda d, t: 2448026.0, get_place_timezone_offset=lambda *a: 5.5),
    drik=SimpleNamespace(set_ayanamsa_mode=lambda m: None, Place=lambda *a, **k: ("place", a), Date=lambda *a: a),
    selector=SimpleNamespace(
        list_supported_dhasas=lambda: ["vimsottari", "yogini", "chara"],
        is_dhasa_supported=lambda n: False,
        get_dhasa_root_nodes=roots,
        get_dhasa_children=children,
    ),
    applicability=SimpleNamespace(applicability_check=lambda *a, **k: ["ashtottari"]),
    charts=SimpleNamespace(
        divisional_chart=lambda *a, **k: [["L", (0, 10.0)], [0, (4, 5.0)], [1, (3, 20.0)], [2, (9, 1.0)],
                                          [3, (5, 2.0)], [4, (3, 3.0)], [5, (1, 4.0)], [6, (6, 5.0)],
                                          [7, (10, 6.0)], [8, (4, 6.0)]]
    ),
)
eng.set_backend(stub)

state = BirthState(
    date_of_birth="1990-05-15", time_of_birth="10:30", latitude=19.07, longitude=72.87, timezone="Asia/Kolkata"
)
passed = 0


def check(label, cond):
    global passed
    assert cond, f"FAILED: {label}"
    passed += 1
    print(f"ok  {label}")


# systems / alias
check("alias resolves", eng.resolve_system("Vimshottari") == "vimsottari")
try:
    eng.resolve_system("nonsense")
    check("unknown system rejected", False)
except eng.DashaError as e:
    check("unknown system rejected with hint", "Unknown dasha system" in str(e))
check("list_systems count", eng.list_systems()["count"] == 3)

# basis / applicability
b = eng.get_basis(state)
check("moon nakshatra computed", b["moon"]["nakshatra"] == "Ashlesha")  # Cancer 20deg = 110deg abs (106.67-120)
check("lagna sign", b["lagna"]["sign"] == "Aries")
check("applicability", eng.check_applicability(state)["applicable_conditional_dashas"] == ["ashtottari"])

# balance
bal = eng.get_balance(state)
check("balance running lord is Venus", bal["running_at_birth"]["lord"] == "Venus")
check("balance ~5 years", abs(bal["balance_years"] - 5.0) < 0.01)

# mahadashas
md = eng.get_mahadashas(state)
check("9 mahadashas", md["count"] == 9 and md["mahadashas"][0]["running_at_birth"])

# sub periods
sp = eng.get_sub_periods(state, maha="Venus", levels_below=2)
check("9 antardashas", len(sp["sub_periods"]) == 9 and "children" in sp["sub_periods"][0])
sp2 = eng.get_sub_periods(state, maha=1, path=["Venus", "Sun"])
check("path descent", sp2["parent_path"] == ["Venus", "Venus", "Sun"] or sp2["parent_path"][-1] == "Sun")

# current / at age / upcoming
cur = eng.get_current(state, on="2026-11-18", depth=5)
check("ladder depth 5", len(cur["levels"]) == 5)
check("ladder dates nest", all(
    cur["levels"][i]["start"] <= cur["levels"][i + 1]["start"] for i in range(4)))
at0 = eng.get_at_age(state, age=0.0, depth=2)
check("age 0 => Venus", at0["levels"][0]["lord"] == "Venus")
up = eng.get_upcoming(state, on="2026-11-18", level=2, count=3)
check("upcoming returns running + 3", len(up["periods"]) == 4 and up["periods"][0]["status"] == "running")
try:
    eng.get_current(state, on="1800-01-01")
    check("outside range rejected", False)
except eng.DashaError:
    check("outside range rejected", True)

# timeline / search
tl = eng.get_timeline(state, start="2020-01-01", end="2030-01-01", depth=2)
check("timeline non-empty", tl["count"] > 0)
se = eng.search_periods(state, lord="Jupiter", level=2, start="1990-05-15", end="2050-01-01")
check("search finds Jupiter antardashas", se["count"] >= 1 and all(p["lord"] == "Jupiter" for p in se["periods"]))

# compare / analyze
cmp_ = eng.compare_systems(state, ["vimsottari", "bogus"], on="2026-11-18")
check("compare handles bad system", "error" in cmp_["systems"]["bogus"] and "running" in cmp_["systems"]["vimsottari"])
an = eng.analyze_lords(state, on="2026-11-18", depth=2)
check("analysis has position + relationship", an["levels"][0]["position"] is not None and len(an["lord_relationships"]) == 1)

print(f"\nAll {passed} checks passed")
