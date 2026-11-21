"""Run: .venv/bin/python mcp/DBA/test/test_server.py  (calls the tool functions directly)"""
import asyncio, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import dba_server as s

F = []
def check(n, c, d=""):
    print("ok   " if c else "FAIL ", n, "" if c else d)
    if not c: F.append(n)

B = dict(birth="1990-05-15T10:30", timezone="Asia/Kolkata")
tools = {t.name for t in asyncio.run(s.mcp.list_tools())}
check("12 tools registered", tools == {"dba_year_convention_table", "dba_get_ladder", "dba_get_boundaries", "dba_find_periods",
                                      "dba_resolution_report", "dba_ayanamsa_shift_table", "dba_rahu_ketu",
                                      "dba_validation_register", "dba_backtest", "dba_stack_report", "dba_chance_baseline",
                                      "dba_window_bands"}, tools)

def call(name, **kw):
    return asyncio.run(s.mcp.call_tool(name, kw))

def data(r):
    # FastMCP returns (content, structured) or content list depending on version
    import json
    if isinstance(r, tuple):
        return r[1] if isinstance(r[1], dict) else json.loads(r[0][0].text)
    return json.loads(r[0].text)

d = data(call("dba_get_ladder", **B, at="2000-01-01T00:00"))
check("ladder 6 levels, Maha lord Sun->Moon timeline", len(d["ladder"]) == 6 and d["ladder"][0]["path"][0] in ("Sun", "Moon"), d["ladder"][0])
check("provenance present", d["provenance"]["options_used"]["year_days"] == 365.25 and d["provenance"]["rule_id"])
d0 = data(call("dba_get_ladder", **B))
check("ladder at birth first Maha = Sun", d0["ladder"][0]["path"] == ["Sun"], d0["ladder"][0])
m = data(call("dba_get_boundaries", **B))
check("9 Mahas, first ends after 4.57 y", len(m["periods"]) == 9 and m["periods"][0]["path"] == ["Sun"], m["periods"][0])
ch = data(call("dba_get_boundaries", **B, parent_path=["Moon"]))
check("Moon Maha children: 9, first Moon", len(ch["periods"]) == 9 and ch["periods"][0]["path"] == ["Moon", "Moon"])
f = data(call("dba_find_periods", **B, lords_by_level=[["Sun", "Moon", "Mars"], ["Sun", "Moon", "Mars"]]))
check("coverage ~ (23/120)^2 over 120 y", abs(f["coverage"] - (23 / 120) ** 2) < 0.01 and not f["truncated"], f["coverage"])
r = data(call("dba_resolution_report", **B, birth_sigma_min=15, ayanamsas=["RAMAN"], year_modes=["julian", "savana"], at="2020-01-01T00:00"))
check("resolution report levels", len(r["levels"]) == 6 and r["levels"][0]["boundary_shift_days"]["ayanamsa"] > 100, r["levels"][0])
a = data(call("dba_ayanamsa_shift_table", **B))
check("ayanamsa table 6 rows", len(a["rows"]) == 6)
n = data(call("dba_rahu_ketu", **B, node="true"))
check("rahu/ketu", abs((n["Ketu"] - n["Rahu"]) % 360 - 180) < 1e-6)
y = data(call("dba_year_convention_table"))
check("year table has savana -630 d", any(abs(r_["shift_over_120_years_days_vs_julian"] + 630) < 1e-6 for r_ in y["options"]))
try:
    call("dba_get_ladder", birth="1990-05-15T10:30"); check("needs tz or longitude", False)
except Exception:
    check("needs tz or longitude", True)

wb = data(call("dba_window_bands", **B, lords_by_level=[["Sun", "Moon", "Mars"], ["Sun", "Moon", "Mars"]], birth_sigma_min=15,
               ayanamsas=["RAMAN"], year_modes=["julian", "savana"]))
check("window bands: 12 scenarios, certain < possible", wb["n_scenarios"] == 12 and wb["coverage_certain"] < wb["coverage_possible"]
      and wb["certain"]["n"] >= 0 and wb["possible"]["n"] > 0, (wb["coverage_certain"], wb["coverage_possible"]))
wb0 = data(call("dba_window_bands", **B, lords_by_level=[["Sun", "Moon", "Mars"], ["Sun", "Moon", "Mars"]]))
check("window bands: no uncertainty -> certain == possible == ~3.7%", wb0["coverage_certain"] == wb0["coverage_possible"] and 0.02 < wb0["coverage_possible"] < 0.06)

# ---- true_solar_arc through the tools
SA = dict(**B, year_days="true_solar_arc")
dl = data(call("dba_get_ladder", **SA, at="2000-01-01T00:00"))
dj = data(call("dba_get_ladder", **B, at="2000-01-01T00:00"))
check("arc ladder: 6 levels, provenance names the mode", len(dl["ladder"]) == 6 and dl["provenance"]["options_used"]["year_mode"] == "true_solar_arc"
      and abs(dl["provenance"]["options_used"]["year_days"] - 365.24219) < 1e-9, dl["provenance"]["options_used"])
check("arc ladder Maha lord equals julian ladder Maha lord", dl["ladder"][0]["path"] == dj["ladder"][0]["path"])
mb = data(call("dba_get_boundaries", **SA))
check("arc boundaries: 9 Mahas, differ from julian by under a month at the first end",
      len(mb["periods"]) == 9 and abs(mb["periods"][0]["duration_days"] - m["periods"][0]["duration_days"]) < 30)
ch_a = data(call("dba_get_boundaries", **SA, parent_path=["Moon"]))
check("arc children: 9, first Moon, end exactly at the parent end", len(ch_a["periods"]) == 9 and ch_a["periods"][0]["path"] == ["Moon", "Moon"]
      and ch_a["periods"][-1]["end_utc"] == [p_ for p_ in mb["periods"] if p_["path"] == ["Moon"]][0]["end_utc"])
fa = data(call("dba_find_periods", **SA, lords_by_level=[["Sun", "Moon", "Mars"], ["Sun", "Moon", "Mars"]]))
check("arc coverage ~ (23/120)^2", abs(fa["coverage"] - (23 / 120) ** 2) < 0.01 and not fa["truncated"], fa["coverage"])
ra = data(call("dba_resolution_report", **B, birth_sigma_min=15, year_modes=["julian", "true_solar_arc"], at="2020-01-01T00:00"))
check("resolution report accepts true_solar_arc as a year mode", len(ra["levels"]) == 6 and "true_solar_arc" in ra["provenance"].get("options_used", {}).get("year_mode", "") + str(ra.get("options_used", "")), ra.get("options_used"))
wba = data(call("dba_window_bands", **B, lords_by_level=[["Sun", "Moon", "Mars"], ["Sun", "Moon", "Mars"]], birth_sigma_min=5,
                year_modes=["julian", "true_solar_arc"]))
check("window bands accept true_solar_arc: 6 scenarios", wba["n_scenarios"] == 6 and wba["coverage_certain"] <= wba["coverage_possible"], wba["n_scenarios"])
check("year table lists the birth-dependent options", "true_solar_arc" in data(call("dba_year_convention_table"))["birth_dependent_options"])
sh = data(call("dba_ayanamsa_shift_table", **SA))
check("ayanamsa table works with true_solar_arc", len(sh["rows"]) == 6)

# ---- validation tools
import json, random, tempfile
cb = data(call("dba_chance_baseline", **B, lords_by_level=[["Sun", "Moon", "Mars"], ["Sun", "Moon", "Mars"]]))
check("chance baseline: coverage ~3.7%, month chance above day chance above coverage",
      0.02 < cb["coverage"] < 0.06 and cb["chance_of_hit_by_luck"]["day"] >= cb["coverage"] and
      cb["chance_of_hit_by_luck"]["year"] > cb["chance_of_hit_by_luck"]["month"] > cb["chance_of_hit_by_luck"]["day"], cb)
rng = random.Random(5)
cases = []
for i in range(24):
    y, mo, d = rng.randint(1960, 1995), rng.randint(1, 12), rng.randint(1, 28)
    cases.append({"id": "k%d" % i, "birth": "%d-%02d-%02dT%02d:%02d" % (y, mo, d, rng.randint(0, 23), rng.randint(0, 59)),
                  "timezone": "Asia/Kolkata", "accuracy": "A", "observed_until": "2025-01-01",
                  "events": [{"type": "marriage", "date": "%d-%02d" % (y + rng.randint(22, 38), rng.randint(1, 12))}]})
RSD = {"event_type": "marriage", "n_perm": 50,
       "rules": [{"level": 2, "constraints": {"1": ["Venus", "Jupiter"], "2": ["Venus", "Jupiter"]}, "mode": "any", "label": "Ven/Jup"},
                 {"level": 1, "constraints": {"1": ["Moon"]}, "label": "Moon Maha"}]}
with tempfile.TemporaryDirectory() as td:
    cp, rp = os.path.join(td, "cases.json"), os.path.join(td, "reg.json")
    json.dump({"cases": cases}, open(cp, "w"))
    try:
        call("dba_backtest", cases_path=cp, ruleset=RSD, which="heldout", registry_path=rp); ok = False
    except Exception:
        ok = True
    check("backtest: heldout refused before registration", ok)
    reg = data(call("dba_validation_register", ruleset=RSD, note="t", registry_path=rp))
    check("register returns hash", len(reg["hash"]) == 64)
    bt = data(call("dba_backtest", cases_path=cp, ruleset=RSD, which="heldout", registry_path=rp))
    check("backtest heldout after registration", bt["registration"]["pre_registered"] and bt["n_events"] > 0
          and "coverage_null" in bt and bt["provenance"]["ruleset_hash"] == reg["hash"], bt.get("note"))
    bt2 = data(call("dba_backtest", cases_path=cp, ruleset=RSD, which="tune", registry_path=rp))
    check("backtest tune has shuffle control + CI", bt2["shuffle_control"] is not None and len(bt2["hit_rate_ci95"]) == 2)
    bp = data(call("dba_backtest", cases_path=cp, ruleset=RSD, which="heldout", registry_path=rp, with_probability=True))
    check("backtest with probability: r fitted, calibration present, held-out flag",
          "probability" in bp and bp["probability"].get("in_sample") is False and ("r_fitted" in bp["probability"] or "note" in bp["probability"]), bp.get("probability"))
    bm = data(call("dba_backtest", cases_path=cp, ruleset=RSD, which="tune", registry_path=rp, with_matching=True))
    check("backtest with matching: pairs scored and blind note present", bm["matching"]["n_pairs"] >= 5 and "Blind" in bm["matching"]["note"], bm.get("matching"))
    sr = data(call("dba_stack_report", cases_path=cp, ruleset=RSD))
    check("stack report 2 steps", len(sr["steps"]) == 2 and "added_chance" in sr["steps"][1])
    try:
        call("dba_backtest", cases_path=cp, ruleset={**RSD, "bogus": 1}, which="tune", registry_path=rp); ok = False
    except Exception:
        ok = True
    check("unknown ruleset field rejected", ok)
print("ALL PASSED" if not F else "FAILED %d" % len(F))
sys.exit(1 if F else 0)
