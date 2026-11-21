"""Run: .venv/bin/python mcp/DBA/test/test_validate.py"""
import json, os, random, sys, tempfile
from fractions import Fraction as Fr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import dba_validate as dv
import dba_algebra as alg

F = []
def check(n, c, d=""):
    print("ok   " if c else "FAIL ", n, "" if c else d)
    if not c: F.append(n)

AGE = dv.AGE_YEAR_DAYS
OBS_HI = 80 * AGE
S = ["Sun", "Moon", "Mars"]

def rs(rules, **kw):
    base = dict(event_type="marriage", n_perm=300)
    base.update(kw)
    return dv.RuleSet(rules=tuple(rules), **base)

def lord_rule(level, cons, mode="all", label=""):
    return dv.RuleSpec("lord_sets", {"level": level, "constraints": cons, "mode": mode}, label)

def day_event(age_years):
    a = Fr(int(age_years * 365.2425))
    return dv.PEvent("marriage", a, a + 1, "day")

# ---- statistics helpers
lo, hi = dv.wilson(5, 10)
check("wilson(5,10)", abs(lo - 0.2366) < 1e-3 and abs(hi - 0.7634) < 1e-3, (lo, hi))
up, low = dv.poisson_binomial_tail([0.5, 0.5], 1)
check("poisson-binomial two coins", abs(up - 0.75) < 1e-12 and abs(low - 0.75) < 1e-12, (up, low))
check("poisson-binomial k=0 is 1", abs(dv.poisson_binomial_tail([0.1] * 3, 0)[0] - 1.0) < 1e-12)
check("BH adjust", [round(x, 6) for x in dv.bh_adjust([0.01, 0.04, 0.03, 0.005])] == [0.02, 0.04, 0.04, 0.02])
check("bonferroni", [round(x, 9) for x in dv.bonferroni([0.01, 0.4, 0.6])] == [0.03, 1.0, 1.0])

# ---- event intervals and loaders
a, b = dv.event_interval(dv.Event("m", "2020-12"))
check("month interval wraps December", (a.year, a.month, b.year, b.month) == (2020, 12, 2021, 1))
a, b = dv.event_interval(dv.Event("m", "2020"))
check("year interval", (a.year, b.year) == (2020, 2021))
try:
    dv.event_interval(dv.Event("m", "2020-05", "day")); ok = False
except ValueError:
    ok = True
check("precision mismatch rejected", ok)
good = {"id": "c1", "birth": "1990-05-15T10:30", "timezone": "Asia/Kolkata", "accuracy": "AA",
        "events": [{"type": "marriage", "date": "2015-02-14"}]}
check("json case loads", len(dv.load_cases_json([good])) == 1)
for bad, name in [({**good, "longitude": 72.8}, "both tz and longitude"), ({**good, "timezone": ""}, "neither tz nor longitude"),
                  ({**good, "accuracy": "Z"}, "bad grade")]:
    try:
        dv.load_cases_json([bad]); ok = False
    except ValueError:
        ok = True
    check("rejects " + name, ok)
try:
    dv.load_cases_json([good, good]); ok = False
except ValueError:
    ok = True
check("rejects duplicate ids", ok)
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "cases.csv")
    with open(p, "w") as fh:
        fh.write("case_id,birth,timezone,longitude,accuracy,observed_until,event_type,event_date,precision,source\n")
        fh.write("c1,1990-05-15T10:30,Asia/Kolkata,,AA,2025-01-01,marriage,2015-02,,diary\n")
        fh.write("c1,1990-05-15T10:30,Asia/Kolkata,,AA,2025-01-01,job,2012-07-01,day,cv\n")
        fh.write("c2,1985-01-02T04:00,,72.8,B,,,,,\n")
    cs = dv.load_cases_csv(p)
    check("csv groups events per case", len(cs) == 2 and len(cs[0].events) == 2 and len(cs[1].events) == 0 and cs[1].longitude == 72.8)

# ---- real conversion: local midnight -> days from birth
case = dv.load_cases_json([good])[0]
prep = dv.prepare(case, rs([lord_rule(1, {"1": ["Sun"]})]))
ev = prep.events[0]
check("event day count from birth", 9000 < ev.a < 9100 and ev.b - ev.a == 1, float(ev.a))
check("moon longitude sane", 0 <= prep.moon_deg < 360)

# ---- exact coverage of a rule (Moon at 0 deg, one full 120-year cycle)
yd = Fr(36525, 100)
p0 = dv.Prepared("x", Fr(0), (Fr(0), 120 * yd), [])
w = dv.case_windows(rs([lord_rule(2, {"1": S, "2": S})]), p0)
check("lord_sets coverage = (23/120)^2", alg.coverage_fraction(w, 0, 120 * yd) == Fr(23, 120) ** 2)
w = dv.case_windows(rs([lord_rule(2, {"1": S, "2": S}, "any")]), p0)
check("any-mode coverage = 1-(97/120)^2", alg.coverage_fraction(w, 0, 120 * yd) == 1 - Fr(97, 120) ** 2)
w = dv.case_windows(rs([lord_rule(2, {"1": S}), lord_rule(2, {"2": S})], combine="intersect"), p0)
check("intersect of Maha and Antar rules = (23/120)^2", alg.coverage_fraction(w, 0, 120 * yd) == Fr(23, 120) ** 2)

# ---- planted signal vs no signal (random Moons)
rng = random.Random(7)
rule_any = lord_rule(2, {"1": S, "2": S}, "any", "either lord in Sun/Moon/Mars")

def make(n, frac_in_window, ruleset, prefix="c"):
    prepared = []
    for i in range(n):
        m = Fr(rng.random() * 360)
        pr = dv.Prepared(prefix + str(i), m, (Fr(0), OBS_HI), [])
        wins = dv.case_windows(ruleset, pr)
        if rng.random() < frac_in_window:
            ws = [(max(x.start, 20 * AGE), min(x.end, 60 * AGE)) for x in wins]
            ws = [w_ for w_ in ws if w_[1] - w_[0] > 5]
            x0, x1 = rng.choice(ws)
            t = float(x0) + rng.random() * float(x1 - x0 - 2)
        else:
            t = rng.uniform(15, 60) * float(AGE)
        a = Fr(int(t))
        pr.events.append(dv.PEvent("marriage", a, a + 1, "day"))
        prepared.append(pr)
    return prepared

R = rs([rule_any])
rep = dv.score(R, make(120, 0.7, R))
check("planted: hit rate high", rep["hit_rate"] > 0.75, rep["hit_rate"])
check("planted: lift above 1.5 and p tiny", rep["coverage_null"]["lift"] > 1.5 and rep["coverage_null"]["p_upper"] < 1e-4, rep["coverage_null"])
check("planted: shuffle control also rejects", rep["shuffle_control"]["p_upper"] < 0.01, rep["shuffle_control"])
check("planted: CI contains hit rate", rep["hit_rate_ci95"][0] < rep["hit_rate"] < rep["hit_rate_ci95"][1])
rep0 = dv.score(R, make(120, 0.0, R))
check("no signal: lift near 1", 0.6 < rep0["coverage_null"]["lift"] < 1.6, rep0["coverage_null"]["lift"])
check("no signal: p not small", rep0["coverage_null"]["p_upper"] > 0.01 and rep0["shuffle_control"]["p_upper"] > 0.01,
      (rep0["coverage_null"]["p_upper"], rep0["shuffle_control"]["p_upper"]))


# ---- probability scoring: fit r on one set, score on another
fit_s, test_s = make(100, 0.7, R, "f"), make(100, 0.7, R, "t")
pr_ = dv.probability_report(R, fit_s, test_s)
check("probability planted: r > 1.5, gain CI above 0, BSS > 0",
      pr_["r_fitted"] > 1.5 and pr_["log_score_gain_ci95"][0] > 0 and pr_["brier_skill_score"] > 0, pr_)
check("probability planted: used age-density baseline, held-out flag", pr_["baseline_used"] == "age_density" and not pr_["in_sample"])
cal = pr_["calibration"]
n_ev = pr_["n_events_scored"]
check("calibration: forecast and observed both sum to the event count",
      len(cal) == 8 and abs(sum(c["mean_forecast"] * c["n_bins"] for c in cal) - n_ev) < 1e-6
      and abs(sum(c["observed_rate"] * c["n_bins"] for c in cal) - n_ev) < 1e-6, (n_ev, cal[-1]))
check("calibration: top group observed rate above bottom group", cal[-1]["observed_rate"] > cal[0]["observed_rate"])
nf, nt = make(100, 0.0, R, "nf"), make(100, 0.0, R, "nt")
pn = dv.probability_report(R, nf, nt)
check("probability null held-out: no real gain (CI lower bound below 0.05)", pn["log_score_gain_ci95"][0] < 0.05 and pn["log_score_gain_per_event_nats"] < 0.05, pn["log_score_gain_per_event_nats"])
pin = dv.probability_report(R, nf, nf, in_sample=True)
check("probability null in-sample: gain never negative (nested model)", pin["log_score_gain_per_event_nats"] >= -1e-9 and pin["in_sample"], pin["log_score_gain_per_event_nats"])
pu = dv.probability_report(rs([rule_any], baseline="uniform"), fit_s, test_s)
check("uniform baseline option works and still sees the signal", pu["baseline_used"] == "uniform" and pu["log_score_gain_per_event_nats"] > 0)
few = dv.probability_report(R, fit_s[:6], test_s[:6])
check("age-density baseline falls back to uniform with too few events", few["baseline_used"] == "uniform")
check("bin_years changes the hash", rs([rule_any]).hash() != rs([rule_any], bin_years=0.5).hash())

# ---- chart-matching test
mt = dv.matching_test(R, fit_s, test_s)
check("matching planted: true chart ranks above chance, p small", mt["mean_percentile_of_true_chart"] > 0.55
      and mt["p_upper_mean_percentile"] < 0.01 and mt["top_k_hits"] > mt["top_k_expected_by_chance"], mt)
mn = dv.matching_test(R, nf, nt)
check("matching null: percentile near 0.5, p not small", abs(mn["mean_percentile_of_true_chart"] - 0.5) < 0.07
      and mn["p_upper_mean_percentile"] > 0.01, mn)
shuf = test_s[:]
random.Random(1).shuffle(shuf)
mt2 = dv.matching_test(R, fit_s, shuf)
check("matching: result does not depend on input order (blind shuffling inside)",
      abs(mt2["mean_percentile_of_true_chart"] - mt["mean_percentile_of_true_chart"]) < 1e-9, (mt2["mean_percentile_of_true_chart"], mt["mean_percentile_of_true_chart"]))
check("matching: too few people handled", "note" in dv.matching_test(R, fit_s, test_s[:3]) and dv.matching_test(R, fit_s, test_s[:3])["n_pairs"] == 3)

# ---- age clustering: coverage null is fooled, age-density null is not
rng2 = random.Random(3)
rv = rs([lord_rule(1, {"1": ["Venus"]}, label="Venus Maha")])
prepared = []
for i in range(400):
    pr = dv.Prepared("a%d" % i, Fr(0), (Fr(0), OBS_HI), [day_event(rng2.uniform(20, 40))])
    prepared.append(pr)
rep = dv.score(rv, prepared)                       # Moon fixed at 0: Venus Maha spans ages 7-27
cov_chance, ad = rep["coverage_null"]["chance_hit_rate"], rep["age_density_null"]
check("clustered ages: hit rate ~ 0.35", abs(rep["hit_rate"] - 0.35) < 0.08, rep["hit_rate"])
check("clustered ages: coverage null says ~0.25", abs(cov_chance - 0.25) < 0.02, cov_chance)
check("clustered ages: age-density chance closer to observed", abs(rep["hit_rate"] - ad["chance_hit_rate"]) < abs(rep["hit_rate"] - cov_chance),
      (rep["hit_rate"], ad["chance_hit_rate"], cov_chance))
check("clustered ages: age-density p not significant, coverage p is", ad["p_upper"] > 0.05 and rep["coverage_null"]["p_upper"] < 0.05,
      (ad["p_upper"], rep["coverage_null"]["p_upper"]))

# ---- precision awareness: a year-long event is much easier to hit
pm = dv.Prepared("p", Fr(0), (Fr(0), OBS_HI), [])
yr = dv.PEvent("marriage", Fr(int(30 * 365.2425)), Fr(int(30 * 365.2425)) + 365, "year")
pm.events.append(yr)
pm.events.append(day_event(30))
rep = dv.score(rs([rule_any], min_precision="year", n_perm=0), [pm])
rows = dv.score(rs([rule_any], min_precision="year", n_perm=0), [pm], include_events=True)["events"]
check("year-precision chance exceeds day-precision chance", rows[0]["chance"] > rows[1]["chance"], rows)
rep = dv.score(rs([rule_any], min_precision="day", n_perm=0), [pm])
check("min_precision drops the coarse event", rep["n_events"] == 1 and rep["excluded"]["precision_too_coarse"] == 1, rep["excluded"])

# ---- exclusions
pr = dv.Prepared("e", Fr(0), (Fr(0), 50 * AGE), [dv.PEvent("job", Fr(1000), Fr(1001), "day"),
                                                   day_event(70), day_event(10)])
rep = dv.score(rs([rule_any], n_perm=0), [pr])
check("outside window / wrong type excluded", rep["n_events"] == 1 and rep["excluded"]["wrong_type"] == 1
      and rep["excluded"]["outside_observation_window"] == 1, rep["excluded"])

# ---- stacking guard
st = dv.stack_report(rs([lord_rule(1, {"1": ["Venus"]}, label="Venus"), lord_rule(1, {"1": ["Sun"]}, label="Sun"),
                         lord_rule(1, {"1": ["Moon", "Mars"]}, label="Moon/Mars")]), prepared)["steps"]
check("stacking: chance and hit rate never fall", all(b["chance_hit_rate"] >= a["chance_hit_rate"] and b["hit_rate"] >= a["hit_rate"]
      for a, b in zip(st, st[1:])), st)
check("stacking: increments reported", "added_lift_of_increment" in st[1])

# ---- pre-registration, split, looks
check("hash stable under param key order",
      rs([dv.RuleSpec("lord_sets", {"level": 2, "constraints": {"2": ["Moon", "Sun"], "1": ["Mars"]}})]).hash()
      == rs([dv.RuleSpec("lord_sets", {"constraints": {"1": ["Mars"], "2": ["Sun", "Moon"]}, "level": 2})]).hash())
check("hash changes when an option changes", rs([rule_any]).hash() != rs([rule_any], ayanamsa="RAMAN").hash())
rngc = random.Random(11)
cases = []
for i in range(24):
    y, mo, d = rngc.randint(1960, 1995), rngc.randint(1, 12), rngc.randint(1, 28)
    age = rngc.randint(22, 38)
    cases.append(dv.Case("r%d" % i, "%d-%02d-%02dT%02d:%02d" % (y, mo, d, rngc.randint(0, 23), rngc.randint(0, 59)), "Asia/Kolkata",
                         None, "A", "2025-01-01", (dv.Event("marriage", "%d-%02d" % (y + age, rngc.randint(1, 12))),)))
t1, h1 = dv.split_cases(cases, 0, 0.5)
t2, h2 = dv.split_cases(cases, 0, 0.5)
check("split deterministic and exhaustive", [c.id for c in t1] == [c.id for c in t2] and len(t1) + len(h1) == 24 and 6 <= len(t1) <= 18)
with tempfile.TemporaryDirectory() as td:
    reg = os.path.join(td, "prereg.json")
    RS = rs([rule_any], n_perm=50)
    try:
        dv.evaluate(RS, cases, "heldout", reg); ok = False
    except ValueError as exc:
        ok = "not pre-registered" in str(exc)
    check("held-out refused before registration", ok)
    r = dv.evaluate(RS, cases, "tune", reg)
    check("tune allowed unregistered, flagged exploratory", r["registration"]["pre_registered"] is False and "warning" in r["registration"])
    h = dv.register(RS, reg, "test")
    check("register is idempotent", dv.register(RS, reg) == h == RS.hash())
    r = dv.evaluate(RS, cases, "heldout", reg)
    check("held-out allowed after registration, first look has no warning",
          r["registration"]["pre_registered"] and r["registration"]["looks_this_hash"]["heldout"] == 1 and "warning" not in r["registration"], r["registration"])
    r = dv.evaluate(RS, cases, "heldout", reg)
    check("second held-out look is counted and warned", r["registration"]["looks_this_hash"]["heldout"] == 2
          and r["registration"]["heldout_looks_total_all_hashes"] == 2 and "warning" in r["registration"])
    edited = rs([rule_any], n_perm=50, ayanamsa="RAMAN")
    try:
        dv.evaluate(edited, cases, "heldout", reg); ok = False
    except ValueError:
        ok = True
    check("edited rule set cannot reach held-out cases", ok)
    check("report carries provenance hash", r["provenance"]["ruleset_hash"] == RS.hash())
    check("registry file is valid json with the entry", RS.hash() in json.load(open(reg))["entries"])


# ---- true_solar_arc per case (map built from each case's own birth instant)
_c = dv.Case("arc1", "1990-05-15T10:30", timezone="Asia/Kolkata", events=(dv.Event("marriage", "2015-06-10"),))
_rule = lord_rule(2, {"1": S, "2": S})
_pj = dv.prepare(_c, rs([_rule], year_days="julian"))
_pa = dv.prepare(_c, rs([_rule], year_days="true_solar_arc"))
check("prepare resolves a per-case Sun-arc map", _pa.year_days is not None and str(_pa.year_days) == "true_solar_arc" and _pj.year_days == Fr(1461, 4))
_wj = dv.case_windows(rs([_rule], year_days="julian"), _pj)
_wa = dv.case_windows(rs([_rule], year_days="true_solar_arc"), _pa)
_cj = sum(w.end - w.start for w in _wj) / (_pj.obs[1] - _pj.obs[0])
_ca = sum(w.end - w.start for w in _wa) / (_pa.obs[1] - _pa.obs[0])
check("arc windows exist and coverage is close to the julian one (%.3f vs %.3f)" % (float(_cj), float(_ca)),
      len(_wa) > 0 and abs(float(_cj - _ca)) < 0.03)

print("ALL PASSED" if not F else "FAILED %d: %s" % (len(F), F))
sys.exit(1 if F else 0)
