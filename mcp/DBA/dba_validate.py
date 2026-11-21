"""DBA Phase 4 (H): validation harness. Measures how specific and how accurate a timing rule is.

Facts only. Nothing here predicts anything; it scores a stated rule set against dated events the user supplies.

Pieces
  case store      Case / Event, JSON and CSV loaders (user-supplied data only, nothing is scraped)
  rules           RuleSpec ('lord_sets'), RuleSet (rules + options + split + test settings), frozen by a hash
  pre-registration  register() writes the hash BEFORE scoring; held-out cases are refused for unregistered hashes,
                  and every look is counted (held-out looks are reported, since each extra look weakens the result)
  metrics         hit rate (Wilson CI), exact coverage, chance hit rate, lift = hits / expected hits, p-values
  controls        (1) coverage null: Poisson-binomial with the exact chance of each event given its precision;
                  (2) age-density null: same, but events are not uniform in age (outside view from the other cases);
                  (3) chart-shuffle control: person A's windows against person B's events, permutation p-value
  stacking guard  stack_report(): chance rate and incremental lift as rules are added one by one
  multiplicity    bonferroni(), bh_adjust()

Conventions
  hit      = the event's interval [a, b) overlaps a rule window (day / month / year precision gives an interval)
  chance   = P(an interval of that length placed at random in the observation window overlaps a rule window), exact
             (windows widened by half the interval on both sides, measured over the valid midpoints)
  times    = exact Fractions of a day from birth; windows are half-open [start, end)
Probability scoring (probability_report): the rule is turned into a forecast over time bins, p(bin) proportional to
baseline(bin) x (1 + (r-1) x share of the bin inside a window); r is fitted on the tune split, scored on held-out with
log score, Brier score, paired bootstrap and a calibration table, against the baseline (r = 1).
Chart-matching test (matching_test, Carlson style): N charts against N shuffled event histories; each chart is scored by the
likelihood ratio its forecast gives to the event's time bin; rank of the true chart, top-k and percentile against a permutation null.
Not done yet: uncertainty bands on windows.
"""
import csv
import hashlib
import json
import math
import os
import random
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from fractions import Fraction

import dba_algebra as alg
import dba_astro as astro
import dba_vimshottari as vim
import dba_solararc as solar
from dba_time import to_utc

HARNESS_VERSION = "0.2"
AGE_YEAR_DAYS = Fraction(3652425, 10000)       # real-time age axis (not the dasha year option)
PRECISIONS = ("day", "month", "year")           # finest to coarsest
GRADES = ("AA", "A", "B", "C", "X", "?")        # Rodden-style, supplied by the user; '?' = unknown


# ---------------------------------------------------------------- case store
@dataclass(frozen=True)
class Event:
    type: str
    date: str                  # 'YYYY-MM-DD' | 'YYYY-MM' | 'YYYY'
    precision: str = ""        # inferred from the date string when empty
    source: str = ""


@dataclass(frozen=True)
class Case:
    id: str
    birth: str                 # local 'YYYY-MM-DDTHH:MM[:SS]'
    timezone: str = ""
    longitude: float = None
    accuracy: str = "?"
    observed_until: str = ""   # 'YYYY-MM-DD': last date the event history is complete for; '' = use the age range end
    events: tuple = ()


def _precision_of(s):
    return {10: "day", 7: "month", 4: "year"}.get(len(s.strip()))


def event_interval(ev):
    """(start, end) as naive local datetimes, end exclusive."""
    s = ev.date.strip()
    prec = ev.precision or _precision_of(s)
    if prec not in PRECISIONS or _precision_of(s) != prec:
        raise ValueError("event date %r does not match precision %r" % (ev.date, ev.precision))
    try:
        if prec == "day":
            d = date.fromisoformat(s)
            a = datetime(d.year, d.month, d.day)
            b = datetime.fromordinal(a.toordinal() + 1)
        elif prec == "month":
            y, m = int(s[:4]), int(s[5:7])
            a = datetime(y, m, 1)
            b = datetime(y + (m == 12), 1 if m == 12 else m + 1, 1)
        else:
            y = int(s)
            a, b = datetime(y, 1, 1), datetime(y + 1, 1, 1)
    except ValueError as exc:
        raise ValueError("bad event date %r: %s" % (ev.date, exc))
    return a, b


def _mk_case(d):
    cid = str(d.get("id", "")).strip()
    if not cid:
        raise ValueError("case without id")
    if not d.get("birth"):
        raise ValueError("case %s: birth missing" % cid)
    tz = (d.get("timezone") or "").strip()
    lon = d.get("longitude")
    lon = None if lon in (None, "") else float(lon)
    if bool(tz) == (lon is not None):
        raise ValueError("case %s: give exactly one of timezone or longitude" % cid)
    acc = (d.get("accuracy") or "?").strip() or "?"
    if acc not in GRADES:
        raise ValueError("case %s: accuracy %r not in %s" % (cid, acc, GRADES))
    datetime.fromisoformat(str(d["birth"]).strip())
    evs = []
    for e in d.get("events", []):
        ev = Event(str(e["type"]).strip(), str(e["date"]).strip(), (e.get("precision") or "").strip(), e.get("source", "") or "")
        event_interval(ev)                       # validates
        evs.append(ev)
    ou = (d.get("observed_until") or "").strip()
    if ou:
        date.fromisoformat(ou)
    return Case(cid, str(d["birth"]).strip(), tz, lon, acc, ou, tuple(evs))


def _check_unique(cases):
    seen = set()
    for c in cases:
        if c.id in seen:
            raise ValueError("duplicate case id %s" % c.id)
        seen.add(c.id)
    return cases


def load_cases_json(path_or_obj):
    obj = path_or_obj
    if isinstance(path_or_obj, (str, os.PathLike)):
        with open(path_or_obj, encoding="utf-8") as fh:
            obj = json.load(fh)
    rows = obj["cases"] if isinstance(obj, dict) else obj
    return _check_unique([_mk_case(d) for d in rows])


CSV_COLUMNS = ("case_id", "birth", "timezone", "longitude", "accuracy", "observed_until",
               "event_type", "event_date", "precision", "source")


def load_cases_csv(path):
    """One event per row; case fields repeat on each row of a case (first row wins). Rows with empty event_type add no event."""
    cases, order = {}, []
    with open(path, newline="", encoding="utf-8") as fh:
        for n, row in enumerate(csv.DictReader(fh), start=2):
            cid = (row.get("case_id") or "").strip()
            if not cid:
                raise ValueError("row %d: case_id missing" % n)
            if cid not in cases:
                cases[cid] = {"id": cid, "birth": row.get("birth"), "timezone": row.get("timezone"),
                              "longitude": row.get("longitude"), "accuracy": row.get("accuracy"),
                              "observed_until": row.get("observed_until"), "events": []}
                order.append(cid)
            if (row.get("event_type") or "").strip():
                cases[cid]["events"].append({"type": row["event_type"], "date": row["event_date"],
                                             "precision": row.get("precision"), "source": row.get("source")})
    return _check_unique([_mk_case(cases[c]) for c in order])


# ---------------------------------------------------------------- rules and rule sets
@dataclass(frozen=True)
class RuleSpec:
    rule_id: str               # only 'lord_sets' for now
    params: dict = field(default_factory=dict)
    label: str = ""


def _norm_params(rule_id, params):
    if rule_id != "lord_sets":
        raise ValueError("unknown rule_id %r (available: lord_sets)" % rule_id)
    level = int(params["level"])
    if not 1 <= level <= len(vim.LEVEL_NAMES):
        raise ValueError("level must be 1..6")
    cons = {}
    for k, lords in params["constraints"].items():
        k = int(k)
        if not 1 <= k <= level:
            raise ValueError("constraint level %d outside 1..%d" % (k, level))
        bad = [x for x in lords if x not in vim.LORDS]
        if bad or not lords:
            raise ValueError("bad lords %s at level %d" % (bad or "(empty)", k))
        cons[str(k)] = sorted(set(lords))
    mode = params.get("mode", "all")
    if mode not in ("all", "any") or not cons:
        raise ValueError("need constraints and mode 'all' or 'any'")
    return {"level": level, "constraints": cons, "mode": mode}


@dataclass(frozen=True)
class RuleSet:
    event_type: str
    rules: tuple                                  # of RuleSpec
    combine: str = "union"                        # 'union' | 'intersect' across rules
    ayanamsa: str = "LAHIRI"
    year_days: str = "julian"
    age_years: tuple = (0, 80)                    # observation window in years of age
    min_precision: str = "month"                  # drop events coarser than this
    split_seed: int = 0
    tune_fraction: float = 0.5
    null: str = "coverage"                        # headline null: 'coverage' | 'age_density' (both reported when possible)
    age_bandwidth_years: float = 2.0
    n_perm: int = 1000
    perm_seed: int = 0
    bin_years: float = 1.0                        # probability model: time bins of this length
    baseline: str = "age_density"                 # probability model baseline: 'age_density' | 'uniform'
    n_boot: int = 1000

    def canonical(self):
        return {"event_type": self.event_type,
                "rules": [{"rule_id": r.rule_id, "params": _norm_params(r.rule_id, r.params)} for r in self.rules],
                "combine": self.combine, "ayanamsa": self.ayanamsa, "year_days": str(self.year_days),
                "age_years": list(self.age_years), "min_precision": self.min_precision,
                "split": {"seed": self.split_seed, "tune_fraction": self.tune_fraction},
                "test": {"null": self.null, "age_bandwidth_years": self.age_bandwidth_years,
                         "n_perm": self.n_perm, "perm_seed": self.perm_seed,
                         "bin_years": self.bin_years, "baseline": self.baseline, "n_boot": self.n_boot},
                "harness_version": HARNESS_VERSION}

    def hash(self):
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------- preparing cases
@dataclass
class PEvent:
    type: str
    a: Fraction
    b: Fraction
    precision: str
    source: str = ""

    @property
    def length(self):
        return self.b - self.a


@dataclass
class Prepared:
    id: str
    moon_deg: Fraction
    obs: tuple                  # (lo, hi) days from birth
    events: list                # of PEvent
    accuracy: str = "?"
    warnings: list = field(default_factory=list)
    year_days: object = None    # resolved year option for this case (Fraction or SunArcMap); None = use the rule set's


def _days(a, b):
    d = b - a
    return Fraction(d.days) + Fraction(d.seconds, 86400) + Fraction(d.microseconds, 86400 * 10 ** 6)


def prepare(case, ruleset):
    birth = to_utc(datetime.fromisoformat(case.birth), tz_name=case.timezone or None,
                   longitude_deg=case.longitude)
    utc0 = birth["utc_used"]
    warnings = list(birth["warnings"])
    m = astro.moon(utc0, ruleset.ayanamsa)

    def t_of(local):
        return _days(utc0, to_utc(local, tz_name=case.timezone or None, longitude_deg=case.longitude)["utc_used"])

    lo = Fraction(ruleset.age_years[0]) * AGE_YEAR_DAYS
    hi = Fraction(ruleset.age_years[1]) * AGE_YEAR_DAYS
    if case.observed_until:
        ou = date.fromisoformat(case.observed_until)
        hi = min(hi, t_of(datetime(ou.year, ou.month, ou.day)))
    else:
        warnings.append("no observed_until: coverage measured over the whole age range")
    evs = []
    for e in case.events:
        a, b = event_interval(e)
        evs.append(PEvent(e.type, t_of(a), t_of(b), e.precision or _precision_of(e.date), e.source))
    return Prepared(case.id, Fraction(m["longitude_deg"]), (lo, hi), evs, case.accuracy, warnings,
                    solar.resolve_year_mode(ruleset.year_days, utc0))


def split_cases(cases, seed, tune_fraction):
    """Deterministic split by hash of (seed, id). Returns (tune, heldout)."""
    tune, held = [], []
    for c in cases:
        h = int(hashlib.sha256(("%s:%s" % (seed, c.id)).encode()).hexdigest()[:12], 16) / float(16 ** 12)
        (tune if h < tune_fraction else held).append(c)
    return tune, held


# ---------------------------------------------------------------- rule evaluation
def _rule_windows(spec, prep, ruleset, max_results=300000):
    p = _norm_params(spec.rule_id, spec.params)
    cons = {int(k): set(v) for k, v in p["constraints"].items()}
    agg = all if p["mode"] == "all" else any

    def pred(per):
        return agg(per.path[k - 1] in lords for k, lords in cons.items())

    lo, hi = prep.obs
    yd = prep.year_days if prep.year_days is not None else ruleset.year_days
    r = alg.find_periods(prep.moon_deg, pred, p["level"], lo, hi, yd, max_results)
    if r["truncated"]:
        raise ValueError("rule %s hit max_results=%d; use a coarser level" % (spec.label or spec.rule_id, max_results))
    return alg.merge_adjacent(r["periods"])


def case_windows(ruleset, prep):
    """Merged windows of the whole rule set, clipped to the case's observation window."""
    parts = [_rule_windows(r, prep, ruleset) for r in ruleset.rules]
    if ruleset.combine == "union":
        w = alg.union(*parts)
    elif ruleset.combine == "intersect":
        w = parts[0]
        for q in parts[1:]:
            w = alg.intersect(w, q)
    else:
        raise ValueError("combine must be union or intersect")
    lo, hi = prep.obs
    return alg.intersect(w, [alg.Window(lo, hi)])


def _overlaps(windows, ends, a, b):
    i = bisect_right(ends, a)                    # first window with end > a
    return i < len(windows) and windows[i].start < b


def _chance_windows(windows, half, lo, hi):
    """Windows widened by `half` both sides, merged, clipped to the valid midpoint range [lo+half, hi-half]."""
    a, b = lo + half, hi - half
    wide = alg.merge_adjacent([alg.Window(w.start - half, w.end + half) for w in windows])
    return alg.intersect(wide, [alg.Window(a, b)]), a, b


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=1.959964):
    if n == 0:
        return (None, None)
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    s = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def poisson_binomial_tail(ps, k):
    """(P(X >= k), P(X <= k)) for X = sum of independent Bernoulli(p_i). Exact DP; normal approximation above 2000 events."""
    n = len(ps)
    if n == 0:
        return (1.0 if k <= 0 else 0.0, 1.0)
    if n > 2000:
        mu = sum(ps)
        sd = math.sqrt(sum(p * (1 - p) for p in ps) or 1e-12)
        up = 0.5 * math.erfc((k - 0.5 - mu) / (sd * math.sqrt(2)))
        lo = 0.5 * math.erfc(-(k + 0.5 - mu) / (sd * math.sqrt(2)))
        return (min(1.0, up), min(1.0, lo))
    dist = [1.0] + [0.0] * n
    for i, p in enumerate(ps):
        for j in range(i + 1, 0, -1):
            dist[j] = dist[j] * (1 - p) + dist[j - 1] * p
        dist[0] *= (1 - p)
    return (min(1.0, sum(dist[k:])), min(1.0, sum(dist[:k + 1])))


def bonferroni(pvals):
    m = len(pvals)
    return [min(1.0, p * m) for p in pvals]


def bh_adjust(pvals):
    """Benjamini-Hochberg adjusted p-values (q-values), in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out, prev = [0.0] * m, 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, pvals[i] * m / rank)
        out[i] = prev
    return out


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


class _AgeDensity:
    """Pooled Gaussian-kernel CDF of event midpoint ages (years), leave-one-case-out."""
    STEP = 0.05

    def __init__(self, groups, hi_years, bandwidth):
        # groups: {case_id: [midpoint ages in years]}
        self.h = bandwidth
        self.x0 = -4 * bandwidth
        n = int((hi_years + 8 * bandwidth) / self.STEP) + 2
        self.xs = [self.x0 + i * self.STEP for i in range(n)]
        self.by_case = {cid: self._cdf(ages) for cid, ages in groups.items()}
        self.total = [sum(c[i] for c in self.by_case.values()) for i in range(n)]

    def _cdf(self, ages):
        return [sum(_phi((x - m) / self.h) for m in ages) for x in self.xs]

    def loo(self, cid):
        own = self.by_case.get(cid)
        return self.total if own is None else [t - o for t, o in zip(self.total, own)]

    def mass(self, G, a, b):
        def at(x):
            f = (x - self.x0) / self.STEP
            if f <= 0:
                return G[0]
            if f >= len(G) - 1:
                return G[-1]
            i = int(f)
            return G[i] + (G[i + 1] - G[i]) * (f - i)
        return at(b) - at(a)


# ---------------------------------------------------------------- scoring
def score(ruleset, prepared, include_events=False):
    """Score a rule set on prepared cases. Returns a report dict (no registry logic)."""
    if ruleset.min_precision not in PRECISIONS:
        raise ValueError("min_precision must be one of %s" % (PRECISIONS,))
    max_idx = PRECISIONS.index(ruleset.min_precision)
    excluded = {"wrong_type": 0, "precision_too_coarse": 0, "outside_observation_window": 0, "interval_longer_than_window": 0}
    caveats = ["Hit = event interval overlaps a window. Chance values assume the event could fall anywhere in the "
               "observation window (coverage null) or follow the other cases' age pattern (age-density null).",
               "A small number of events gives wide intervals; read hit_rate_ci95 and the p-values together."]
    wins, rows, per_case_cov = {}, [], {}
    for prep in prepared:
        lo, hi = prep.obs
        sel = []
        for e in prep.events:
            if e.type != ruleset.event_type:
                excluded["wrong_type"] += 1
            elif PRECISIONS.index(e.precision) > max_idx:
                excluded["precision_too_coarse"] += 1
            elif e.a < lo or e.b > hi:
                excluded["outside_observation_window"] += 1
            elif e.length >= hi - lo:
                excluded["interval_longer_than_window"] += 1
            else:
                sel.append(e)
        if not sel:
            continue
        w = case_windows(ruleset, prep)
        wins[prep.id] = w
        per_case_cov[prep.id] = alg.coverage_fraction(w, lo, hi)
        ends = [x.end for x in w]
        for e in sel:
            cw, a, b = _chance_windows(w, e.length / 2, lo, hi)
            p_cov = alg.coverage_fraction(cw, a, b)
            rows.append({"case": prep.id, "event": e, "hit": _overlaps(w, ends, e.a, e.b),
                         "p_cov": float(p_cov), "windows": w, "range": (a, b), "half": e.length / 2})
    n = len(rows)
    hits = sum(r["hit"] for r in rows)
    out = {"event_type": ruleset.event_type, "n_cases": len(wins), "n_events": n, "n_hits": hits,
           "excluded": excluded, "caveats": caveats}
    if n == 0:
        out["note"] = "no usable events"
        return out
    covs = [float(c) for c in per_case_cov.values()]
    out["hit_rate"] = hits / n
    out["hit_rate_ci95"] = list(wilson(hits, n))
    out["coverage"] = {"mean": sum(covs) / len(covs), "min": min(covs), "max": max(covs)}
    # coverage null
    ps = [r["p_cov"] for r in rows]
    exp = sum(ps)
    up, low = poisson_binomial_tail(ps, hits)
    out["coverage_null"] = {"chance_hit_rate": exp / n, "expected_hits": exp,
                            "lift": (hits / exp) if exp > 0 else None, "p_upper": up, "p_lower": low,
                            "approx": n > 2000}
    # age-density null
    groups = {}
    for r in rows:
        e = r["event"]
        groups.setdefault(r["case"], []).append(float((e.a + e.b) / 2 / AGE_YEAR_DAYS))
    ad = None
    if n - 1 >= 10 and len(groups) >= 3:
        hi_years = float(max(p.obs[1] for p in prepared) / AGE_YEAR_DAYS)
        dens = _AgeDensity(groups, hi_years, ruleset.age_bandwidth_years)
        loo = {cid: dens.loo(cid) for cid in groups}
        pa = []
        for r in rows:
            G = loo[r["case"]]
            a, b = r["range"]
            denom = dens.mass(G, float(a / AGE_YEAR_DAYS), float(b / AGE_YEAR_DAYS))
            cw = alg.intersect(alg.merge_adjacent([alg.Window(w.start - r["half"], w.end + r["half"]) for w in r["windows"]]),
                               [alg.Window(a, b)])
            num = sum(dens.mass(G, float(w.start / AGE_YEAR_DAYS), float(w.end / AGE_YEAR_DAYS)) for w in cw)
            pa.append(min(1.0, max(0.0, num / denom)) if denom > 1e-9 else None)
        if None not in pa:
            e2 = sum(pa)
            u2, l2 = poisson_binomial_tail(pa, hits)
            ad = {"chance_hit_rate": e2 / n, "expected_hits": e2, "lift": (hits / e2) if e2 > 0 else None,
                  "p_upper": u2, "p_lower": l2, "bandwidth_years": ruleset.age_bandwidth_years, "approx": n > 2000}
    out["age_density_null"] = ad
    if ad is None:
        out["age_density_note"] = "needs at least 10 other events from at least 3 cases"
    # chart-shuffle control
    ids = list(wins)
    if len(ids) >= 3 and ruleset.n_perm > 0:
        rng = random.Random(ruleset.perm_seed)
        ends_of = {cid: [x.end for x in wins[cid]] for cid in ids}
        by_case = {}
        for r in rows:
            by_case.setdefault(r["case"], []).append(r["event"])
        counts = []
        for _ in range(ruleset.n_perm):
            perm = ids[:]
            for _try in range(50):
                rng.shuffle(perm)
                if all(x != y for x, y in zip(ids, perm)):
                    break
            tot = 0
            for cid, src in zip(ids, perm):
                tot += sum(_overlaps(wins[src], ends_of[src], ev.a, ev.b) for ev in by_case[cid])
            counts.append(tot)
        ge = sum(c >= hits for c in counts)
        mean = sum(counts) / len(counts)
        out["shuffle_control"] = {"n_perm": ruleset.n_perm, "mean_hits_other_charts": mean,
                                  "sd": math.sqrt(sum((c - mean) ** 2 for c in counts) / len(counts)),
                                  "lift_vs_shuffle": (hits / mean) if mean > 0 else None,
                                  "p_upper": (1 + ge) / (1 + len(counts))}
    else:
        out["shuffle_control"] = None
    use_ad = ruleset.null == "age_density" and ad is not None
    head = ad if use_ad else out["coverage_null"]
    out["headline"] = {"null": "age_density" if use_ad else "coverage", "lift": head["lift"], "p_upper": head["p_upper"]}
    if include_events:
        out["events"] = [{"case": r["case"], "type": r["event"].type, "precision": r["event"].precision,
                          "hit": r["hit"], "chance": r["p_cov"]} for r in rows]
    return out


# ---------------------------------------------------------------- stacking guard
def stack_report(ruleset, prepared):
    """Add the rules one at a time (union): chance rate and lift at every step, plus the increment of each rule."""
    steps, prev = [], None
    for k in range(1, len(ruleset.rules) + 1):
        sub = RuleSet(**{**ruleset.__dict__, "rules": ruleset.rules[:k], "combine": "union", "n_perm": 0})
        r = score(sub, prepared)
        if not r.get("n_events"):
            steps.append({"rules": k, "note": "no usable events"})
            continue
        row = {"rules": k, "last_rule": ruleset.rules[k - 1].label or ruleset.rules[k - 1].rule_id,
               "hit_rate": r["hit_rate"], "chance_hit_rate": r["coverage_null"]["chance_hit_rate"],
               "lift": r["coverage_null"]["lift"], "mean_coverage": r["coverage"]["mean"],
               "p_upper": r["coverage_null"]["p_upper"]}
        if prev:
            row["added_hit_rate"] = row["hit_rate"] - prev["hit_rate"]
            row["added_chance"] = row["chance_hit_rate"] - prev["chance_hit_rate"]
            row["added_lift_of_increment"] = (row["added_hit_rate"] / row["added_chance"]) if row["added_chance"] > 1e-12 else None
        steps.append(row)
        prev = row
    return {"steps": steps,
            "note": "A rule that raises the hit rate by about as much as it raises the chance rate adds no information "
                    "(increment lift near 1). Stacking rules always raises the hit rate."}


# ---------------------------------------------------------------- pre-registration
def _load_registry(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {"version": 1, "entries": {}, "heldout_looks_total": 0}


def _save_registry(path, reg):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(reg, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def register(ruleset, registry_path, note=""):
    """Freeze the rule set (rules, options, split, test settings) by hash. Idempotent."""
    reg = _load_registry(registry_path)
    h = ruleset.hash()
    if h not in reg["entries"]:
        reg["entries"][h] = {"registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "note": note, "canonical": ruleset.canonical(), "looks": {"tune": 0, "heldout": 0}}
        _save_registry(registry_path, reg)
    return h


def evaluate(ruleset, cases, which, registry_path, include_events=False, with_probability=False, with_matching=False):
    """Score on the 'tune' or 'heldout' part of the split. Held-out needs a registered hash; every look is counted."""
    if which not in ("tune", "heldout"):
        raise ValueError("which must be 'tune' or 'heldout'")
    reg = _load_registry(registry_path)
    h = ruleset.hash()
    entry = reg["entries"].get(h)
    if which == "heldout" and entry is None:
        raise ValueError("held-out evaluation refused: rule set hash %s is not pre-registered (the rule set, options, split "
                         "or test settings changed since registration, or it was never registered)" % h[:12])
    tune, held = split_cases(cases, ruleset.split_seed, ruleset.tune_fraction)
    subset = tune if which == "tune" else held
    prepared = [prepare(c, ruleset) for c in subset]
    rep = score(ruleset, prepared, include_events)
    if with_probability:
        fit_prep = prepared if which == "tune" else [prepare(c, ruleset) for c in tune]
        rep["probability"] = probability_report(ruleset, fit_prep, prepared, in_sample=(which == "tune"))
    if with_matching:
        fit_prep = prepared if which == "tune" else [prepare(c, ruleset) for c in tune]
        rep["matching"] = matching_test(ruleset, fit_prep, prepared, in_sample=(which == "tune"))
    reg_block = {"hash": h, "pre_registered": entry is not None, "split": which,
                 "n_cases_in_split": len(subset), "n_tune": len(tune), "n_heldout": len(held)}
    if entry is not None:
        entry["looks"][which] += 1
        if which == "heldout":
            reg["heldout_looks_total"] += 1
        _save_registry(registry_path, reg)
        reg_block.update({"registered_at": entry["registered_at"], "looks_this_hash": dict(entry["looks"]),
                          "heldout_looks_total_all_hashes": reg["heldout_looks_total"]})
        if which == "heldout" and (entry["looks"]["heldout"] > 1 or reg["heldout_looks_total"] > 1):
            reg_block["warning"] = ("held-out cases have been looked at more than once; treat p-values as optimistic "
                                    "and apply a multiplicity correction across all looks")
    else:
        reg_block["warning"] = "not pre-registered: exploratory result, do not report as a test"
    rep["registration"] = reg_block
    rep["provenance"] = {"harness_version": HARNESS_VERSION, "ruleset_hash": h, "canonical": ruleset.canonical()}
    return rep


# ---------------------------------------------------------------- chance of a hit for one rule (no cases needed)
def chance_probability(windows, lo, hi, length):
    """Exact chance that an event interval of `length` days, placed uniformly at random in [lo, hi), overlaps the windows."""
    lo, hi, length = Fraction(lo), Fraction(hi), Fraction(length)
    if length >= hi - lo:
        raise ValueError("interval is not shorter than the observation window")
    cw, a, b = _chance_windows(windows, length / 2, lo, hi)
    return alg.coverage_fraction(cw, a, b)


# ---------------------------------------------------------------- probability scoring (log score, Brier, calibration)
def _usable_events(ruleset, prep, max_len):
    lo, hi = prep.obs
    idx = PRECISIONS.index(ruleset.min_precision)
    return [e for e in prep.events if e.type == ruleset.event_type and PRECISIONS.index(e.precision) <= idx
            and lo <= e.a and e.b <= hi and e.length <= max_len]


def _cover(windows, ends, a, b):
    i, tot = bisect_right(ends, a), Fraction(0)
    while i < len(windows) and windows[i].start < b:
        tot += min(b, windows[i].end) - max(a, windows[i].start)
        i += 1
    return tot / (b - a)


def _forecasts(ruleset, prepared, dens, in_sample, bin_days):
    """Per case: base weights, window shares per bin and the bins holding its events (floats)."""
    out = []
    for prep in prepared:
        lo, hi = prep.obs
        evs = _usable_events(ruleset, prep, bin_days)
        if not evs:
            continue
        n = int(math.ceil((hi - lo) / bin_days))
        if n < 2:
            continue
        w = case_windows(ruleset, prep)
        ends = [x.end for x in w]
        G = None
        if dens is not None:
            G = dens.loo(prep.id) if in_sample and prep.id in dens.by_case else dens.total
        base, f = [], []
        for k in range(n):
            a = lo + k * bin_days
            b = min(hi, a + bin_days)
            base.append(dens.mass(G, float(a / AGE_YEAR_DAYS), float(b / AGE_YEAR_DAYS)) if G is not None else float((b - a) / (hi - lo)))
            f.append(float(_cover(w, ends, a, b)))
        base = [max(x, 1e-12) for x in base]
        idxs = [min(n - 1, int(((e.a + e.b) / 2 - lo) // bin_days)) for e in evs]
        B = sum(base)
        S = sum(x * y for x, y in zip(base, f))
        out.append({"id": prep.id, "base": base, "f": f, "idx": idxs, "B": B, "m": S / B})
    return out


def _ll(fcs, r):
    tot = 0.0
    for c in fcs:
        z = 1 + (r - 1) * c["m"]
        for i in c["idx"]:
            tot += math.log(1 + (r - 1) * c["f"][i]) - math.log(z)
    return tot


def _fit_r(fcs):
    """Maximum-likelihood r in [1e-3, 1e3] (log scale grid, then golden-section). Never worse than r = 1."""
    grid = [math.exp(x / 4.0) for x in range(-28, 29)]
    best = max(grid, key=lambda r: _ll(fcs, r))
    lo, hi = math.log(best) - 0.25, math.log(best) + 0.25
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = _ll(fcs, math.exp(a)), _ll(fcs, math.exp(b))
    for _ in range(40):
        if fa < fb:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = _ll(fcs, math.exp(b))
        else:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = _ll(fcs, math.exp(a))
    r = math.exp((lo + hi) / 2)
    return r if _ll(fcs, r) >= _ll(fcs, 1.0) else 1.0


def probability_report(ruleset, fit_prepared, test_prepared, in_sample=False, calibration_groups=8):
    """Forecast = distribution over time bins for where a person's event falls. r is fitted on `fit_prepared`,
    scored on `test_prepared`. Compared with the baseline (r = 1: uniform or age-density, no chart information)."""
    bin_days = Fraction(ruleset.bin_years) * AGE_YEAR_DAYS
    dens, base_used = None, "uniform"
    if ruleset.baseline == "age_density":
        groups = {}
        for prep in fit_prepared:
            for e in _usable_events(ruleset, prep, bin_days):
                groups.setdefault(prep.id, []).append(float((e.a + e.b) / 2 / AGE_YEAR_DAYS))
        if sum(len(v) for v in groups.values()) >= 10 and len(groups) >= 3:
            hi_years = float(max(p.obs[1] for p in list(fit_prepared) + list(test_prepared)) / AGE_YEAR_DAYS)
            dens, base_used = _AgeDensity(groups, hi_years, ruleset.age_bandwidth_years), "age_density"
    elif ruleset.baseline != "uniform":
        raise ValueError("baseline must be 'age_density' or 'uniform'")
    fit = _forecasts(ruleset, fit_prepared, dens, True, bin_days)
    test = _forecasts(ruleset, test_prepared, dens, in_sample, bin_days)
    n_ev = sum(len(c["idx"]) for c in test)
    out = {"bin_years": ruleset.bin_years, "baseline_requested": ruleset.baseline, "baseline_used": base_used,
           "in_sample": in_sample, "n_cases_fit": len(fit), "n_cases_scored": len(test), "n_events_scored": n_ev}
    if not fit or n_ev == 0:
        out["note"] = "not enough usable events to fit and score"
        return out
    r = _fit_r(fit)
    out["r_fitted"] = r
    out["r_meaning"] = ("a bin fully inside a rule window is r times as likely as the same bin outside "
                        "(r = 1: the rule carries no information; r < 1: events avoid the windows)")
    d_ll, brier_r, brier_b, pairs = [], 0.0, 0.0, []
    for c in test:
        z = 1 + (r - 1) * c["m"]
        n = len(c["base"])
        p = [b * (1 + (r - 1) * f) for b, f in zip(c["base"], c["f"])]
        tot = sum(p)
        p = [x / tot for x in p]
        q = [b / c["B"] for b in c["base"]]
        for i in c["idx"]:
            d_ll.append(math.log(p[i]) - math.log(q[i]))
            brier_r += sum((p[k] - (k == i)) ** 2 for k in range(n))
            brier_b += sum((q[k] - (k == i)) ** 2 for k in range(n))
            pairs.append((p, i))
    mean = sum(d_ll) / len(d_ll)
    rng = random.Random(ruleset.perm_seed)
    boots = sorted(sum(rng.choice(d_ll) for _ in d_ll) / len(d_ll) for _ in range(max(1, ruleset.n_boot)))
    ci = (boots[int(0.025 * (len(boots) - 1))], boots[int(0.975 * (len(boots) - 1))])
    out["log_score_gain_per_event_nats"] = mean
    out["log_score_gain_ci95"] = list(ci)
    out["probability_ratio_geometric_mean"] = math.exp(mean)
    out["brier_rule"] = brier_r / len(d_ll)
    out["brier_baseline"] = brier_b / len(d_ll)
    out["brier_skill_score"] = 1 - brier_r / brier_b if brier_b > 0 else None
    # calibration: pool (forecast probability, did the event fall in this bin) over all bins of all scored events
    flat = []
    for c in test:
        n = len(c["base"])
        p = [b * (1 + (r - 1) * f) for b, f in zip(c["base"], c["f"])]
        tot = sum(p)
        for k in range(n):
            flat.append((p[k] / tot, sum(1 for i in c["idx"] if i == k)))
    flat.sort()
    g = max(2, calibration_groups)
    table = []
    for j in range(g):
        part = flat[j * len(flat) // g:(j + 1) * len(flat) // g]
        if part:
            table.append({"mean_forecast": sum(x for x, _ in part) / len(part),
                          "observed_rate": sum(y for _, y in part) / len(part), "n_bins": len(part)})
    out["calibration"] = table
    out["verdict_hint"] = ("in-sample: r was fitted on these cases, so a gain is expected and proves nothing" if in_sample
                           else "held-out: a gain whose CI excludes 0 is evidence that the chart information improves the forecast")
    out["caveats"] = ["Events of one person that share a forecast are not independent; the bootstrap treats them as independent.",
                      "Calibration pools many bins with rare events; read it loosely."]
    return out


# ---------------------------------------------------------------- chart-matching test (Carlson style)
def matching_test(ruleset, fit_prepared, test_prepared, in_sample=False, top_fraction=0.1):
    """Can the rule pair charts with event histories better than chance?

    One event per person (the first usable one). Charts and event histories are shuffled independently and the scoring
    code only ever sees the two shuffled lists (no ids, no pairing); the true pairing is used only afterwards to read off
    ranks. A chart's score for an event is the likelihood ratio its forecast gives to the event's time bin,
    (1 + (r-1) x share of that bin inside the chart's windows) / (1 + (r-1) x baseline share of the chart's windows),
    with r fitted on the fit split. An event outside a chart's observation window gives that chart a neutral score (1).
    """
    bin_days = Fraction(ruleset.bin_years) * AGE_YEAR_DAYS
    rng = random.Random(ruleset.perm_seed)
    dens = None
    if ruleset.baseline == "age_density":
        groups = {}
        for prep in fit_prepared:
            for e in _usable_events(ruleset, prep, bin_days):
                groups.setdefault(prep.id, []).append(float((e.a + e.b) / 2 / AGE_YEAR_DAYS))
        if sum(len(v) for v in groups.values()) >= 10 and len(groups) >= 3:
            hi_years = float(max(p.obs[1] for p in list(fit_prepared) + list(test_prepared)) / AGE_YEAR_DAYS)
            dens = _AgeDensity(groups, hi_years, ruleset.age_bandwidth_years)
    fit = _forecasts(ruleset, fit_prepared, dens, True, bin_days)
    r = _fit_r(fit) if fit else 1.0
    people = [p for p in test_prepared if _usable_events(ruleset, p, bin_days)]
    n = len(people)
    out = {"n_pairs": n, "r_fitted": r, "in_sample": in_sample}
    if n < 5:
        out["note"] = "need at least 5 people with a usable event"
        return out
    fc = {f["id"]: f for f in _forecasts(ruleset, people, dens, in_sample, bin_days)}
    charts, events, truth = [], [], {}
    order_c, order_e = list(range(n)), list(range(n))
    rng.shuffle(order_c)
    rng.shuffle(order_e)
    for pos, k in enumerate(order_c):
        prep = people[k]
        w = case_windows(ruleset, prep)
        charts.append({"lo": prep.obs[0], "hi": prep.obs[1], "w": w, "ends": [x.end for x in w], "m": fc[prep.id]["m"]})
        truth[("c", k)] = pos
    for pos, k in enumerate(order_e):
        e = _usable_events(ruleset, people[k], bin_days)[0]
        events.append((e.a + e.b) / 2)
        truth[("e", k)] = pos
    # ---- blind scoring: only `charts` and `events` (shuffled, no ids) are used
    S = [[1.0] * n for _ in range(n)]
    for i, c in enumerate(charts):
        for j, mid in enumerate(events):
            if c["lo"] <= mid < c["hi"]:
                k = (mid - c["lo"]) // bin_days
                a = c["lo"] + k * bin_days
                b = min(c["hi"], a + bin_days)
                f = float(_cover(c["w"], c["ends"], a, b))
                S[i][j] = (1 + (r - 1) * f) / (1 + (r - 1) * c["m"])
    # ---- only now the true pairing is used
    true_chart_of_event = [truth[("c", k)] for k in order_e]          # event position j -> chart position
    P = [[0.0] * n for _ in range(n)]        # percentile of chart i among all charts for event j (1 = best)
    for j in range(n):
        col = [S[i][j] for i in range(n)]
        for i in range(n):
            better = sum(1 for x in col if x > col[i] + 1e-12)
            tied = sum(1 for x in col if abs(x - col[i]) <= 1e-12) - 1
            P[i][j] = 1 - (better + 0.5 * tied) / (n - 1)
    pct = [P[true_chart_of_event[j]][j] for j in range(n)]
    k_top = max(1, int(round(top_fraction * n)))
    ranks = [round((1 - p) * (n - 1)) + 1 for p in pct]
    obs_pct = sum(pct) / n
    obs_top = sum(1 for rk in ranks if rk <= k_top)
    out.update({"top_k": k_top, "top_k_hits": obs_top, "top_k_expected_by_chance": n * k_top / n,
                "mean_percentile_of_true_chart": obs_pct, "chance_mean_percentile": 0.5,
                "median_rank": sorted(ranks)[n // 2], "n_charts": n})
    if ruleset.n_perm > 0:
        ge_p = ge_t = 0
        idx = list(range(n))
        for _ in range(ruleset.n_perm):
            rng.shuffle(idx)
            vals = [P[idx[j]][j] for j in range(n)]
            ge_p += (sum(vals) / n) >= obs_pct - 1e-12
            ge_t += sum(1 for j in range(n) if round((1 - vals[j]) * (n - 1)) + 1 <= k_top) >= obs_top
        out["p_upper_mean_percentile"] = (1 + ge_p) / (1 + ruleset.n_perm)
        out["p_upper_top_k"] = (1 + ge_t) / (1 + ruleset.n_perm)
    out["note"] = ("Blind by construction: scoring sees shuffled charts and shuffled event ages only. "
                   + ("In-sample: r was fitted on these people, so a pass proves nothing." if in_sample else
                      "Held-out: r was fitted on other people."))
    return out
