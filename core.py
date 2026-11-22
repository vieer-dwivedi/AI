"""
core.py - deterministic engine for the Resume Studio MCP.
NO LLM calls anywhere in this project: storage, validation, JD matching, rendering and form
handling are plain code. The *client* LLM decides what to write; this code makes sure it is
consistent, truthful-to-master, well formatted and recoverable.
"""
import collections, copy, datetime as dt, json, os, pathlib, re, shutil

import fnmatch
import store

BASE = pathlib.Path(__file__).parent

# ----------------------------------------------------------------------------- utils
def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-") or "x"

def clean(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()

NUM_RE = re.compile(r"(?<![A-Za-z0-9.])\d+(?:\.\d+)?(?![A-Za-z0-9])")
YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\b", re.I)

def numbers_in(text: str) -> set:
    return set(NUM_RE.findall(text or ""))

# ----------------------------------------------------------------------------- storage (see store.py: files locally, Redis on Vercel)
def _key(variant: str) -> str:
    v = slug(variant)
    return "master" if v == "master" else f"variant/{v}"

def exists(variant: str) -> bool:
    return store.exists(_key(variant))

def mtime(variant: str) -> float:
    return store.mtime(_key(variant))

def list_variants() -> list:
    return ["master"] + sorted(k.split("/", 1)[1] for k in store.keys("variant/"))

def load(variant: str = "master") -> dict:
    d = store.get_json(_key(variant))
    if d is None:
        raise ValueError(f"Variant '{variant}' does not exist. Existing: {list_variants()}. Use create_variant first.")
    meta = d.setdefault("_meta", {})
    ap = meta.setdefault("approved", {})
    for k in ("skills", "numbers", "companies"):
        ap.setdefault(k, [])
    for j in d.get("experience", []):
        j.setdefault("id", slug(j.get("company", "job")))
    return d

def save(variant: str, data: dict, note: str = "") -> None:
    key, v = _key(variant), slug(variant)
    old = store.get_json(key)
    if old is not None:
        store.set_json(f"hist/{v}/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f"), old)
        for k in store.keys(f"hist/{v}/")[:-30]:
            store.delete(k)
    data.setdefault("_meta", {})["updated"] = dt.datetime.now().isoformat(timespec="seconds")
    if note:
        data["_meta"]["last_change"] = note
    store.set_json(key, data)

def history(variant: str) -> list:
    return [k.rsplit("/", 1)[1] for k in sorted(store.keys(f"hist/{slug(variant)}/"), reverse=True)]

def undo(variant: str) -> str:
    snaps = sorted(store.keys(f"hist/{slug(variant)}/"))
    if not snaps:
        raise ValueError("No history to undo for this variant.")
    last = snaps[-1]
    store.set_json(_key(variant), store.get_json(last))
    store.delete(last)
    return last.rsplit("/", 1)[1]

def delete_variant(name: str) -> None:
    store.delete(_key(name))

# generated files (PDF, .eml, screenshots)
def out_write(name: str, data: bytes) -> str:
    name = os.path.basename(name)
    store.set_bytes("out/" + name, data)
    return name

def out_read(name: str):
    return store.get_bytes("out/" + os.path.basename(name))

def out_exists(name: str) -> bool:
    return store.exists("out/" + os.path.basename(name))

def out_mtime(name: str) -> float:
    return store.mtime("out/" + os.path.basename(name))

def out_list(pattern: str) -> list:
    return [k.split("/", 1)[1] for k in store.keys("out/", pattern)]

def find_job(data: dict, ref) -> dict:
    jobs = data.get("experience", [])
    r = str(ref).strip().lower()
    if r.isdigit() and int(r) < len(jobs):
        return jobs[int(r)]
    for j in jobs:
        if r in (j["id"].lower(), j["company"].lower()):
            return j
    hits = [j for j in jobs if r in j["company"].lower() or r in j["title"].lower()]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f"Job '{ref}' not found or ambiguous. Available: {[(i, j['id']) for i, j in enumerate(jobs)]}")

def contact_line(d: dict) -> str:
    c = d.get("contact", {})
    if isinstance(c, str):
        return c
    return " | ".join(x for x in [c.get("email"), c.get("phone"), c.get("location"), *c.get("links", [])] if x)

# ----------------------------------------------------------------------------- dates
MONTHS = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}
DATE_RE = re.compile(r"^\s*([A-Za-z]{3,9})\.?\s+(\d{4})\s*[–—-]\s*(present|current|[A-Za-z]{3,9}\.?\s+\d{4})\s*$", re.I)

def parse_range(s: str, today: dt.date = None):
    m = DATE_RE.match(s or "")
    if not m:
        return None
    sm, sy = MONTHS.get(m[1][:3].lower()), int(m[2])
    if not sm:
        return None
    e = m[3].strip()
    if e.lower() in ("present", "current"):
        t = today or dt.date.today()
        em, ey = t.month, t.year
    else:
        mm = re.match(r"([A-Za-z]{3,9})\.?\s+(\d{4})", e)
        em, ey = MONTHS.get(mm[1][:3].lower()), int(mm[2])
        if not em:
            return None
    return sy * 12 + sm, ey * 12 + em

def experience_years(d: dict) -> float:
    iv = sorted(r for j in d.get("experience", []) if (r := parse_range(j.get("dates", ""))))
    total, cur_s, cur_e = 0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return round(total / 12, 1)

# ----------------------------------------------------------------------------- master pool
def pool():
    m = load("master")
    skills = {s.lower() for v in m["skills"].values() for s in v}
    blob = json.dumps([m["experience"], m.get("certifications", []), m.get("education", []), m.get("awards", [])])
    return skills, numbers_in(blob), {j["company"].lower() for j in m["experience"]}, m

def flat_skills(d: dict) -> list:
    return [s for v in d.get("skills", {}).values() for s in v]

# ----------------------------------------------------------------------------- validation
PLACEHOLDER = re.compile(r"\b(TODO|TBD|FIXME|XXX|lorem ipsum)\b|\[[^\]]*\]|\{\{|\}\}", re.I)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")

def _strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for k, v in o.items():
            if k != "_meta":
                yield from _strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _strings(v)

def validate(variant: str = "master") -> dict:
    d = load(variant)
    is_master = slug(variant) == "master"
    E, W = [], []
    skp, nump, compp, m = pool()
    ap = d["_meta"]["approved"]
    ok_sk = {s.lower() for s in ap["skills"]}
    ok_nu, ok_co = set(ap["numbers"]), {c.lower() for c in ap["companies"]}
    c = d.get("contact", {})
    email = c.get("email", "") if isinstance(c, dict) else ""
    if not clean(d.get("name")): E.append("name is empty")
    if not EMAIL_RE.match(email): E.append(f"contact.email invalid: '{email}'")
    if not clean(c.get("phone", "") if isinstance(c, dict) else ""): W.append("contact.phone is empty")
    if not clean(d.get("headline")): E.append("headline is empty")
    summ = clean(d.get("summary"))
    if not summ: E.append("summary is empty")
    elif len(summ) > 500: W.append(f"summary is {len(summ)} chars; keep it <= 500 (2-3 sentences)")
    elif len(summ) < 60: W.append("summary is very short")
    for s in _strings(d):
        if PLACEHOLDER.search(s): E.append(f"placeholder/bracket text found: '{s[:60]}'")

    # experience
    jobs = d.get("experience", [])
    if not jobs: E.append("no experience entries")
    spans = []
    for j in jobs:
        tag = f"[{j.get('company', '?')}]"
        for f in ("title", "company", "dates"):
            if not clean(j.get(f)): E.append(f"{tag} {f} is empty")
        rng = parse_range(j.get("dates", ""))
        if not rng: E.append(f"{tag} dates '{j.get('dates')}' must look like 'Apr 2023 – Present' or 'Sep 2022 – Apr 2023'")
        else:
            spans.append((rng, j))
            if rng[1] < rng[0]: E.append(f"{tag} end date is before start date")
        bl = j.get("bullets", [])
        if not bl: W.append(f"{tag} has no bullets")
        if len(bl) > 10: W.append(f"{tag} has {len(bl)} bullets; recruiters skim - prefer <= 8")
        seen = set()
        for i, b in enumerate(bl):
            if not clean(b): E.append(f"{tag} bullet {i} is empty")
            elif len(b) > 320: W.append(f"{tag} bullet {i} is {len(b)} chars (> 320); split or tighten it")
            if clean(b).lower() in seen: E.append(f"{tag} bullet {i} duplicates another bullet")
            seen.add(clean(b).lower())
        if not is_master:
            mj = next((x for x in m["experience"] if x["company"].lower() == j.get("company", "").lower()), None)
            if not mj and j.get("company", "").lower() not in ok_co:
                E.append(f"{tag} company is not in master and not approved - never invent employers")
            if mj:
                if clean(mj["dates"]) != clean(j.get("dates")): E.append(f"{tag} dates differ from master ('{mj['dates']}') - dates are immutable in variants")
                if clean(mj["title"]) != clean(j.get("title")): W.append(f"{tag} title differs from master ('{mj['title']}'); keep real titles, put targeting in the headline")
    starts = [r[0][0] for r in spans]
    if starts != sorted(starts, reverse=True): W.append("experience is not in reverse-chronological order")
    for a in range(len(spans)):
        for b in range(a + 1, len(spans)):
            if spans[a][0][0] < spans[b][0][1] and spans[b][0][0] < spans[a][0][1]:
                W.append(f"jobs overlap in time: {spans[a][1]['company']} / {spans[b][1]['company']}")

    # skills
    sk = d.get("skills", {})
    allsk = flat_skills(d)
    if not allsk: E.append("no skills")
    low = [s.lower() for s in allsk]
    dup = sorted({s for s in low if low.count(s) > 1})
    if dup: W.append(f"duplicate skills: {dup}")
    if len(allsk) > 90: W.append(f"{len(allsk)} skills listed; > 90 reads as keyword stuffing - trim to what the JD needs")
    for k, v in sk.items():
        if not clean(k): E.append("a skill category has an empty name")
        if not v: E.append(f"skill category '{k}' is empty")
    if not is_master:
        unv = sorted({s for s in allsk if s.lower() not in skp and s.lower() not in ok_sk})
        if unv: E.append(f"skills not in master and not approved (possible fabrication): {unv}")

    # numbers + years claims
    exp_years = experience_years(d)
    texts = [("summary", summ), ("headline", clean(d.get("headline")))] + [(j.get("company", "?"), b) for j in jobs for b in j.get("bullets", [])]
    for where, t in texts:
        for y in YEARS_RE.findall(t):
            if float(y) > exp_years + 0.75:
                E.append(f"[{where}] claims {y}+ years but dated experience totals {exp_years} years - fix the claim or add the missing role")
        t2 = YEARS_RE.sub(" ", t)
        if not is_master:
            bad = sorted(numbers_in(t2) - nump - ok_nu)
            if bad: E.append(f"[{where}] numbers not found in master (possible invented metric): {bad}")

    # education / certs / overlap
    if not d.get("education"): W.append("education is empty")
    if not is_master:
        for e in d.get("education", []):
            if e not in m.get("education", []): W.append(f"education entry not in master - confirm it is real: {e.get('degree')}")
        for ce in d.get("certifications", []):
            if ce not in m.get("certifications", []): W.append(f"certification not in master - confirm it is real: {ce}")
    for e in d.get("education", []):
        r = parse_range(e.get("dates", ""))
        if r:
            for (s, en), j in spans:
                if s < r[1] and r[0] < en:
                    W.append(f"education '{e.get('degree')}' overlaps full-time role at {j['company']} - label it part-time/distance if true")
                    break
    return {"errors": E, "warnings": W, "info": {
        "variant": variant, "skills": len(allsk), "experience_years": exp_years,
        "bullets": {j.get("company"): len(j.get("bullets", [])) for j in jobs}}}

# ----------------------------------------------------------------------------- JD matching
EXCLUDE = {"kind", "vector", "alloy", "detective", "secrets", "deployments", "inspector", "consul", "bastion"}
ALIASES = {"kubernetes (k8s)": ["k8s"], "argo cd": ["argocd"], "opentelemetry": ["otel"],
           "github actions": ["gh actions"], "azure devops": ["ado"], "terraform": ["tf"],
           "continuous integration (ci)": ["continuous integration"], "continuous delivery (cd)": ["continuous delivery"]}
EXTRA = """gcp|google cloud|pulumi|hashicorp vault|vault|packer|nomad|crossplane|backstage|cilium|calico|envoy|kafka|rabbitmq|redis|postgresql|
postgres|mysql|mongodb|java|golang|rust|node.js|javascript|typescript|react|.net|c#|scala|ruby|php|puppet|chef|saltstack|octopus deploy|airflow|
mlflow|kubeflow|langchain|langgraph|pytorch|tensorflow|snowflake|databricks|dbt|spark|hadoop|windows server|powershell|vmware|openstack|cloudflare|
akamai|fastly|nginx|apache|haproxy|tomcat|jira|confluence|servicenow|sre|site reliability|finops|soc 2|iso 27001|pci dss|hipaa|gdpr|itil|agile|scrum|
owasp|wiz|prisma cloud|falco|opa|gatekeeper|kyverno|sentry|mimir|tempo|cortex|statsd|bazel|maven|gradle|artifactory|nexus|sonatype|argo rollouts|
flagger|linkerd2|keda|velero|longhorn|ceph|oci|oracle cloud|alibaba cloud|digitalocean|heroku|vercel|netlify|openai|llm|rag|mlops|aiops|dora""".replace("\n", "").split("|")

def _variants(skill: str) -> list:
    out = {skill.lower()}
    m = re.match(r"^(.*?)\s*\((.*?)\)\s*$", skill)
    if m:
        out.add(m[1].strip().lower())
        out |= {x.strip().lower() for x in m[2].split(",") if x.strip()}
    mm = re.match(r"^(AWS|Amazon|Azure|Google)\s+(.+)$", skill)
    if mm and ((mm[2].isupper() and len(mm[2]) >= 3) or mm[2] in {"Lambda", "Fargate", "Bedrock", "SageMaker"}):
        out.add(mm[2].lower())
    out |= set(ALIASES.get(skill.lower(), []))
    return [v for v in out if v]

def _pat(term: str):
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(term) + r"(?![A-Za-z0-9_])", re.I)

def lexicon(master: dict):
    """list of (canonical, category or None, [patterns]) - master skills first, then known-but-not-in-master tech."""
    lex, seen = [], set()
    for cat, items in master["skills"].items():
        for s in items:
            if s.lower() in EXCLUDE or s.lower() in seen:
                continue
            seen.add(s.lower())
            lex.append((s, cat, [_pat(v) for v in _variants(s)]))
    mv = {v for _, _, ps in lex for v in [p.pattern for p in ps]}
    for t in EXTRA:
        t = t.strip()
        if t and t not in seen and _pat(t).pattern not in mv:
            lex.append((t, None, [_pat(t)]))
    return lex

def count_terms(text: str, lex) -> collections.OrderedDict:
    res = {}
    for canon, cat, pats in lex:
        n = sum(len(p.findall(text)) for p in pats)
        if n:
            res[canon] = (n, cat)
    return collections.OrderedDict(sorted(res.items(), key=lambda kv: -kv[1][0]))

def analyze_jd(jd_text: str, variant: str = "master") -> dict:
    d = load(variant)
    _, _, _, m = pool()
    lex = lexicon(m)
    found = count_terms(jd_text, lex)
    have = {s.lower() for s in flat_skills(d)}
    matched = [{"skill": k, "mentions": n, "category": c} for k, (n, c) in found.items() if c and k.lower() in have]
    avail = [{"skill": k, "mentions": n, "category": c} for k, (n, c) in found.items() if c and k.lower() not in have]
    gaps = [{"term": k, "mentions": n} for k, (n, c) in found.items() if c is None]
    total = len(matched) + len(avail) + len(gaps)
    titles = [clean(l)[:100] for l in jd_text.splitlines()
              if re.search(r"\b(engineer|developer|architect|sre|manager|lead|specialist|administrator|consultant)\b", l, re.I) and len(l) < 120][:5]
    reqy = [{"years": float(y), "context": clean(jd_text[max(0, mm.start() - 50):mm.end() + 60])}
            for mm in YEARS_RE.finditer(jd_text) for y in [mm.group(1)]][:6]
    return {"title_candidates": titles, "required_years_mentions": reqy,
            "your_experience_years": experience_years(d),
            "matched_in_variant": matched, "in_master_but_not_in_variant": avail,
            "gaps_not_in_master": gaps,
            "coverage_pct": round(100 * len(matched) / total) if total else None,
            "guidance": "Add 'in_master_but_not_in_variant' freely. NEVER add 'gaps_not_in_master' unless the user confirms real experience (then use allow_new=True)."}

def build_skills_for_jd(jd_text: str, max_skills: int = 40, min_skills: int = 20) -> dict:
    _, _, _, m = pool()
    found = count_terms(jd_text, lexicon(m))
    hits = [(k, n, c) for k, (n, c) in found.items() if c]
    chosen = [k for k, _, _ in hits][:max_skills]
    cat_hits = collections.Counter()
    for k, n, c in hits:
        cat_hits[c] += n
    order = sorted(m["skills"], key=lambda c: (-cat_hits[c], list(m["skills"]).index(c)))
    if len(chosen) < min_skills:  # pad from the best-matching categories, in master order
        for c in order:
            for s in m["skills"][c]:
                if len(chosen) >= min_skills: break
                if s not in chosen: chosen.append(s)
    cs = set(chosen)
    out = {}
    for c in order:
        items = [s for s in m["skills"][c] if s in cs]
        if items:
            out[c] = items
    return out

def merge_skills_for_jd(cur: dict, jd_text: str) -> dict:
    """NEVER removes a skill: union of the variant's skills and master's, JD-mentioned skills first inside each category, best-matching categories first."""
    _, _, _, m = pool()
    hit = {k.lower(): n for k, (n, c) in count_terms(jd_text, lexicon(m)).items() if c}
    cats = list(dict.fromkeys(list(cur) + list(m["skills"])))
    out = {}
    for c in cats:
        items = list(dict.fromkeys(list(cur.get(c, [])) + list(m["skills"].get(c, []))))
        out[c] = sorted(items, key=lambda x: -hit.get(x.lower(), 0))
    return dict(sorted(out.items(), key=lambda kv: -sum(hit.get(x.lower(), 0) for x in kv[1])))

def sync_experience_for_jd(variant: str, jd_text: str) -> dict:
    """Restore/add master bullets that evidence JD skills (never removes or rewrites), then order bullets by JD relevance."""
    d, m = load(variant), load("master")
    lex = lexicon(m)
    jd_terms = set(count_terms(jd_text, lex))
    added = {}
    for j in d["experience"]:
        mj = next((x for x in m["experience"] if x["company"].lower() == j["company"].lower()), None)
        have = {clean(b).lower() for b in j["bullets"]}
        if mj:
            for b in mj["bullets"]:
                if clean(b).lower() not in have and set(count_terms(b, lex)) & jd_terms:
                    j["bullets"].append(b)
                    added.setdefault(j["company"], []).append(b[:70])
        j["bullets"] = sorted(j["bullets"], key=lambda b: -len(set(count_terms(b, lex)) & jd_terms))
    d["_meta"]["exp_synced"] = True
    save(variant, d, "synced experience with JD skills")
    return {"added_bullets": added}

def rank_bullets(jd_text: str, variant: str) -> list:
    d = load(variant)
    _, _, _, m = pool()
    lex = lexicon(m)
    jd_terms = set(count_terms(jd_text, lex))
    res = []
    for j in d["experience"]:
        rows = []
        for i, b in enumerate(j["bullets"]):
            hit = sorted(set(count_terms(b, lex)) & jd_terms)
            rows.append({"index": i, "score": len(hit), "matched_terms": hit, "text": b})
        order = [r["index"] for r in sorted(rows, key=lambda r: (-r["score"], r["index"]))]
        res.append({"job": j["id"], "recommended_order": order, "bullets": rows})
    return res

# ----------------------------------------------------------------------------- tracker
def log_application(entry: dict) -> None:
    rows = store.get_json("apps") or []
    rows.append({"date": dt.datetime.now().isoformat(timespec="seconds"), **entry})
    store.set_json("apps", rows)

def list_applications() -> list:
    return store.get_json("apps") or []
