"""
Resume Studio MCP  -  a deterministic toolbox. There is NO LLM inside this server.
The connected LLM (Claude etc.) decides *what* to write; these tools guarantee it is
consistent with your master resume, validated, well formatted, recoverable (undo), and
that applications are filled safely.

Local (Claude Desktop / Claude Code / any MCP client):  python server.py
Hosted:  TRANSPORT=streamable-http MCP_API_KEY=<secret> python server.py
"""
import io, ipaddress, json, os, re, socket, sys, tempfile, urllib.parse, datetime as dt
from email.message import EmailMessage
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
from mcp.server.fastmcp import FastMCP

INSTRUCTIONS = """Resume Studio: tools to tailor a resume to a job description, render it, email it and apply.
ONE-COMMAND MODE: if the user says 'apply <link>' (or 'apply thenlink'), do ONLY this: apply_start(url) -> then loop apply_next(job_id), executing exactly the action it returns, until step == 'done' or 'needs_user'.
Do not improvise the order; apply_next reads disk state so it always knows the next step. Email route = recruiter email found in JD (or in a pasted JD: apply_start(jd_text=...)) -> compose_email -> gmail_send_application (server sends via Gmail with the PDF attached; only if apply_next says so, fall back to YOUR OWN connected Gmail tool with the PDF attached, or if it cannot attach, the public resume link from the step's body_with_resume_link (never ask the user to upload) -> mark_application_sent.  No email -> form route (fill_application_form with submit=True; guards still stop on CAPTCHA/login/missing required fields). If apply_start says 'llm_fetch_needed', fetch the page with your OWN browsing tool and call apply_start again with jd_text (never ask the user to paste). For ANY application question (salary, notice, visa, experience, contact...) call answer_form_question first; experience years = dated total, never ask the user which figure. Ask only if it returns answer=null. Reply to the user with one line when finished.
TAILORING RULE: only ADD (JD skills from master, relevant master bullets) and REORDER. NEVER remove skills, bullets or roles.
MANUAL WORKFLOW (only when the user asks for individual steps):
WORKFLOW for a job: (1) fetch_jd if you were given a link; (2) create_variant(name) - NEVER edit 'master' unless the user explicitly asks to change their real resume;
(3) analyze_jd; (4) optimize_skills_for_jd, then fine-tune with add_skills/remove_skills; (5) rank_bullets_for_jd then reorder_bullets / update_bullet (rephrase only, keep facts);
(6) set_basics for headline + summary; (7) validate_resume until errors == [] ; (8) render_pdf; (9) compose_email; (10) for a link: inspect_application_form -> fill_application_form
(submit=False) -> show the user the screenshot path and filled/missing fields -> submit only after the user explicitly says yes.
TRUTH RULES: never add a skill, metric, employer, degree or certification that is not in master. If the user confirms something is genuinely true, pass allow_new=True (variants) or confirm_master=True (master).
Every write returns validation results - fix all 'errors' before rendering."""

SECRET = os.environ.get("RESUME_SECRET", "")
_kw = dict(instructions=INSTRUCTIONS, stateless_http=True, json_response=True)
try:  # Host-header (DNS-rebinding) checks are for localhost-only servers; we are behind a Cloudflare tunnel + secret path.
    from mcp.server.transport_security import TransportSecuritySettings
    _kw["transport_security"] = TransportSecuritySettings(enable_dns_rebinding_protection=False)
except Exception:
    pass
mcp = FastMCP("resume-studio", **_kw)
import apply_flow
apply_flow.register(mcp)   # apply_start, apply_next, answer_form_question, gmail_send_application, ...

# ----------------------------------------------------------------------------- helpers
def _edit(variant: str, confirm_master: bool) -> dict:
    if core.slug(variant) == "master" and not confirm_master:
        raise ValueError("'master' is the user's source of truth. Edit a variant (create_variant) or, if the user explicitly asked to change the real resume, pass confirm_master=True.")
    return core.load(variant)

def _guard_new(variant: str, d: dict, skills=(), texts=(), allow_new=False) -> None:
    if core.slug(variant) == "master":
        return
    skp, nump, _, _ = core.pool()
    ap = d["_meta"]["approved"]
    new_sk = sorted({s for s in skills if s.lower() not in skp and s.lower() not in {x.lower() for x in ap["skills"]}})
    t = " ".join(texts)
    new_nu = sorted(core.numbers_in(core.YEARS_RE.sub(" ", t)) - nump - set(ap["numbers"]))
    if (new_sk or new_nu) and not allow_new:
        raise ValueError(f"Blocked: not in master -> skills {new_sk}, numbers {new_nu}. Only pass allow_new=True if the user confirmed these are TRUE for them; otherwise remove them.")
    if allow_new:
        ap["skills"] = sorted(set(ap["skills"]) | set(new_sk))
        ap["numbers"] = sorted(set(ap["numbers"]) | set(new_nu))

def _done(variant: str, d: dict, note: str, changed: Any = None) -> dict:
    core.save(variant, d, note)
    v = core.validate(variant)
    return {"ok": True, "change": note, "result": changed, "errors": v["errors"], "warnings": v["warnings"], "info": v["info"]}

def _check_url(url: str) -> str:
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("Only http(s) URLs are allowed.")
    for ai in socket.getaddrinfo(u.hostname, None):
        ip = ipaddress.ip_address(ai[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("URL resolves to a private/internal address - refused.")
    return url

def _safe_pdf(path: str) -> str:
    """Accepts a PDF file name (or an old-style path) that exists in the output store; returns the bare file name."""
    n = os.path.basename(path or "")
    if not n.lower().endswith(".pdf") or not core.out_exists(n):
        raise ValueError("resume_pdf_path must be the file name returned by render_pdf (use render_pdf first).")
    return n

# ----------------------------------------------------------------------------- read tools
@mcp.tool()
def list_variants() -> dict:
    """List resume versions: 'master' (source of truth) plus per-job variants."""
    return {"variants": core.list_variants()}

@mcp.tool()
def get_resume(variant: str = "master", section: str = "all") -> dict:
    """Read a resume. section: all | basics | summary | skills | experience | education | certifications | awards | languages.
    Experience bullets are returned with their index so you can edit them precisely."""
    d = core.load(variant)
    d.pop("_meta", None)
    for j in d.get("experience", []):
        j["bullets"] = [{"index": i, "text": b} for i, b in enumerate(j["bullets"])]
    if section == "all":
        return d
    if section == "basics":
        return {k: d[k] for k in ("name", "headline", "contact", "summary")}
    if section not in d:
        raise ValueError(f"Unknown section '{section}'.")
    return {section: d[section]}

@mcp.tool()
def validate_resume(variant: str = "master") -> dict:
    """Run all correctness checks: dates, chronology, years-of-experience claims vs dated history, duplicate/over-long skills,
    skills/metrics/employers not present in master (fabrication guard), placeholders, empty fields."""
    return core.validate(variant)

@mcp.tool()
async def fetch_jd(url: str) -> dict:
    """Fetch a job-posting URL and return its readable text (falls back to a headless browser for JS-rendered pages)."""
    import httpx
    from bs4 import BeautifulSoup
    _check_url(url)
    text, title, src = "", "", "http"
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}) as c:
            r = await c.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style", "noscript", "nav", "footer"]):
            t.decompose()
        title, text = clean_title(soup), core.clean(soup.get_text(" "))
    except Exception:
        pass
    if len(text) < 500:
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                b, pg = await _open(p, url)
                title, text, src = await pg.title(), core.clean(await pg.inner_text("body")), "browser"
                await b.close()
        except Exception:
            return {"title": title, "source": src, "chars": len(text), "text": text[:20000],
                    "note": "Page is JS-rendered/blocked. Open it with YOUR OWN browsing tool and pass the text on (apply_start(jd_text=...))."}
    return {"title": title, "source": src, "chars": len(text), "text": text[:20000],
            "note": "Verify this is the full job description before tailoring." if len(text) < 800 else ""}

def clean_title(soup) -> str:
    return core.clean(soup.title.get_text()) if soup.title else ""

@mcp.tool()
def analyze_jd(jd_text: str, variant: str = "master") -> dict:
    """Deterministic JD analysis: which JD technologies the variant already shows, which exist in master but are missing from the variant (safe to add),
    and which are gaps not in master (must NOT be claimed). Also title candidates, years required, coverage %."""
    return core.analyze_jd(jd_text, variant)

@mcp.tool()
def rank_bullets_for_jd(variant: str, jd_text: str) -> list:
    """Score every experience bullet by overlap with the JD and recommend an order. Use with reorder_bullets."""
    return core.rank_bullets(jd_text, variant)

@mcp.tool()
def history(variant: str) -> dict:
    """List undo snapshots for a variant (newest first)."""
    return {"snapshots": core.history(variant)}

@mcp.resource("resume://{variant}")
def resume_resource(variant: str) -> str:
    d = core.load(variant)
    d.pop("_meta", None)
    return json.dumps(d, indent=2, ensure_ascii=False)

@mcp.prompt()
def tailor_for_job(jd: str, variant_name: str = "job") -> str:
    """Kick off the full tailoring workflow."""
    return (f"Tailor my resume for this job. Create variant '{variant_name}', analyze the JD, optimize skills, reorder/rephrase bullets (facts only), "
            f"set a role-matching headline and summary, validate until no errors, render the PDF and compose the email. Do not claim anything not in my master resume.\n\nJD:\n{jd}")

# ----------------------------------------------------------------------------- variants
@mcp.tool()
def create_variant(name: str, base: str = "master") -> dict:
    """Copy a resume into a new editable variant (e.g. one per job application)."""
    n = core.slug(name)
    if n == "master" or core.exists(n):
        raise ValueError(f"Variant '{n}' already exists or is reserved.")
    d = core.load(base)
    d["_meta"] = {"base": base, "approved": {"skills": [], "numbers": [], "companies": []}}
    core.save(n, d, f"created from {base}")
    return {"ok": True, "variant": n}

@mcp.tool()
def delete_variant(name: str) -> dict:
    """Delete a variant (never master)."""
    n = core.slug(name)
    if n == "master":
        raise ValueError("Cannot delete master.")
    core.load(n)
    core.delete_variant(n)
    return {"ok": True}

@mcp.tool()
def undo(variant: str, confirm_master: bool = False) -> dict:
    """Revert the last change to a variant."""
    _edit(variant, confirm_master)
    return {"ok": True, "restored_snapshot": core.undo(variant)}

# ----------------------------------------------------------------------------- basics
@mcp.tool()
def set_basics(variant: str, name: Optional[str] = None, headline: Optional[str] = None, summary: Optional[str] = None,
               email: Optional[str] = None, phone: Optional[str] = None, location: Optional[str] = None,
               links: Optional[list[str]] = None, allow_new: bool = False, confirm_master: bool = False) -> dict:
    """Update name, headline (role title line under the name), summary, or contact fields. Only pass what changes.
    Summary: 2-3 sentences, no invented numbers; years-of-experience claims are checked against dated history."""
    d = _edit(variant, confirm_master)
    if summary is not None:
        _guard_new(variant, d, texts=[summary], allow_new=allow_new)
    for k, v in (("name", name), ("headline", headline), ("summary", summary)):
        if v is not None: d[k] = core.clean(v)
    c = d.setdefault("contact", {})
    for k, v in (("email", email), ("phone", phone), ("location", location)):
        if v is not None: c[k] = core.clean(v)
    if links is not None: c["links"] = [core.clean(x) for x in links]
    return _done(variant, d, "updated basics")

# ----------------------------------------------------------------------------- skills
@mcp.tool()
def set_skill_category(variant: str, category: str, skills: list[str], position: Optional[int] = None,
                       allow_new: bool = False, confirm_master: bool = False) -> dict:
    """Create or fully replace one skill category (ordered list). position = 0-based place among categories."""
    d = _edit(variant, confirm_master)
    skills = [core.clean(s) for s in skills if core.clean(s)]
    _guard_new(variant, d, skills=skills, allow_new=allow_new)
    cats = dict(d["skills"])
    cats[core.clean(category)] = skills
    if position is not None:
        keys = [k for k in cats if k != core.clean(category)]
        keys.insert(position, core.clean(category))
        cats = {k: cats[k] for k in keys}
    d["skills"] = cats
    return _done(variant, d, f"set skills for '{category}'")

@mcp.tool()
def add_skills(variant: str, category: str, skills: list[str], allow_new: bool = False, confirm_master: bool = False) -> dict:
    """Append skills to a category (created if missing); duplicates are ignored."""
    d = _edit(variant, confirm_master)
    skills = [core.clean(s) for s in skills if core.clean(s)]
    _guard_new(variant, d, skills=skills, allow_new=allow_new)
    cur = d["skills"].setdefault(core.clean(category), [])
    have = {s.lower() for s in cur}
    cur += [s for s in skills if s.lower() not in have]
    return _done(variant, d, f"added skills to '{category}'")

@mcp.tool()
def remove_skills(variant: str, skills: list[str], confirm_master: bool = False) -> dict:
    """Remove skills (case-insensitive) from every category; empty categories are dropped."""
    d = _edit(variant, confirm_master)
    rm = {s.lower() for s in skills}
    d["skills"] = {k: [s for s in v if s.lower() not in rm] for k, v in d["skills"].items()}
    d["skills"] = {k: v for k, v in d["skills"].items() if v}
    return _done(variant, d, f"removed {len(rm)} skill(s)")

@mcp.tool()
def delete_skill_category(variant: str, category: str, confirm_master: bool = False) -> dict:
    """Delete a whole skill category."""
    d = _edit(variant, confirm_master)
    if category not in d["skills"]: raise ValueError(f"No category '{category}'. Have: {list(d['skills'])}")
    del d["skills"][category]
    return _done(variant, d, f"deleted category '{category}'")

@mcp.tool()
def reorder_skill_categories(variant: str, order: list[str], confirm_master: bool = False) -> dict:
    """Reorder categories; unlisted ones keep their relative order after the listed ones."""
    d = _edit(variant, confirm_master)
    keys = [k for k in order if k in d["skills"]] + [k for k in d["skills"] if k not in order]
    d["skills"] = {k: d["skills"][k] for k in keys}
    return _done(variant, d, "reordered skill categories")

@mcp.tool()
def optimize_skills_for_jd(variant: str, jd_text: str, max_skills: int = 0, min_skills: int = 0) -> dict:
    """Adds JD-required skills that exist in master and moves JD-mentioned skills to the front of their category. NEVER removes any skill
    (max_skills/min_skills are ignored). Gaps not in master are returned, not added (add them with add_skills(..., allow_new=True) only if the user confirms they are true)."""
    if core.slug(variant) == "master":
        raise ValueError("Use a variant, not master.")
    d = core.load(variant)
    before = {s.lower() for s in core.flat_skills(d)}
    d["skills"] = core.merge_skills_for_jd(d["skills"], jd_text)
    d["_meta"]["jd_tailored"] = True
    out = _done(variant, d, "added JD skills, reordered (nothing removed)")
    out["added_skills"] = [s for s in core.flat_skills(d) if s.lower() not in before]
    out["gaps_not_in_master"] = core.analyze_jd(jd_text, variant)["gaps_not_in_master"]
    return out

@mcp.tool()
def sync_experience_for_jd(variant: str, jd_text: str) -> dict:
    """Adds master bullets that demonstrate JD skills to the matching roles (if missing from the variant) and orders bullets by JD relevance. Never removes or invents bullets."""
    if core.slug(variant) == "master":
        raise ValueError("Use a variant, not master.")
    r = core.sync_experience_for_jd(variant, jd_text)
    v = core.validate(variant)
    return {"ok": True, **r, "errors": v["errors"], "warnings": v["warnings"]}

# ----------------------------------------------------------------------------- experience
@mcp.tool()
def add_job(variant: str, title: str, company: str, dates: str, bullets: list[str], position: int = 0,
            allow_new: bool = False, confirm_master: bool = False) -> dict:
    """Add a role (dates like 'Jan 2020 – Mar 2022'). Only for a REAL role the user told you about -> requires allow_new=True on variants."""
    d = _edit(variant, confirm_master)
    if core.slug(variant) != "master":
        if not allow_new: raise ValueError("Adding a role not in master requires allow_new=True and the user's confirmation that it is real.")
        d["_meta"]["approved"]["companies"] = sorted(set(d["_meta"]["approved"]["companies"]) | {core.clean(company)})
        d["_meta"]["approved"]["numbers"] = sorted(set(d["_meta"]["approved"]["numbers"]) | core.numbers_in(" ".join(bullets)))
    d["experience"].insert(position, {"id": core.slug(company), "title": core.clean(title), "company": core.clean(company),
                                      "dates": core.clean(dates), "bullets": [core.clean(b) for b in bullets]})
    return _done(variant, d, f"added job {company}")

@mcp.tool()
def update_job(variant: str, job: str, title: Optional[str] = None, company: Optional[str] = None, dates: Optional[str] = None,
               confirm_master: bool = False) -> dict:
    """Edit a role's title/company/dates. job = id, company name, or 0-based index. (On variants, dates must stay equal to master or validation fails.)"""
    d = _edit(variant, confirm_master)
    j = core.find_job(d, job)
    for k, v in (("title", title), ("company", company), ("dates", dates)):
        if v is not None: j[k] = core.clean(v)
    return _done(variant, d, f"updated job {j['company']}")

@mcp.tool()
def delete_job(variant: str, job: str, confirm_master: bool = False) -> dict:
    """Remove a role from the resume."""
    d = _edit(variant, confirm_master)
    j = core.find_job(d, job)
    d["experience"].remove(j)
    return _done(variant, d, f"deleted job {j['company']}")

@mcp.tool()
def set_bullets(variant: str, job: str, bullets: list[str], allow_new: bool = False, confirm_master: bool = False) -> dict:
    """Replace ALL bullets of a role. Keep facts/metrics identical to master; rephrase and reorder only."""
    d = _edit(variant, confirm_master)
    j = core.find_job(d, job)
    bl = [core.clean(b) for b in bullets if core.clean(b)]
    _guard_new(variant, d, texts=bl, allow_new=allow_new)
    j["bullets"] = bl
    return _done(variant, d, f"set bullets for {j['company']}", {"bullets": list(enumerate(bl))})

@mcp.tool()
def add_bullet(variant: str, job: str, text: str, index: Optional[int] = None, allow_new: bool = False, confirm_master: bool = False) -> dict:
    """Insert a bullet at index (default: end)."""
    d = _edit(variant, confirm_master)
    j = core.find_job(d, job)
    _guard_new(variant, d, texts=[text], allow_new=allow_new)
    j["bullets"].insert(len(j["bullets"]) if index is None else index, core.clean(text))
    return _done(variant, d, f"added bullet to {j['company']}", list(enumerate(j["bullets"])))

@mcp.tool()
def update_bullet(variant: str, job: str, index: int, text: str, allow_new: bool = False, confirm_master: bool = False) -> dict:
    """Rewrite one bullet."""
    d = _edit(variant, confirm_master)
    j = core.find_job(d, job)
    if not 0 <= index < len(j["bullets"]): raise ValueError(f"index out of range 0..{len(j['bullets']) - 1}")
    _guard_new(variant, d, texts=[text], allow_new=allow_new)
    j["bullets"][index] = core.clean(text)
    return _done(variant, d, f"updated bullet {index} of {j['company']}", list(enumerate(j["bullets"])))

@mcp.tool()
def remove_bullet(variant: str, job: str, index: int, confirm_master: bool = False) -> dict:
    """Delete one bullet."""
    d = _edit(variant, confirm_master)
    j = core.find_job(d, job)
    if not 0 <= index < len(j["bullets"]): raise ValueError(f"index out of range 0..{len(j['bullets']) - 1}")
    j["bullets"].pop(index)
    return _done(variant, d, f"removed bullet {index} of {j['company']}", list(enumerate(j["bullets"])))

@mcp.tool()
def reorder_bullets(variant: str, job: str, order: list[int], drop_others: bool = False, confirm_master: bool = False) -> dict:
    """Reorder bullets by their indices. Unlisted bullets go to the end, or are removed if drop_others=True (use to keep only the most relevant)."""
    d = _edit(variant, confirm_master)
    j = core.find_job(d, job)
    n = len(j["bullets"])
    if len(set(order)) != len(order) or any(not 0 <= i < n for i in order): raise ValueError(f"order must be unique indices in 0..{n - 1}")
    rest = [] if drop_others else [i for i in range(n) if i not in order]
    j["bullets"] = [j["bullets"][i] for i in order + rest]
    return _done(variant, d, f"reordered bullets of {j['company']}", list(enumerate(j["bullets"])))

# ----------------------------------------------------------------------------- other sections
@mcp.tool()
def set_education(variant: str, items: list[dict], confirm_master: bool = False) -> dict:
    """Replace education. Each item: {"degree": "...", "place": "City, Country", "dates": "Aug 2022 – Sep 2024"}."""
    d = _edit(variant, confirm_master)
    for it in items:
        if not {"degree", "place", "dates"} <= set(it): raise ValueError("each item needs degree, place, dates")
    d["education"] = [{k: core.clean(it[k]) for k in ("degree", "place", "dates")} for it in items]
    return _done(variant, d, "set education")

@mcp.tool()
def set_list_section(variant: str, section: str, items: list[str], confirm_master: bool = False) -> dict:
    """Replace a simple list section: certifications | awards | languages."""
    if section not in ("certifications", "awards", "languages"): raise ValueError("section must be certifications, awards or languages")
    d = _edit(variant, confirm_master)
    d[section] = [core.clean(i) for i in items if core.clean(i)]
    return _done(variant, d, f"set {section}")

# ----------------------------------------------------------------------------- render
@mcp.tool()
async def render_pdf(variant: str = "master", target_pages: int = 2, force: bool = False) -> dict:
    """Validate then render an ATS-friendly single-column A4 PDF (auto-shrinks font until it fits target_pages). Refuses if validation has errors unless force=True.
    After rendering, the PDF text is re-extracted and cross-checked against the data (nothing missing/cut). Returns the PDF file name (pdf_path)."""
    from pypdf import PdfReader
    import pdfgen
    v = core.validate(variant)
    if v["errors"] and not force:
        return {"ok": False, "blocked_by": v["errors"], "hint": "Fix these (or force=True) then render again."}
    d = core.load(variant)
    d.pop("_meta", None)
    pdf, pages, used = b"", 99, 10.0
    for fs in (10.0, 9.5, 9.0, 8.5, 8.0):
        pdf = pdfgen.build(d, core.contact_line(d), fs)
        pages, used = len(PdfReader(io.BytesIO(pdf)).pages), fs
        if pages <= target_pages: break
    text = " ".join((pp.extract_text() or "") for pp in PdfReader(io.BytesIO(pdf)).pages)
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    tn = norm(text)
    must = [d["name"], d["headline"], d["contact"]["email"]] + [j["company"] for j in d["experience"]] + d.get("certifications", []) \
        + [b_[:28] for j in d["experience"] for b_ in j["bullets"]] + [s for v_ in d["skills"].values() for s in v_[:1]]
    missing = [m for m in must if norm(pdfgen._t(m).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")) not in tn]
    name = core.out_write(f"{core.slug(d['name'])}_{core.slug(variant)}.pdf", pdf)
    warn = list(v["warnings"])
    if pages > target_pages: warn.append(f"Still {pages} pages at {used}pt - trim bullets/skills.")
    return {"ok": not missing, "pdf_path": name, "pages": pages, "font_pt": used, "text_check_missing": missing, "warnings": warn}

# ----------------------------------------------------------------------------- email
@mcp.tool()
def compose_email(variant: str, role: str, company: str, hook: str, key_points: list[str], to: str = "",
                  recruiter_name: str = "", closing: str = "", subject: str = "", pdf_path: str = "") -> dict:
    """Assemble the application email deterministically from pieces YOU write: hook (1-2 sentences), 2-4 key_points (each must be backed by the resume),
    optional closing. Checks: word count, placeholders, tools/numbers mentioned must exist in the variant. Saves a .eml (with the resume PDF attached) and returns a mailto link."""
    d = core.load(variant)
    pts = [core.clean(p) for p in key_points if core.clean(p)]
    if not 2 <= len(pts) <= 4: raise ValueError("Provide 2-4 key_points.")
    greet = f"Hi {core.clean(recruiter_name)}," if recruiter_name else "Hi Hiring Team,"
    close = core.clean(closing) or "I've attached my resume and would welcome the chance to discuss how I can contribute."
    c = d["contact"]
    sig = f"Best regards,\n{d['name']}\n" + " | ".join(x for x in [c.get("email"), c.get("phone")] + c.get("links", []) if x)
    body = f"{greet}\n\n{core.clean(hook)}\n\n" + "\n".join(f"- {p}" for p in pts) + f"\n\n{close}\n\n{sig}"
    subj = core.clean(subject) or f"Application for {core.clean(role)} - {d['name']}"
    issues = []
    if len(body.split()) > 220: issues.append(f"email is {len(body.split())} words; keep it under ~200")
    if core.PLACEHOLDER.search(body + subj): issues.append("placeholder/bracket text present")
    _, nump, _, m = core.pool()
    allowed_nums = nump | core.numbers_in(json.dumps(d)) | core.numbers_in(sig)
    bad_n = sorted(core.numbers_in(core.YEARS_RE.sub(" ", " ".join(pts + [hook, close]))) - allowed_nums)
    if bad_n: issues.append(f"numbers not backed by the resume: {bad_n}")
    for y in core.YEARS_RE.findall(" ".join(pts + [hook, close])):
        if float(y) > core.experience_years(d) + 0.75:
            issues.append(f"claims {y}+ years but dated experience totals {core.experience_years(d)} years")
    have = {s.lower() for s in core.flat_skills(d)}
    mentioned = core.count_terms(" ".join(pts + [hook, close]), core.lexicon(m))
    bad_t = sorted(k for k in mentioned if k.lower() not in have)
    if bad_t: issues.append(f"technologies mentioned that are not in this resume: {bad_t}")
    pdf = os.path.basename(pdf_path) if pdf_path else f"{core.slug(d['name'])}_{core.slug(variant)}.pdf"
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subj, c.get("email", ""), to
    msg.set_content(body)
    attached = False
    if core.out_exists(pdf):
        msg.add_attachment(core.out_read(_safe_pdf(pdf)), maintype="application", subtype="pdf", filename=os.path.basename(pdf))
        attached = True
    else:
        issues.append("resume PDF not found - run render_pdf first so it can be attached")
    eml = core.out_write(f"email_{core.slug(company)}_{core.slug(variant)}.eml", bytes(msg))
    mailto = f"mailto:{urllib.parse.quote(to)}?subject={urllib.parse.quote(subj)}&body={urllib.parse.quote(body)}"
    return {"ok": not issues, "issues": issues, "subject": subj, "body": body, "words": len(body.split()), "eml_path": eml,
            "attachment_included": attached, "mailto": mailto, "note": "Send with gmail_send_application (attaches the PDF)."}

# ----------------------------------------------------------------------------- applying
INSPECT_JS = r"""() => {
 const vis = e => !!(e.offsetWidth || e.offsetHeight || e.getClientRects().length);
 const esc = s => (window.CSS && CSS.escape) ? CSS.escape(s) : s.replace(/[^a-zA-Z0-9_-]/g, '\\$&');
 const sel = (e, i) => {
   const t = e.tagName.toLowerCase(), n = e.getAttribute('name'), type = e.getAttribute('type');
   if (e.id && document.querySelectorAll('#' + esc(e.id)).length === 1) return '#' + esc(e.id);
   if (n && (type === 'radio' || type === 'checkbox')) return t + '[name="' + n.replace(/"/g,'\\"') + '"][value="' + (e.value||'').replace(/"/g,'\\"') + '"]';
   if (n && document.querySelectorAll(t + '[name="' + n.replace(/"/g,'\\"') + '"]').length === 1) return t + '[name="' + n.replace(/"/g,'\\"') + '"]';
   return '__idx__' + i;
 };
 const label = e => {
   let t = '';
   if (e.id) { const l = document.querySelector('label[for="' + esc(e.id) + '"]'); if (l) t = l.innerText; }
   if (!t) { const l = e.closest('label'); if (l) t = l.innerText; }
   if (!t) t = e.getAttribute('aria-label') || '';
   if (!t && e.getAttribute('aria-labelledby')) { const x = document.getElementById(e.getAttribute('aria-labelledby')); if (x) t = x.innerText; }
   if (!t) t = e.placeholder || '';
   return t.replace(/\s+/g, ' ').trim().slice(0, 160);
 };
 const all = Array.from(document.querySelectorAll('input,select,textarea'));
 const out = [];
 all.forEach((e, i) => {
   const type = (e.getAttribute('type') || e.tagName).toLowerCase();
   if (['hidden','submit','button','image','reset'].includes(type)) return;
   if (type !== 'file' && !vis(e)) return;
   out.push({selector: sel(e, i), label: label(e), name: e.getAttribute('name') || '', type,
     required: !!(e.required || e.getAttribute('aria-required') === 'true'),
     options: e.tagName === 'SELECT' ? Array.from(e.options).map(o => o.text.trim()).slice(0, 40) : undefined,
     value: (type === 'checkbox' || type === 'radio') ? (e.checked ? 'checked' : '') : (e.value || '')});
 });
 return {fields: out.slice(0, 120),
   title: document.title,
   captcha: !!document.querySelector('iframe[src*="captcha"],iframe[src*="turnstile"],.g-recaptcha,.h-captcha,[data-sitekey]'),
   login_wall: !!document.querySelector('input[type=password]'),
   frames: Array.from(document.querySelectorAll('iframe')).map(f => f.src).filter(s => /greenhouse|lever|ashby|workable|smartrecruiters|bamboohr|recruitee|jobvite|icims|workday/i.test(s)),
   buttons: Array.from(document.querySelectorAll('button,input[type=submit]')).filter(vis).map(b => (b.innerText || b.value || '').trim()).filter(Boolean).slice(0, 10)};
}"""

async def _open(p, url, headless=True):
    """Local: launches Chromium. Serverless (Vercel): connects to a remote browser given by BROWSER_WS_ENDPOINT (Browserless/Browserbase CDP URL)."""
    from playwright.async_api import async_playwright
    ws = os.environ.get("BROWSER_WS_ENDPOINT", "").strip()
    if ws:
        b = await p.chromium.connect_over_cdp(ws)
        ctx = b.contexts[0] if b.contexts else await b.new_context()
    else:
        if os.environ.get("VERCEL"):
            raise RuntimeError("No browser on Vercel. Set BROWSER_WS_ENDPOINT (e.g. Browserless/Browserbase CDP websocket URL) to enable form tools.")
        b = await p.chromium.launch(headless=headless, args=["--disable-http2", "--disable-blink-features=AutomationControlled"])
        ctx = await b.new_context(user_agent=apply_flow.UA, locale="en-US", viewport={"width": 1280, "height": 1800})
    pg = await ctx.new_page()
    await pg.set_viewport_size({"width": 1280, "height": 1800})
    await pg.goto(url, wait_until="domcontentloaded", timeout=45000)
    try: await pg.wait_for_load_state("networkidle", timeout=10000)
    except Exception: pass
    return b, pg

async def _resolve(pg, fields_js):
    """map '__idx__N' pseudo selectors to nth element handles"""
    return await pg.evaluate(fields_js)

@mcp.tool()
async def inspect_application_form(url: str) -> dict:
    """Open an application page (read-only) and list its fields: label, type, required, options, selector. Also reports CAPTCHA / login wall / embedded ATS iframes.
    If 'frames' is non-empty the real form lives in that iframe URL - call this again with it."""
    from playwright.async_api import async_playwright
    _check_url(url)
    async with async_playwright() as p:
        b, pg = await _open(p, url)
        info = await _resolve(pg, INSPECT_JS)
        info["final_url"] = pg.url
        await b.close()
    info["blocked"] = "CAPTCHA present - the user must apply manually" if info["captcha"] else ("login required - the user must sign in manually" if info["login_wall"] else "")
    return info

@mcp.tool()
async def fill_application_form(url: str, answers: dict[str, Any], resume_pdf_path: str, resume_selector: str = "",
                                variant: str = "", role: str = "", company: str = "", submit: bool = False) -> dict:
    """Fill a form and attach the resume. answers = {selector: value} using selectors from inspect_application_form (text -> string, select -> option text,
    checkbox -> true/false, radio -> use the radio's own selector with true). Default submit=False saves a screenshot for the user to review.
    submit=True only after the user explicitly approved; it refuses if CAPTCHA/login or required fields are still empty. Never guess answers to legal/EEO/salary questions - ask the user."""
    from playwright.async_api import async_playwright
    _check_url(url)
    pdf_name = _safe_pdf(resume_pdf_path)
    tmpdir = tempfile.mkdtemp()
    pdf = os.path.join(tmpdir, pdf_name)
    open(pdf, "wb").write(core.out_read(pdf_name))
    status = {}
    async with async_playwright() as p:
        b, pg = await _open(p, url)
        info = await _resolve(pg, INSPECT_JS)
        if info["captcha"] or info["login_wall"]:
            await b.close()
            return {"ok": False, "status": "blocked", "reason": "CAPTCHA or login wall - user must apply manually", "url": url}
        sels = {}
        nth = await pg.query_selector_all("input,select,textarea")
        for f in info["fields"]:
            s = f["selector"]
            if s.startswith("__idx__"):
                f["_h"] = nth[int(s[7:])]
            else:
                f["_h"] = await pg.query_selector(s)
            sels[s] = f
        for s, val in answers.items():
            f = sels.get(s)
            h = f["_h"] if f else await pg.query_selector(s)
            if not h: status[s] = "not found"; continue
            try:
                tag = (await h.evaluate("e => e.tagName")).lower()
                typ = ((await h.get_attribute("type")) or "").lower()
                if tag == "select":
                    try: await h.select_option(label=str(val))
                    except Exception: await h.select_option(value=str(val))
                elif typ in ("checkbox", "radio"):
                    await h.set_checked(bool(val))
                else:
                    await h.fill(str(val))
                status[s] = "ok"
            except Exception as ex:
                status[s] = f"failed: {str(ex)[:80]}"
        files = [f for f in info["fields"] if f["type"] == "file"]
        target = resume_selector or (files[0]["selector"] if files else "")
        attached = False
        if target:
            h = sels[target]["_h"] if target in sels else await pg.query_selector(target)
            if h:
                await h.set_input_files(pdf); attached = True
        after = await _resolve(pg, INSPECT_JS)
        missing = [{"selector": f["selector"], "label": f["label"]} for f in after["fields"]
                   if f["required"] and f["type"] != "file" and not f["value"]]
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        shot = f"form_preview_{stamp}.png"
        core.out_write(shot, await pg.screenshot(full_page=True))
        result = {"ok": True, "status": "filled_awaiting_review", "fields": status, "resume_attached": attached,
                  "missing_required": missing, "screenshot": apply_flow.public_url(shot)}
        if not attached: result["warning"] = "No file input found/used - attach the resume manually."
        if submit:
            if missing or not attached:
                result.update(ok=False, status="not_submitted", reason="required fields empty or resume not attached")
            else:
                btn = await pg.query_selector("button[type=submit], input[type=submit]")
                if not btn:
                    for t in ("Submit application", "Submit", "Apply", "Send"):
                        btn = await pg.query_selector(f"button:has-text('{t}')")
                        if btn: break
                if not btn:
                    result.update(ok=False, status="not_submitted", reason="submit button not found")
                else:
                    await btn.click()
                    await pg.wait_for_timeout(5000)
                    body = (await pg.inner_text("body")).lower()
                    confirmed = bool(re.search(r"thank you|application (has been )?(submitted|received)|successfully|we.ve received", body))
                    shot2 = f"form_after_submit_{stamp}.png"
                    core.out_write(shot2, await pg.screenshot(full_page=True))
                    result.update(status="submitted", confirmed_by_page=confirmed, screenshot_after=apply_flow.public_url(shot2),
                                  note="" if confirmed else "No confirmation text detected - check the screenshot / your email.")
        await b.close()
    core.log_application({"company": company, "role": role, "url": url, "variant": variant, "pdf": pdf_name, "status": result["status"]})
    return result

@mcp.tool()
def log_application(company: str, role: str, url: str = "", variant: str = "", status: str = "applied", notes: str = "") -> dict:
    """Manually record an application (e.g. one you applied to by email)."""
    core.log_application({"company": company, "role": role, "url": url, "variant": variant, "status": status, "notes": notes})
    return {"ok": True}

@mcp.tool()
def list_applications() -> list:
    """Show the application tracker."""
    return core.list_applications()

# ----------------------------------------------------------------------------- HTTP (tunnel) entrypoint
import hmac

def file_sig(secret: str, name: str) -> str:
    return hmac.new(secret.encode(), name.encode(), "sha256").hexdigest()[:32]

class SecretPathGate:
    """ASGI entry (Vercel + local). Serves: /healthz, /dl/<sig>/<file> (public unguessable PDF/PNG links), and MCP at /<SECRET>/mcp
    (or /mcp with 'Authorization: Bearer <SECRET>'). Everything else is 404. A fresh stateless MCP session manager is created per request,
    so it works on serverless where ASGI lifespan / long-lived tasks are not available."""
    def __init__(self, mcp_server, secret: str):
        if len(secret) < 16:
            raise SystemExit("RESUME_SECRET must be at least 16 characters (set it in Vercel -> Settings -> Environment Variables).")
        self.mcp, self.secret = mcp_server, secret

    async def _respond(self, send, status: int, body: bytes = b"", ctype: bytes = b"text/plain", extra=()):
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", ctype), (b"content-length", str(len(body)).encode()), *extra]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                m = await receive()
                if m["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif m["type"] == "lifespan.shutdown":
                    return await send({"type": "lifespan.shutdown.complete"})
        if scope["type"] != "http":
            return
        path = scope.get("path", "")
        if path in ("", "/"):
            return await self._respond(send, 200, b"ok")
        if path == "/healthz":
            return await self._respond(send, 200, ("ok " + __import__("store").info()["backend"]).encode())
        dl = path.split("/", 3)
        if len(dl) == 4 and dl[1] == "dl":   # public, unguessable per-file link (does NOT reveal the MCP secret)
            fn = os.path.basename(urllib.parse.unquote(dl[3]))
            ext = fn.lower().rsplit(".", 1)[-1]
            good = hmac.compare_digest(dl[2].encode(), file_sig(self.secret, fn).encode())
            data = core.out_read(fn) if good and ext in ("pdf", "png") else None
            if data is None:
                return await self._respond(send, 404, b"Not found")
            return await self._respond(send, 200, data, b"application/pdf" if ext == "pdf" else b"image/png",
                                       [(b"content-disposition", f'inline; filename="{fn}"'.encode())])
        parts = path.split("/", 2)
        if len(parts) >= 2 and parts[1] and hmac.compare_digest(parts[1].encode(), self.secret.encode()):
            ok = True
        else:
            auth = dict(scope.get("headers", [])).get(b"authorization", b"").decode()
            ok = auth.lower().startswith("bearer ") and hmac.compare_digest(auth[7:].strip().encode(), self.secret.encode())
        if not ok:
            return await self._respond(send, 404, b"Not found")
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        mgr = StreamableHTTPSessionManager(app=self.mcp._mcp_server, json_response=True, stateless=True,
                                           security_settings=self.mcp.settings.transport_security)
        async with mgr.run():
            await mgr.handle_request(dict(scope, path="/mcp", raw_path=b"/mcp"), receive, send)

app = SecretPathGate(mcp, SECRET) if SECRET else None   # ASGI app: Vercel (api/index.py) and local (python run.py)

if __name__ == "__main__":
    mcp.run()   # stdio (Claude Desktop / Claude Code local)
