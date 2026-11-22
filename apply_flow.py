"""
apply_flow.py - one-command application pipeline layered on top of the existing Resume Studio tools.
Entry point for the LLM:  apply_start(url)  then  apply_next(job_id)  in a loop until step == "done".
apply_next derives the next step from files on disk (variant, PDF, .eml, tracker), so the LLM can never lose its place.
No LLM inside. Mail is sent by gmail_send_application (SMTP/IMAP, creds GMAIL_ADDRESS + GMAIL_APP_PASSWORD in .env); if those are missing, the calling LLM's own Gmail connector is the fallback.
"""
import base64, datetime as dt, hashlib, imaplib, ipaddress, json, os, re, smtplib, socket, urllib.parse
from email import policy
from email.parser import BytesParser
from email.utils import make_msgid
from typing import Optional

import core
import store


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
BAD_EMAIL = re.compile(r"(no-?reply|donotreply|do-not-reply|example\.|sentry|wixpress|@2x|\.(png|jpe?g|gif|svg|webp)$)", re.I)
PREF_LOCAL = re.compile(r"(job|career|hr|recruit|talent|hiring|apply|resume|cv|people)", re.I)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
BROWSER_ARGS = ["--disable-http2", "--disable-blink-features=AutomationControlled"]
ATS = re.compile(r"(greenhouse|lever\.co|ashbyhq|workable|smartrecruiters|bamboohr|recruitee|jobvite|icims|myworkdayjobs|zohorecruit|keka|freshteam|darwinbox|naukri)", re.I)


# ----------------------------------------------------------------------------- helpers
def _check_url(url: str) -> str:
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("Only http(s) URLs are allowed.")
    for ai in socket.getaddrinfo(u.hostname, None):
        ip = ipaddress.ip_address(ai[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("URL resolves to a private/internal address - refused.")
    return url


def _jexists(job_id: str) -> bool:
    return store.exists(f"job/{core.slug(job_id)}")


def _jload(job_id: str) -> dict:
    j = store.get_json(f"job/{core.slug(job_id)}")
    if j is None:
        raise ValueError(f"Unknown job_id '{job_id}'. Call apply_start(url) first.")
    return j


def _jsave(job_id: str, j: dict) -> None:
    store.set_json(f"job/{core.slug(job_id)}", j)


def _deobfuscate(t: str) -> str:
    t = re.sub(r"\s*[\[\(\{]\s*at\s*[\]\)\}]\s*", "@", t, flags=re.I)
    t = re.sub(r"\s*[\[\(\{]\s*dot\s*[\]\)\}]\s*", ".", t, flags=re.I)
    return t


def _emails(text: str, html: str = "") -> list:
    found = EMAIL_RE.findall(_deobfuscate(text))
    for m in re.findall(r"mailto:([^\"'?>\s]+)", html or "", flags=re.I):
        found += EMAIL_RE.findall(urllib.parse.unquote(m))
    out = []
    for e in found:
        e = e.strip(".,;:").lower()
        if not BAD_EMAIL.search(e) and e not in out:
            out.append(e)
    return sorted(out, key=lambda e: (not PREF_LOCAL.search(e.split("@")[0]), out.index(e)))


def _links(html: str, base: str) -> list:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html or "", "html.parser")
    res = []
    for a in soup.find_all("a", href=True):
        href = urllib.parse.urljoin(base, a["href"])
        if not href.startswith("http"):
            continue
        txt = core.clean(a.get_text(" ")).lower()
        if (re.search(r"\bapply\b", txt) or ATS.search(href) or re.search(r"/apply\b", href, re.I)) and href not in res:
            res.append(href)
    return sorted(res, key=lambda h: (not ATS.search(h), res.index(h)))[:5]


async def _fetch(url: str):
    import httpx
    from bs4 import BeautifulSoup
    _check_url(url)
    title, text, html, src = "", "", "", "http"
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers={"User-Agent": UA}) as c:
            r = await c.get(url)
        html = r.text
        soup = BeautifulSoup(html, "html.parser")
        title = core.clean(soup.title.get_text()) if soup.title else ""
        for t in soup(["script", "style", "noscript", "nav", "footer"]):
            t.decompose()
        text = core.clean(soup.get_text(" "))
    except Exception:
        pass
    if len(text) < 500:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            ws = os.environ.get("BROWSER_WS_ENDPOINT", "").strip()
            if not ws and os.environ.get("VERCEL"):
                raise RuntimeError("no browser on Vercel")
            if ws:
                b = await p.chromium.connect_over_cdp(ws)
                ctx = b.contexts[0] if b.contexts else await b.new_context()
            else:
                b = await p.chromium.launch(args=BROWSER_ARGS)
                ctx = await b.new_context(user_agent=UA, locale="en-US", viewport={"width": 1366, "height": 900})
            pg = await ctx.new_page()
            await pg.goto(url, wait_until="domcontentloaded", timeout=45000)
            try: await pg.wait_for_load_state("networkidle", timeout=10000)
            except Exception: pass
            title, text, html, src = await pg.title(), core.clean(await pg.inner_text("body")), await pg.content(), "browser"
            await b.close()
    return title, text, html, src


def _pdf_for(v: str) -> str:
    d = core.load(v)
    return f"{core.slug(d['name'])}_{core.slug(v)}.pdf"


def _eml_for(v: str):
    c = sorted(core.out_list(f"email_*_{core.slug(v)}.eml"), key=core.out_mtime)
    return c[-1] if c else None


def _base_url() -> str:
    for k in ("PUBLIC_BASE_URL", "VERCEL_PROJECT_PRODUCTION_URL", "VERCEL_URL", "RESUME_DOMAIN"):
        v = os.environ.get(k, "").strip().replace("https://", "").replace("http://", "").strip("/")
        if v:
            return "https://" + v
    return "http://127.0.0.1:" + os.environ.get("PORT", "8788")


def public_url(name: str) -> str:
    import hmac
    sig = hmac.new(os.environ.get("RESUME_SECRET", "").encode(), name.encode(), "sha256").hexdigest()[:32]
    return f"{_base_url()}/dl/{sig}/{urllib.parse.quote(name)}"


def _answers() -> dict:
    return store.get_json("answers") or {}


def _app_status(url: str) -> str:
    rows = [r for r in core.list_applications() if r.get("url") == url]
    return rows[-1]["status"] if rows else ""


# ----------------------------------------------------------------------------- gmail
def _gmail_creds():
    a, p = os.environ.get("GMAIL_ADDRESS", "").strip(), os.environ.get("GMAIL_APP_PASSWORD", "").replace(" ", "").strip()
    return (a, p) if a and p else None


def _drafts_folder(imap) -> str:
    typ, data = imap.list()
    for line in data or []:
        s = line.decode(errors="ignore")
        if "\\Drafts" in s:
            m = re.search(r'"([^"]+)"\s*$', s) or re.search(r"(\S+)\s*$", s)
            if m:
                return m.group(1)
    return "[Gmail]/Drafts"


# ----------------------------------------------------------------------------- tools
def register(mcp):
    @mcp.tool()
    async def apply_start(url: str = "", jd_text: str = "") -> dict:
        """START HERE for 'apply <link>'. Fetches the job page, extracts JD text, recruiter email(s) and apply links, picks the route
        ('email' if an email exists, else 'form'), stores the job, and returns job_id. Then call apply_next(job_id) repeatedly."""
        if not url:
            if len(jd_text) < 100:
                return {"ok": False, "reason": "Provide a url or the pasted jd_text."}
            url = "pasted-" + hashlib.sha1(jd_text.encode()).hexdigest()[:8]
        try:
            if url.startswith("pasted-"): raise RuntimeError("pasted")
            title, text, html, src = await _fetch(url)
        except Exception as ex:
            if url.startswith("pasted-"): ex = "pasted JD"
            if not url.startswith("pasted-") and len(jd_text) < 300:
                return {"ok": False, "step": "llm_fetch_needed", "reason": f"server could not fetch the page ({str(ex)[:100]}).",
                        "do": "Open this URL yourself with YOUR OWN browsing/web-fetch tool, copy the FULL job description text (include any email address and the apply link), "
                              "then call apply_start(url=<same url>, jd_text=<that text>). Do not ask the user. Do not summarize the JD."}
            title, text, html, src = "", "", "", "pasted"
        if len(jd_text) >= 300:
            text = jd_text if len(jd_text) > len(text) else text
        emails, links = _emails(text + " " + jd_text, html), (_links(html, url) if html else [])
        jid = core.slug(urllib.parse.urlparse(url).hostname or "pasted")[:30] + "-" + hashlib.sha1(url.encode()).hexdigest()[:6]
        old = _jexists(jid)
        j = _jload(jid) if old else {}
        j.update(url=url, title=title, jd_text=text[:20000], emails=emails, apply_links=links,
                 route="email" if emails else "form", form_url=(links[0] if links else url),
                 created=j.get("created") or dt.datetime.now().isoformat(timespec="seconds"), sent=j.get("sent", False))
        _jsave(jid, j)
        warn = []
        if re.search(r"naukri|linkedin|indeed|glassdoor|bayt", url, re.I) and not emails:
            warn.append("Job board needs login: no email in JD means the form route will stop at 'needs_user'. Look for the company's own careers/ATS link.")
        if len(text) < 800: warn.append("JD text is short - if the page was blocked, fetch it with YOUR OWN browsing tool and call apply_start again with jd_text. Original note: JD text is short - verify it is the full description; LinkedIn/login pages often hide it.")
        return {"job_id": jid, "resumed_existing": old, "route": j["route"], "emails": emails, "form_url": j["form_url"],
                "title": title, "jd_text": j["jd_text"], "warnings": warn,
                "next": f"call apply_next(job_id='{jid}') and do exactly what it says; repeat until step == 'done'."}

    @mcp.tool()
    def apply_next(job_id: str) -> dict:
        """Returns the single next action for this application, derived from files on disk. Execute ONLY that action, then call apply_next again.
        Stops with step 'done' or 'needs_user'. Never skip steps, never invent your own order."""
        j = _jload(job_id)
        v = core.slug(job_id)
        jd = "<the jd_text returned by apply_start; or read jobs/%s.json>" % v
        if j.get("sent") or _app_status(j["form_url"]) == "submitted":
            return {"step": "done", "note": "Application completed and logged. Tell the user in one line."}
        if not core.exists(v):
            return {"step": "1_create_variant", "do": {"tool": "create_variant", "args": {"name": v}}}
        d, m = core.load(v), core.load("master")
        if not d["_meta"].get("jd_tailored"):
            return {"step": "2_skills", "do": {"tool": "optimize_skills_for_jd", "args": {"variant": v, "jd_text": jd}},
                    "then": "Skills are only ADDED/reordered, never removed. Gaps not in master: add via add_skills(allow_new=True) only if the user said they are true."}
        if not d["_meta"].get("exp_synced"):
            return {"step": "3_experience", "do": {"tool": "sync_experience_for_jd", "args": {"variant": v, "jd_text": jd}},
                    "then": "Optionally weave JD skills into existing bullets with update_bullet (truthful, same facts). Never remove bullets."}
        if d["headline"] == m["headline"] or d["summary"] == m["summary"]:
            return {"step": "4_headline_summary",
                    "do": {"tool": "set_basics", "args": {"variant": v, "headline": "<role title from JD, truthful>", "summary": "<2-3 sentences, facts from master only>"}},
                    "optional": "Never drop bullets or skills."}
        val = core.validate(v)
        if val["errors"]:
            return {"step": "4_fix_validation", "errors": val["errors"],
                    "do": "fix every error with the edit tools on this variant (never master), then call apply_next again."}
        pdf = _pdf_for(v)
        if not core.out_exists(pdf) or core.out_mtime(pdf) < core.mtime(v):
            return {"step": "5_render_pdf", "do": {"tool": "render_pdf", "args": {"variant": v}}}
        if j["route"] == "email":
            eml = _eml_for(v)
            if not eml or core.out_mtime(eml) < core.out_mtime(pdf):
                return {"step": "6_compose_email",
                        "do": {"tool": "compose_email", "args": {"variant": v, "role": "<role>", "company": "<company>", "to": j["emails"][0],
                                                                "hook": "<1-2 sentences, professional>", "key_points": ["<2-4 resume-backed points>"],
                                                                "pdf_path": pdf}},
                        "note": "fix every issue it reports, rerun until ok == true."}
            m_ = BytesParser(policy=policy.default).parsebytes(core.out_read(eml))
            body = m_.get_body(preferencelist=("plain",)).get_content()
            if _gmail_creds():
                return {"step": "7_send_email", "do": {"tool": "gmail_send_application", "args": {"job_id": job_id, "send": True}},
                        "note": "Server drafts it in Gmail with the PDF attached, sends it and logs it. Do NOT use your own Gmail connector."}
            link = public_url(pdf)
            body_link = body.replace("Best regards,", f"Resume (PDF): {link}\n\nBest regards,", 1) if "Best regards," in body else body + f"\n\nResume (PDF): {link}"
            return {"step": "7_send_with_your_gmail",
                    "do": ["With YOUR connected Gmail tool SEND (not draft-only) the email below to `to`.",
                           "If your Gmail tool can attach files, attach the PDF (get_resume_pdf). If it CANNOT attach, send `body_with_resume_link` instead - it has the public PDF link. NEVER ask the user to upload the PDF.",
                           "Then call mark_application_sent(job_id, to)."],
                    "to": j["emails"][0], "subject": m_["Subject"], "body": body, "body_with_resume_link": body_link,
                    "note": "Never claim 'sent' unless Gmail confirmed it."}
        st = _app_status(j["form_url"])
        if st in ("blocked", "not_submitted"):
            return {"step": "needs_user", "reason": f"last attempt status = {st} (CAPTCHA/login/required field/resume). Tell the user exactly what is missing; do not retry blindly.",
                    "screenshots": "see resume_mcp/out/form_*.png"}
        return {"step": "6_apply_form",
                "do": [{"tool": "get_applicant_profile", "args": {}},
                       {"tool": "inspect_application_form", "args": {"url": j["form_url"]}},
                       {"tool": "fill_application_form",
                        "args": {"url": j["form_url"], "answers": "<selector: value from profile>", "resume_pdf_path": pdf,
                                 "variant": v, "role": "<role>", "company": "<company>", "submit": True}}],
                "note": "If inspect shows 'frames', use that URL. Unknown required answers (salary, notice period, EEO, legal) -> stop and ask the user. Do not guess."}

    @mcp.tool()
    def get_applicant_profile() -> dict:
        """Facts from master for filling application forms (name, email, phone, location, links, current role, years of experience).
        Anything not listed here (salary, notice period, visa, EEO) is unknown: ask the user."""
        m = core.load("master")
        c = m.get("contact", {})
        first, _, last = m["name"].partition(" ")
        cur = m["experience"][0] if m.get("experience") else {}
        return {"full_name": m["name"], "first_name": first, "last_name": last, "email": c.get("email", ""), "phone": c.get("phone", ""),
                "location": c.get("location", ""), "links": c.get("links", []), "current_title": cur.get("title", ""),
                "current_company": cur.get("company", ""), "total_experience_years": core.experience_years(m),
                "education": m.get("education", []), "saved_answers": _answers(),
                "unknown_ask_user": [k for k in ("current_salary", "expected_salary", "notice_period", "work_authorization", "eeo_demographics") if k not in _answers()],
                "rule": "Never ask the user for experience years: total_experience_years is final (dated history). For ANY form/questionnaire question call answer_form_question(question) first. Only if answer is null, ask the user once and save with set_applicant_answer."}

    @mcp.tool()
    def answer_form_question(question: str) -> dict:
        """Deterministic answer for any application/questionnaire question (salary, notice, visa, experience years, per-skill years, contact, links, role, education).
        Always call this BEFORE asking the user. Experience is always the DATED total from master (never 5+/5.5 unless the dated history supports it).
        Returns answer=None only when truly unknown: then ask the user ONCE and save with set_applicant_answer."""
        q = question.lower()
        m = core.load("master")
        a = _answers()
        c = m.get("contact", {})
        cur = m["experience"][0] if m.get("experience") else {}
        yrs = core.experience_years(m)
        def ans(x, src): return {"answer": x, "source": src}
        mm = re.search(r"(?:experience|years?).{0,40}?(?:with|in|on|using)\s+([A-Za-z0-9+#.\- ]{2,30}?)(?:[?.,]|$)", question, re.I)
        if mm and re.search(r"year|experience", q) and not re.search(r"total|overall", q):
            t = mm.group(1).strip().lower()
            spans = [core.parse_range(j["dates"]) for j in m["experience"] if t in (" ".join(j["bullets"]) + " " + j["title"]).lower()]
            spans = sorted(x for x in spans if x)
            if spans:
                tot, cs, ce = 0, None, None
                for st, en in spans:
                    if ce is None or st > ce:
                        if ce is not None: tot += ce - cs
                        cs, ce = st, en
                    else: ce = max(ce, en)
                tot += ce - cs
                return ans(f"{round(tot / 12, 1)} years", f"dated roles mentioning '{t}'")
        rules = [
            (r"expected.*(salary|ctc|compensation|pay)|salary expectation", lambda: a.get("expected_salary")),
            (r"current.*(salary|ctc|compensation)", lambda: a.get("current_salary")),
            (r"notice", lambda: a.get("notice_period")),
            (r"visa|sponsor|work (permit|authori)|right to work|eligible to work", lambda: a.get("work_authorization")),
            (r"total.*experience|overall.*experience|years of experience|how many years|experience.*years", lambda: f"{yrs} years"),
            (r"first name", lambda: m["name"].split(" ")[0]),
            (r"last name|surname", lambda: m["name"].split(" ", 1)[-1]),
            (r"full name|^name|your name", lambda: m["name"]),
            (r"e-?mail", lambda: c.get("email")),
            (r"phone|mobile|contact number|whatsapp", lambda: c.get("phone")),
            (r"linkedin", lambda: next((l for l in c.get("links", []) if "linkedin" in l.lower()), None)),
            (r"github", lambda: next((l for l in c.get("links", []) if "github" in l.lower()), None)),
            (r"current (location|city)|where.*(located|based)|location", lambda: c.get("location")),
            (r"current (company|employer)|present (company|employer)", lambda: cur.get("company")),
            (r"current (title|designation|role)|designation|job title", lambda: cur.get("title")),
            (r"highest.*(qualification|education|degree)|education|qualification|degree", lambda: m["education"][0]["degree"] if m.get("education") else None),
            (r"relocat", lambda: a.get("willing_to_relocate")),
            (r"cover letter|why .*(join|interested|apply)|about yourself", lambda: m.get("summary")),
        ]
        for pat, fn in rules:
            if re.search(pat, q):
                v = fn()
                if v: return ans(v, "master/profile_answers")
        return {"answer": None, "reason": "unknown - ask the user once, then save it with set_applicant_answer(key, value)"}

    @mcp.tool()
    def set_applicant_answer(key: str, value: str) -> dict:
        """Save a user-confirmed answer (expected_salary, current_salary, notice_period, work_authorization, ...) so it is never asked again."""
        a = _answers(); a[core.slug(key).replace("-", "_")] = core.clean(value)
        store.set_json("answers", a)
        return {"ok": True, "saved_answers": a}

    @mcp.tool()
    def gmail_send_application(job_id: str, send: bool = True, eml_path: str = "") -> dict:
        """Put the composed application email (resume PDF attached) into Gmail Drafts, send it (send=True), remove the draft, log it.
        Recipient must be an email found in the job posting. Needs GMAIL_ADDRESS + GMAIL_APP_PASSWORD in .env."""
        j = _jload(job_id)
        v = core.slug(job_id)
        eml = os.path.basename(eml_path) if eml_path else (_eml_for(v) or "")
        if not eml or not core.out_exists(eml):
            raise ValueError("No composed email found. Run compose_email first.")
        cr = _gmail_creds()
        if not cr:
            raise ValueError("GMAIL_ADDRESS / GMAIL_APP_PASSWORD missing in resume_mcp/.env (Google Account -> Security -> App passwords). Add them and restart run.py.")
        addr, pw = cr
        msg = BytesParser(policy=policy.default).parsebytes(core.out_read(eml))
        to = [a.strip().lower() for a in re.split(r"[;,]", msg["To"] or "") if a.strip()]
        if not to or any(t not in j["emails"] for t in to):
            raise ValueError(f"Recipient {to} is not an email found in the job posting {j['emails']}.")
        if not any(p.get_content_type() == "application/pdf" for p in msg.iter_attachments()):
            raise ValueError("Email has no PDF attached. Run render_pdf then compose_email again.")
        del msg["From"]; msg["From"] = addr
        if not msg["Message-ID"]: msg["Message-ID"] = make_msgid()
        mid = msg["Message-ID"]
        imap = imaplib.IMAP4_SSL("imap.gmail.com")
        imap.login(addr, pw)
        folder = _drafts_folder(imap)
        imap.append(f'"{folder}"', "\\Draft", imaplib.Time2Internaldate(dt.datetime.now().timestamp()), bytes(msg))
        out = {"ok": True, "drafted": True, "sent": False, "to": to, "subject": msg["Subject"], "attachment": "resume PDF"}
        if send:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as sm:
                sm.login(addr, pw)
                sm.send_message(msg)
            out["sent"] = True
            try:
                imap.select(f'"{folder}"')
                typ, ids = imap.search(None, "HEADER", "Message-ID", mid)
                for i in (ids[0].split() if ids and ids[0] else []):
                    imap.store(i, "+FLAGS", "\\Deleted")
                imap.expunge()
            except Exception:
                out["note"] = "sent; could not remove the draft copy"
            j["sent"] = True
            _jsave(job_id, j)
        imap.logout()
        core.log_application({"company": j.get("title", ""), "role": "", "url": j["url"], "variant": v, "status": "emailed" if send else "drafted", "notes": ",".join(to)})
        return out

    @mcp.tool()
    def get_resume_pdf(job_id: str, include_base64: bool = False) -> dict:
        """Download link (and optionally base64) of the tailored resume PDF for this job, to attach in the user's own Gmail connector.
        `public_url` is a public link to the PDF; put it in the email body if your Gmail tool cannot attach files."""
        pdf = _pdf_for(core.slug(job_id))
        if not core.out_exists(pdf):
            raise ValueError("PDF not rendered yet - call apply_next.")
        data = core.out_read(pdf)
        out = {"filename": pdf, "bytes": len(data), "public_url": public_url(pdf)}
        if include_base64:
            out["base64"] = base64.b64encode(data).decode()
        return out

    @mcp.tool()
    def mark_application_sent(job_id: str, to: str = "", note: str = "") -> dict:
        """Call AFTER your Gmail connector actually sent the email. Marks the job done and logs it in the tracker."""
        j = _jload(job_id)
        j["sent"] = True
        _jsave(job_id, j)
        core.log_application({"company": j.get("title", ""), "role": "", "url": j.get("url", ""), "variant": core.slug(job_id), "status": "emailed", "notes": f"{to} {note}".strip()})
        return {"ok": True}
