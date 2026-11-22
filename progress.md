# Resume Studio MCP - Progress

_Last updated: 2026-11-22_

## AGENT QUICKSTART (LLM: read this first)
User says `apply <link>` -> use ONLY the `resume-studio` MCP:
1. `apply_start(url)` -> returns `job_id`, JD text, recruiter emails, route (`email` or `form`)
2. Loop `apply_next(job_id)` -> do exactly the returned action -> call `apply_next` again. Stop at `done` or `needs_user`.
- If `apply_start` returns `llm_fetch_needed` (site blocks the server): open the URL with YOUR OWN browsing tool, copy the full JD, call `apply_start(url, jd_text=...)`. Do not ask the user.
3. Reply with ONE line. Do not explain. Do not improvise order. Never edit `master`. Never add skills/metrics not in master.
- Email route: `compose_email` -> `gmail_send_application` (server sends via Gmail SMTP with PDF attached, when GMAIL creds are in `.env`). Fallback only if apply_next says so: `get_resume_pdf` -> send with YOUR OWN Gmail connector (PDF attached; if it can't attach, use `body_with_resume_link` = public unguessable PDF link `/dl/<sig>/<name>.pdf`; never ask the user to upload) -> `mark_application_sent`. Pasted JD with email works: `apply_start(jd_text=...)` (url optional). Gmail logic is back in the server (`gmail_send_application`); creds in `.env`.
- Form route: `get_applicant_profile` -> `inspect_application_form` -> `fill_application_form(submit=True)`; stops on CAPTCHA/login/missing required fields; never guess salary/notice/EEO/legal answers (ask user)
- State lives on disk (`jobs/<job_id>.json`, `variants/`, `out/`, `applications.json`), so `apply_next` always knows the next step, even after a restart.

## Vercel deploy (current hosting; Cloudflare tunnels removed)
- [x] Serverless-ready: `api/index.py` (ASGI entry), `vercel.json` (rewrite all -> /api/index), `.vercelignore`, `run.py` = local dev only
- [x] `store.py`: Upstash Redis on Vercel (`KV_REST_API_URL`/`KV_REST_API_TOKEN`), plain files locally. Keys: master, variant/, hist/, job/, apps, answers, out/
- [x] PDF via reportlab (`pdfgen.py`), no Chromium; per-request stateless MCP session (works without ASGI lifespan)
- [x] PDFs/screenshots served at `/dl/<sig>/<file>` (unguessable, does not expose the MCP secret); `PUBLIC_BASE_URL` or Vercel URL used for links
- [x] `smoke_test.py` passes end to end on files + fake-Redis backends
- [ ] Deploy: import repo in Vercel, Root Directory = `resume_mcp`; add Upstash Redis (Storage tab); env vars `RESUME_SECRET`, `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD` (+ optional `BROWSER_WS_ENDPOINT`)
- [ ] Connector URL: `https://<project>.vercel.app/<RESUME_SECRET>/mcp`; check `https://<project>.vercel.app/healthz` -> `ok redis`
- [ ] Form tools on Vercel need `BROWSER_WS_ENDPOINT` (remote Chromium, e.g. Browserless/Browserbase). Without it they return a clear error; email route is unaffected
- [x] `playwright` removed from requirements.txt (form tools off on Vercel; email route unaffected)
- [ ] Keep the GitHub repo PRIVATE (master_resume.json + profile_answers.json are deployed)

## Done
- [x] Master resume converted from PDF to structured `master_resume.json` (name, headline, contact, summary, skills by category, experience, education, certifications, awards, languages)
- [x] Redesigned as a deterministic MCP: no LLM inside; the client LLM calls the tools
- [x] `core.py`: storage, undo history (last 30 snapshots), validation, JD matching, bullet ranking, application tracker
- [x] `server.py`: 33 tools
  - Read: `list_variants`, `get_resume`, `validate_resume`, `fetch_jd`, `analyze_jd`, `rank_bullets_for_jd`, `history`
  - Variants: `create_variant`, `delete_variant`, `undo`
  - Edit: `set_basics`, `set_skill_category`, `add_skills`, `remove_skills`, `delete_skill_category`, `reorder_skill_categories`, `optimize_skills_for_jd`, `add_job`, `update_job`, `delete_job`, `set_bullets`, `add_bullet`, `update_bullet`, `remove_bullet`, `reorder_bullets`, `set_education`, `set_list_section`
  - Output: `render_pdf`, `compose_email`
  - Apply: `inspect_application_form`, `fill_application_form`, `log_application`, `list_applications`
- [x] Truthfulness guards: variants limited to master skills/metrics/employers; dates immutable; years-of-experience claims checked against dated history; master edits need `confirm_master=True`
- [x] Render gate: PDF blocked while validation errors exist; PDF text re-extracted and cross-checked
- [x] Safety: `fill_application_form` never submits unless `submit=True`; stops on CAPTCHA/login; uploads only PDFs from `out/`; SSRF guard on URLs
- [x] Smoke test passed end to end (variant, JD analysis, skills, PDF 2 pages, email, guards)
- [x] Test prompts written (full flow, guards, edit/undo, apply dry-run, tracker)

## One-command apply (added 2026-11-22)
- [x] `apply_flow.py` (existing tools untouched): `apply_start`, `apply_next` (disk-derived state machine), `get_applicant_profile`, `get_resume_pdf`, `mark_application_sent` 
- [x] `server.py`: registers apply_flow; INSTRUCTIONS start with one-command mode
- [x] `.env.example`: `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD`
- [x] Gmail logic restored (`gmail_send_application`); public PDF link kept as fallback
- [ ] User: re-add `GMAIL_ADDRESS` + `GMAIL_APP_PASSWORD` to `.env`, restart `run.py`
- [ ] Test `apply <link>` on a real JD with an email, and one with a form

## Open
- [x] Tailoring never removes: `optimize_skills_for_jd` = union + JD skills first; new `sync_experience_for_jd` adds master bullets that evidence JD skills; skill warning threshold 90; steps gated by flags `jd_tailored`/`exp_synced`
- [x] `answer_form_question(question)`: deterministic answers for salary/notice/visa/experience/per-skill years/contact/links/role/education; experience = dated total (4.2), no more 5 vs 5.5 questions
- [x] Saved answers in `profile_answers.json` (expected salary AED 20,000/month, notice 30 days, visa sponsorship needed); `get_applicant_profile` returns them, `set_applicant_answer` adds more
- [ ] Google Forms (forms.gle) link: form tools may not read it; test, else fill manually
- [ ] NaukriGulf/LinkedIn/Bayt block the Codespace IP or need login. `apply_start(url, jd_text=<pasted JD>)` now works as a fallback; job boards go email route only, else `needs_user`
- [ ] Master resume claims "5+ years" but dated experience = 4.2 years -> validation error blocks rendering until fixed
- [ ] Add connector in Claude (Settings -> Connectors -> Add custom connector) and run the test prompts
- [ ] Run Test 4 (apply dry-run) on a real Greenhouse/Lever/Ashby link; form tools not yet tested on a live page
- [ ] Fill `contact.links` (LinkedIn / GitHub) in master
- [ ] Trim master bullets (V2 Solutions has 12, one SimSol bullet is 353 chars)
- [ ] Label MCA (Aug 2022 - Sep 2024) as part-time/distance if true (overlaps full-time role)
- [ ] Decide on keeping `smoke_test.py` or removing it

## Known limits
- Workday, LinkedIn Easy Apply, CAPTCHAs, login walls: apply manually
- Master edits (confirm_master) are saved in Redis and override the shipped master_resume.json
- The secret URL is the only authentication; rotate by changing `RESUME_SECRET` in Vercel and redeploying

## Files
- `server.py`, `core.py`, `store.py`, `pdfgen.py`, `apply_flow.py`, `api/index.py`, `vercel.json`, `run.py` (local), `jobs/` (local only)
- `master_resume.json`
- `variants/`, `history/`, `out/`, `applications.json`
- `.env`, `.env.example`, `requirements.txt`, `README.md`, `smoke_test.py`
