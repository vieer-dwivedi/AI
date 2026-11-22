# Resume Studio MCP (no LLM inside) - Vercel

Deterministic MCP server: stores, edits, validates, renders (reportlab PDF), emails (Gmail SMTP) and applies. Refuses anything not backed by `master_resume.json`.

## Deploy
1. Push to a PRIVATE GitHub repo. Vercel -> Add New Project -> import -> **Root Directory = resume_mcp**.
2. Vercel -> Storage -> add **Upstash Redis** (injects `KV_REST_API_URL`, `KV_REST_API_TOKEN`).
3. Environment variables: `RESUME_SECRET` (16+ chars), `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD`. Optional: `BROWSER_WS_ENDPOINT` (remote Chromium for form filling), `PUBLIC_BASE_URL`.
4. Deploy. Connector URL: `https://<project>.vercel.app/<RESUME_SECRET>/mcp` (health: `/healthz`).

## Local
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python run.py          # http://127.0.0.1:<RESUME_PORT>/<secret>/mcp
    .venv/bin/python smoke_test.py   # end-to-end test, no network

## Use
Say `apply <link>` (or paste the JD). The server's instructions drive `apply_start` -> `apply_next` loop.

## Safety
- Variants only use skills/metrics/employers from master; tailoring only adds/reorders, never removes.
- render_pdf is blocked while validation has errors; PDF text is cross-checked.
- Emails only go to addresses found in the job posting; form filling never submits through CAPTCHA/login or with empty required fields.
