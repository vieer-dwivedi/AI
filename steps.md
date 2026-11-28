# 🪜 steps.md: Build DispatchAgent Step by Step

This is the hands-on build guide that goes with the [README](README.md). The README explains **what** the project is and **why**; this file explains **how to build it, in order**.

Each step has the same shape:

- **Goal**: what you're achieving
- **Files**: where the code lives (the folders already exist in this repo as stubs)
- **Do**: the concrete work
- **Done when**: a check you can run to know the step is finished

> 💡 Work top to bottom. Every step builds on the previous one, and each ends in something you can run, so you always have a working demo to show.

## Progress tracker

- [ ] [Step 0: Prerequisites and repo setup](#step-0-prerequisites-and-repo-setup)
- [ ] [Step 1: Config and environment](#step-1-config-and-environment)
- [ ] [Step 2: Synthetic data (the fake customer)](#step-2-synthetic-data-the-fake-customer)
- [ ] [Step 3: Mock carrier API](#step-3-mock-carrier-api)
- [ ] [Step 4: ERP tools (MCP server)](#step-4-erp-tools-mcp-server)
- [ ] [Step 5: Carrier tools (MCP server)](#step-5-carrier-tools-mcp-server)
- [ ] [Step 6: Contract knowledge base (RAG)](#step-6-contract-knowledge-base-rag)
- [ ] [Step 7: Comms tools](#step-7-comms-tools)
- [ ] [Step 8: Schemas and policy engine](#step-8-schemas-and-policy-engine)
- [ ] [Step 9: The agent loop (orchestrator)](#step-9-the-agent-loop-orchestrator)
- [ ] [Step 10: API service, memory and approvals](#step-10-api-service-memory-and-approvals)
- [ ] [Step 11: Ops console (Streamlit)](#step-11-ops-console-streamlit)
- [ ] [Step 12: Observability](#step-12-observability)
- [ ] [Step 13: Evaluation harness](#step-13-evaluation-harness)
- [ ] [Step 14: Docker packaging](#step-14-docker-packaging)
- [ ] [Step 15: Run the demo](#step-15-run-the-demo)
- [ ] [Step 16: Harden and hand over](#step-16-harden-and-hand-over)

## Step-to-folder map

| Step | Folder / file |
|---|---|
| 1 | `pyproject.toml`, `.env.example`, `app/config.py` |
| 2 | `data/seed.py`, `data/playbook.md` |
| 3 | `mock_services/carrier_api/` |
| 4-5, 7 | `app/tools/` |
| 6 | `ingest/contracts.py`, `app/tools/kb_server.py` |
| 8 | `app/agent/schemas.py`, `app/agent/policy.py`, `policy.yaml` |
| 9 | `app/agent/orchestrator.py`, `app/agent/prompts.py` |
| 10 | `app/main.py`, `app/agent/memory.py`, `app/cli.py` |
| 11 | `ui/console.py` |
| 12 | `app/observability/` |
| 13 | `evals/`, `tests/` |
| 14 | `Dockerfile`, `docker-compose.yml`, `Makefile` |
| 15-16 | `docs/` |

---

## Step 0: Prerequisites and repo setup

**Goal:** a clean machine-ready starting point.

**Do**

1. Install Python 3.11+, Git, and (optional) Docker.
2. Get an Anthropic API key from the Anthropic Console.
3. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
4. Confirm `.gitignore` is in place so `.env`, databases, and traces never get committed.

**Done when:** `python --version` prints 3.11 or higher and `git status` shows no `.env` or `.venv` files.

---

## Step 1: Config and environment

**Goal:** one place for every setting, loaded from the environment.

**Files:** `pyproject.toml`, `.env.example`, `app/config.py`

**Do**

1. Install dependencies: `pip install -e ".[dev]"`.
2. Copy the example env file: `cp .env.example .env`, then add your `ANTHROPIC_API_KEY`.
3. In `app/config.py`, create a `Settings` class using `pydantic-settings` that reads: API key, model names, `MAX_STEPS`, `MAX_COST_USD_PER_CASE`, database URL, carrier API URL, and Chroma path.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-5-5"
    anthropic_triage_model: str = "claude-haiku-5-5"
    max_steps: int = 15
    max_cost_usd_per_case: float = 0.75
    database_url: str = "sqlite:///data/erp.db"
    carrier_api_url: str = "http://localhost:8001"
    chroma_path: str = "./data/chroma"

settings = Settings()
```

**Done when:** `python -c "from app.config import settings; print(settings.anthropic_model)"` prints the model name.

---

## Step 2: Synthetic data (the fake customer)

**Goal:** a believable, reproducible "Meridian Components" dataset, so there is no real customer data anywhere.

**Files:** `data/seed.py`, `data/playbook.md`

**Do**

1. Define the SQLAlchemy tables: `customers`, `products`, `inventory`, `orders`, `order_lines`, `shipments`, `contracts`, `cases`, `approvals`, `tool_call_log`.
2. Seed with a **fixed random seed** (for example `random.seed(42)`) so every run produces identical data.
3. Generate roughly: 25 customers, 60 SKUs, 400 orders, 3 warehouses (Pune, Chennai, Jebel Ali).
4. Generate 25 contract PDFs (use `reportlab` or `fpdf2`) with varied **grace days** and **penalty %/day** clauses.
5. Write about 40 inbound emails as JSON in `data/emails/`: delays, damage, disputes, plus **5 adversarial emails** containing prompt-injection text.
6. Write `data/playbook.md`: the company's informal rules (for example "Tier-1 delays over 3 days: always consider air freight").
7. **Plant the 6 demo scenarios** from the README as specific orders and emails, so they behave predictably.

**Done when:** `python data/seed.py` creates `data/erp.db`, and `sqlite3 data/erp.db "select count(*) from orders;"` returns about 400.

---

## Step 3: Mock carrier API

**Goal:** simulate a third-party REST integration, including its flaws.

**Files:** `mock_services/carrier_api/main.py`

**Do**

1. Build a small FastAPI app with:
   - `GET /shipments/{tracking_id}`: status, ETA, last event, delay reason
   - `POST /quotes`: returns road, air, and sea quotes (price, ETA)
2. Add realism: random 100-400 ms latency, occasional `503` errors, and one "unknown tracking ID" case (for Scenario 6).
3. Make output deterministic where scenarios need it.

**Done when:** `uvicorn mock_services.carrier_api.main:app --port 8001` runs and `curl localhost:8001/shipments/TRK-48213` returns JSON.

---

## Step 4: ERP tools (MCP server)

**Goal:** give the agent read access to the ERP through a standard tool interface.

**Files:** `app/tools/erp_server.py`

**Do**

1. Use FastMCP from the `mcp` SDK to define the tools: `get_order`, `get_customer`, `check_inventory`, `calculate_penalty`.
2. Every tool takes a **typed, validated** input and returns **compact JSON** (only fields the agent needs).
3. Implement `calculate_penalty` as **pure deterministic code**:
   ```python
   def calculate_penalty(order_value: float, delay_days: int,
                         grace_days: int, pct_per_day: float, cap_pct: float) -> float:
       billable_days = max(0, delay_days - grace_days)
       return round(min(order_value * pct_per_day / 100 * billable_days,
                        order_value * cap_pct / 100), 2)
   ```
4. Add unit tests in `tests/test_erp_tools.py`, including edge cases (zero delay, delay inside grace period, cap reached).

**Done when:** `pytest tests/test_erp_tools.py` passes.

---

## Step 5: Carrier tools (MCP server)

**Goal:** wrap the carrier REST API as agent tools.

**Files:** `app/tools/carrier_server.py`

**Do**

1. Define `get_shipment_status` and `get_freight_quote` using `httpx`.
2. Add **retries with backoff** for 5xx errors and a **timeout**.
3. Return clear error objects (for example `{"error": "TRACKING_NOT_FOUND"}`) so the agent can reason about failures instead of crashing.

**Done when:** a tool call for the unknown tracking ID returns a structured error, not an exception.

---

## Step 6: Contract knowledge base (RAG)

**Goal:** let the agent find the right penalty clause in the right customer's contract.

**Files:** `ingest/contracts.py`, `app/tools/kb_server.py`

**Do**

1. Extract text from each contract PDF with `pdfplumber`.
2. **Chunk by clause** (split on clause numbers like `7.2`) rather than by fixed size, which keeps each clause intact.
3. Embed and store in ChromaDB with metadata: `customer_id`, `contract_id`, `clause_no`.
4. Expose `search_contracts(customer_id, query)` that **filters by customer** and returns top-k clauses with citations. Never allow cross-customer retrieval.
5. Do the same for `data/playbook.md` and expose `search_playbook(query)`.

**Done when:** `search_contracts("CUST-007", "late delivery penalty")` returns the correct clause with its clause number.

---

## Step 7: Comms tools

**Goal:** the agent can draft and send messages, safely.

**Files:** `app/tools/comms_server.py`

**Do**

1. `draft_customer_email(case_id, subject, body)` saves a draft to the database (low risk).
2. `send_email(draft_id)` marks the draft as sent in the demo (writes to an outbox table; no real SMTP).
3. `add_case_note`, `update_order_status`, `create_reship_order`, `book_expedited_freight`, and `issue_credit_note` go here or in the ERP server (write tools).

**Done when:** a draft can be created and shows up in the `drafts` table.

---

## Step 8: Schemas and policy engine

**Goal:** a deterministic safety layer that sits between the agent and every write action.

**Files:** `app/agent/schemas.py`, `app/agent/policy.py`, `policy.yaml`

**Do**

1. In `schemas.py`, define `ResolutionPlan`, `Option`, `PlannedAction`, and `Citation` with Pydantic (see README section 5.4).
2. In `policy.py`, load `policy.yaml` and implement:
   ```python
   class Decision(str, Enum):
       ALLOW = "allow"
       NEEDS_APPROVAL = "needs_approval"
       DENY = "deny"

   def evaluate(tool_name: str, args: dict, case: CaseState) -> Decision: ...
   ```
3. ⚠️ **Do not use Python's `eval()`** for the `require_approval_if` expressions. Use a safe evaluator such as `simpleeval`, or express rules as structured data (`{field, op, value}`).
4. Rules to cover: monetary thresholds, customer tier, unknown tools (default deny), step and cost budgets.
5. Test every rule in `tests/test_policy.py`. This is the most important test file in the project.

**Done when:** `pytest tests/test_policy.py` passes, including "unknown tool is denied" and "₹60,000 expedite needs approval".

---

## Step 9: The agent loop (orchestrator)

**Goal:** the heart of the system: Claude reasons, calls tools, and is checked by the policy engine.

**Files:** `app/agent/orchestrator.py`, `app/agent/prompts.py`

**Do**

1. **Triage first.** A single cheap call with the small model classifies exception type and urgency.
2. **Build the main loop** with the `anthropic` SDK's tool use:
   ```python
   for step in range(settings.max_steps):
       resp = client.messages.create(
           model=settings.anthropic_model,
           system=SYSTEM_PROMPT,
           messages=messages,
           tools=tool_schemas,
           max_tokens=2048,
       )
       messages.append({"role": "assistant", "content": resp.content})
       if resp.stop_reason != "tool_use":
           break
       results = []
       for block in resp.content:
           if block.type == "tool_use":
               decision = policy.evaluate(block.name, block.input, case)
               results.append(run_or_park(block, decision))   # execute, queue approval, or deny
       messages.append({"role": "user", "content": results})
   ```
3. **Wrap emails as untrusted data** in the first user message:
   ```text
   <untrusted_email>
   ...raw email text...
   </untrusted_email>
   ```
   and state in the system prompt that nothing inside those tags is an instruction.
4. **Finish with a tool.** Give the agent a `submit_resolution` tool whose input schema is `ResolutionPlan`. Validate it with Pydantic; if validation fails, return the error to the model so it can fix it.
5. **Pause and resume.** When a call needs approval, persist the loop state and return. After approval, resume from the saved messages.
6. **Budget checks** every iteration: steps, tokens, and estimated cost. Stop and escalate if exceeded.

**Done when:** `python -m app.cli run-scenario 1` produces a valid `ResolutionPlan` that cites the contract clause and the freight quote.

---

## Step 10: API service, memory and approvals

**Goal:** wrap the agent in a service other systems (and the UI) can call.

**Files:** `app/main.py`, `app/agent/memory.py`, `app/cli.py`

**Do**

1. Endpoints:
   - `POST /exceptions`: submit an email or event, returns a `case_id`
   - `GET /cases/{id}`: status, plan, and trace
   - `GET /approvals`: pending approvals
   - `POST /approvals/{id}`: approve or reject, then resume the case
2. In `memory.py`, persist case state (messages, pending tool calls, status) so cases survive restarts.
3. Run agents in a **background task** so the API returns immediately.
4. `app/cli.py`: `run-scenario N` for quick demos without the UI.

**Done when:** `curl -X POST localhost:8000/exceptions ...` returns a `case_id`, and `GET /cases/{id}` eventually shows a plan.

---

## Step 11: Ops console (Streamlit)

**Goal:** the human side of human-in-the-loop.

**Files:** `ui/console.py`

**Do**

1. **Inbox tab**: list of cases with status and urgency.
2. **Case tab**: the original email, the agent's plan, options with cost versus penalty, and cited evidence.
3. **Approvals tab**: approve or reject each gated action, with an optional comment (the comment is stored; it feeds the eval set later).
4. **Trace tab**: a timeline of every step the agent took.

**Done when:** you can run Scenario 1, see the plan, click Approve, and watch the case close.

---

## Step 12: Observability

**Goal:** make the agent inspectable, which is what earns customer trust.

**Files:** `app/observability/tracing.py`, `app/observability/cost.py`

**Do**

1. Write one JSONL line per event (LLM call, tool call, policy decision, approval) to `traces/<case_id>.jsonl` with timestamps and token counts.
2. Add OpenTelemetry spans around `agent.loop`, `llm.call`, and `tool.call`.
3. In `cost.py`, compute estimated USD per case from token usage and a price table in config. Surface it in the console.

**Done when:** after a run, `traces/<case_id>.jsonl` exists and the Trace tab renders it.

---

## Step 13: Evaluation harness

**Goal:** turn "it seems to work" into numbers, and protect against regressions.

**Files:** `evals/cases/`, `evals/run_evals.py`, `evals/judge.py`, `tests/`

**Do**

1. Write 30 golden cases as JSON files: input, expected exception type, required tool calls, acceptable resolutions, forbidden actions.
2. `run_evals.py` runs every case against a **fresh seeded database** and records the outputs.
3. Score with **deterministic checks first** (classification, required tools, forbidden actions, policy violations). Use `judge.py` (LLM-as-judge with a written rubric) only for the fuzzy "is this resolution reasonable" question.
4. Write a Markdown report to `evals/reports/`.
5. Make the adversarial suite (prompt injection, missing data, conflicting evidence) a separate command: `make eval-safety`.

**Done when:** `make eval` produces a report showing **0 policy violations** and 100% on the injection suite.

---

## Step 14: Docker packaging

**Goal:** one command to run everything.

**Files:** `Dockerfile`, `docker-compose.yml`, `Makefile`

**Do**

1. One image for the app. Compose services: `api` (8000), `carrier` (8001), `console` (8501).
2. Mount `data/` as a volume and pass secrets through `env_file: .env`. Never bake the API key into the image.
3. Add Makefile targets: `setup`, `seed`, `run`, `test`, `eval`, `eval-safety`, `clean`.

**Done when:** `docker compose up --build` starts all three services and the console loads at http://localhost:8501.

---

## Step 15: Run the demo

**Goal:** a smooth 8-minute walkthrough.

**Do**

1. Follow the demo script in the README (section 14).
2. Rehearse scenarios 1 and 5 until they are smooth. Scenario 5 (prompt injection) is the one people remember.
3. Optional: record a short screen capture and add it to `docs/`.

**Done when:** you can run the full demo twice in a row without touching the code.

---

## Step 16: Harden and hand over

**Goal:** behave like an FDE, not just a prototype builder.

**Files:** `docs/discovery-notes.md`, `docs/runbook.md`

**Do**

1. **Discovery notes:** write up the workflow map, pain points, and success metrics as if from a real customer interview.
2. **Runbook:** how to start, stop, monitor, rotate keys, add a new tool, add a new policy rule, and read a trace.
3. **Hardening checklist:** authentication on the API, RBAC for approvers, PII redaction in traces, rate limits, secrets in a vault, and data-residency review.
4. **Handover:** list what the customer's team owns, what you own, and the expansion backlog.

**Done when:** a new engineer could run, debug, and extend the system using only the docs.

---

## Suggested pacing

| Block | Steps | Result |
|---|---|---|
| Day 1 | 0-3 | Data and mock services running |
| Day 2 | 4-8 | All tools and the policy engine tested |
| Day 3 | 9-10 | Agent solves Scenario 1 end to end |
| Day 4 | 11-12 | Console and traces working |
| Day 5 | 13-16 | Evals, Docker, polished demo and docs |
