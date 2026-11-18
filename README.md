# AI DevOps Incident Response Agent

An agent team that triages a cloud alert, finds the root cause, and **proposes** fixes with
exact commands, so an engineer can approve them instead of starting from a blank page.

```
alert (CloudWatch / PagerDuty / Alertmanager)
   -> intake -> triage -> diagnosis -> recall -> remediation -> report
                  |           |           |            |
              severity    root cause   past cases   proposed fixes
              P1-P4       + evidence   (ChromaDB)   (human approves)
```

* **LangGraph** runs the workflow (`agents/orchestrator.py`) and routes it (P4 noise skips the LLM agents).
* **Deep agents** (`create_deep_agent`) power the three specialists (triage / diagnosis / remediation).
* **`init_chat_model`** builds the LLM from `.env`, so **any provider** works with no code change.
* Tools: CloudWatch alarms/logs/metrics, ECS/RDS/EC2, log pattern analysis, runbook RAG, Slack, Azure (optional).
* Memory: short-term notebook per incident, long-term past incidents in ChromaDB.

## Quick start

```bash
cd incident-response-agent
python main.py check            # shows your config, builds the model
python main.py list             # the 4 sample AWS incidents
python main.py run ecs_oom --show
python main.py run all
python main.py eval             # score against known incidents
python main.py serve            # webhook API on http://127.0.0.1:8000
python -m pytest                # all tests
```

Reports are written to `out/reports/<incident-id>.md`; Slack messages go to `out/slack_outbox.jsonl`
until you set `SLACK_WEBHOOK_URL`.

## Choosing the LLM (`.env`)

The default `LLM_MODEL=fake` is an **offline rule-based test brain** (no key, no internet). It reads the real
tool outputs and decides the next step, so the whole pipeline is testable, but it is not a real LLM.
For real reasoning, change one line:

```
LLM_MODEL=anthropic:claude-sonnet-4-5      # + ANTHROPIC_API_KEY=...
LLM_MODEL=openai:gpt-4o                    # + OPENAI_API_KEY=...  (pip install langchain-openai)
LLM_MODEL=google_genai:gemini-2.5-flash    # + GOOGLE_API_KEY=...
LLM_MODEL=ollama:llama3.1                  # local; pip install langchain-ollama
LLM_MODEL=openai:my-model  LLM_BASE_URL=http://localhost:8000/v1   # any OpenAI-compatible server
```

Then run `python main.py check --ping` to confirm the key works, and `python main.py eval` to see how that
model scores. Use `TRIAGE_LLM_MODEL`, `DIAGNOSIS_LLM_MODEL`, `REMEDIATION_LLM_MODEL` to give each agent a different model.
Model names change often, so use whatever your provider currently offers.

## Safety design

* Agents **propose**; they never execute by default (`AUTO_REMEDIATE=false`).
* High-risk actions are always blocked; execution on real AWS is not implemented (simulated on the mock backend only).
* Diagnosis is read-only. Severity P1-P4 is computed by fixed rules, not by the LLM.
* External MCP tools are attached only if listed in `MCP_TOOL_ALLOWLIST`.
* Webhook can require `X-Webhook-Secret`. A saved incident in long-term memory is marked `UNVERIFIED` until a human confirms the fix.

## Webhook API

```bash
python main.py serve
curl -X POST localhost:8000/webhook/manual -H 'content-type: application/json' \
  -d '{"id":"INC-1","title":"High 5xx on payments-service","service":"payments-service"}'
curl localhost:8000/incidents/INC-1
```
Sources: `/webhook/pagerduty`, `/webhook/alertmanager`, `/webhook/cloudwatch` (SNS), `/webhook/manual`.

## MCP

`python main.py mcp` exposes the read-only tools plus `investigate_scenario` to any MCP client.
To plug in your **own** MCP server, set `MCP_SERVERS_JSON` and `MCP_TOOL_ALLOWLIST` in `.env`
(the diagnosis agent then gets those tools), or add `@mcp.tool()` functions in `mcp_server/server.py`.

## Sample AWS data

`sample_data/scenarios/*.json` hold CloudWatch-shaped alarms, JSON logs, metrics, ECS/RDS/EC2 descriptions and
deployments for 4 incidents (ECS OOM, RDS connection exhaustion, EC2 disk full, bad deploy). Regenerate with
`python sample_data/generate_sample_data.py`. `CLOUD_BACKEND=aws` uses boto3 against real AWS; that path
is written against the boto3 docs but has **not** been tested on a live account, so try a sandbox first.

## Docker

```bash
docker compose up --build
```

## Layout

```
agents/   orchestrator (LangGraph) + triage / diagnosis / remediation agents
tools/    aws, azure, log analyzer, runbook RAG, slack, remediation, mcp loader
memory/   short_term.py (per incident), long_term.py (ChromaDB)
core/     config (.env), llm (init_chat_model), embeddings, schemas, scenarios, fake_llm
api/      webhook.py     mcp_server/  server.py     evals/  scenarios + scorer
runbooks/ sample_data/   tests/
```
