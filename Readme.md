# Learning Project: LangChain / LangGraph Agents

A learning project that builds AI agents step by step on a local PC, with a professional structure.
Today it contains a console chat, local conversation storage, a **Dasha (Vedic astrology) MCP server**,
a **Dasha agent**, and an **orchestrator** that decides which agent should answer each message.

## How it fits together

```
 user ──► orchestrator.py ──► agents/core/graph.py (LangGraph)
                                  │
                         route node (LLM router)
                          │                 │
              picks an agent                none
                          │                 │
                 agent_<name> node      direct node (general chat)
                          │
              agents/dasha.py (Dasha agent) ──HTTP──► mcp/dasha (Dasha MCP server, PyJHora)
                                            runs SEPARATELY, default http://127.0.0.1:8000/mcp
```

- The **router** reads the conversation and the *description* of every registered agent, then picks one agent or `none`.
- A follow-up like "1990-05-15 10:30 Mumbai" after the Dasha agent asked for birth details goes back to the same agent.
- If no agent fits, the **direct** node answers as a normal assistant.
- Each agent keeps its own memory per conversation thread.

## Project structure

```
project/
├── .env                    # model config + API keys (never commit, never print)
├── Readme.md               # this file
├── progress.md             # running log of work, read this first when resuming
├── requirements.txt        # all dependencies (install manually, see Setup)
├── main.py                 # MAIN ENTRY POINT: console chat through the orchestrator, saved to SQLite per user id
├── orchestrator.py         # PLUGINS list + create_orchestrator(); also a simple console chat without a database
├── agents/
│   ├── dasha.py            # Dasha agent (LangGraph) + `dasha_plugin()` that plugs into the orchestrator
│   ├── core/
│   │   ├── llm.py          # chat model built from .env
│   │   ├── registry.py     # AgentSpec + AgentRegistry (the plug-in contract)
│   │   ├── nodes.py        # orchestrator nodes: router, direct answer, agent-node factory
│   │   └── graph.py        # builds the orchestrator graph (nodes + edges) and the Orchestrator class
│   ├── state/              # SQLite chat storage (createdb.py, chat.db); checkpoint.py is a placeholder
│   └── tests/              # try_* scripts and tests for agents
└── mcp/
    ├── dasha/              # Dasha MCP server (13 tools), engine, state, tests
    └── kp/                 # KP (Krishnamurti Paddhati) MCP server (43 tools): Swiss Ephemeris + local KP tables, no web APIs;
                            # progress.md in this folder has the design, research notes and known gaps
```

> The `mcp/` folder must **not** contain an `__init__.py` (it would shadow the official `mcp` SDK).
> Run the server as a script: `python mcp/dasha/main.py`.

## Setup

1. Create a `.env` with `MODEL_NAME`, `MODEL_PROVIDER`, `MODEL_TEMPERATURE`, `MODEL_MAX_TOKENS` and the API key
   for your provider (example: Gemini `gemini-2.5-flash`, provider `google_genai`).
   Optional: `DASHA_MCP_URL` (default `http://127.0.0.1:8000/mcp`) if the Dasha server runs elsewhere.
2. Install dependencies yourself in your virtual environment: `pip install -r requirements.txt`
   (the project tooling only edits `requirements.txt`, it never installs).

## Run

The **Dasha MCP server runs separately**. Start it first in its own terminal and leave it running:

```
python mcp/dasha/main.py --http                  # http://127.0.0.1:8000/mcp
python mcp/dasha/main.py --http --port 9000      # another port (then set DASHA_MCP_URL=http://127.0.0.1:9000/mcp)
```

Then, in a second terminal from the project root with the venv active:

| What | Command |
|---|---|
| **Main chat** (orchestrator + agents, saved to SQLite) | `python main.py` |
| Orchestrator console chat (no database) | `python orchestrator.py` |
| Dasha agent alone | `python -m agents.dasha` |
| Dasha MCP server (separate terminal, HTTP) | `python mcp/dasha/main.py --http [--host H] [--port P]` |
| Dasha MCP server (stdio, for clients that launch it) | `python mcp/dasha/main.py` |
| Dasha offline tests | `python mcp/dasha/test/test_dasha_offline.py` |
| KP MCP server (separate terminal, HTTP, port 8002) | `python mcp/kp/main.py --http [--host H] [--port P]` (env `KP_MCP_HOST` / `KP_MCP_PORT`) |
| KP MCP server (stdio) | `python mcp/kp/main.py` |
| KP tests (447 checks, real Swiss Ephemeris) | `python mcp/kp/test/test_kp.py` |
| KP server through a real MCP client (stdio) | `python mcp/kp/test/try_client.py` |
| Agent <-> separate server test (no LLM) | `DASHA_MCP_URL=http://127.0.0.1:8765/mcp python -m agents.tests.test_dasha_http` (server started on that port first) |
| Orchestrator tests (no LLM calls) | `python -m agents.tests.test_orchestrator` |
| `main.py` flow test (temp DB, no LLM) | `python -m agents.tests.test_main_flow` |

`main.py` asks for a user id, then starts a new conversation or resumes the latest one. The conversation id is the
orchestrator thread. On resume, earlier turns are loaded so the router has context; an agent's own memory (for example
saved birth details in the Dasha agent) lives in memory only and is not restored, so the agent may ask for it again.
A turn is saved only after it succeeds (a quota error leaves nothing half-saved).

Console commands in the orchestrator: `/new` (fresh conversation), `/agents` (list agents), `exit`.

If the Dasha server is not running, `main.py` / `orchestrator.py` stop at startup with a message that tells you the
command to start it. `--host 0.0.0.0` exposes the server to the network **without authentication**; keep the default
`127.0.0.1` unless you trust the network.

## KP server: tools and options
Tools by group (all facts and classical KP rules, never predictions; `lookup_place` is the only one that needs internet).
- **Reference:** `get_kp_sub_table`, `lookup_longitude`, `get_ayanamsa`, `list_event_types`, `get_house_themes`, `get_timezone_info`, `lookup_place`.
- **Natal chart:** `get_kp_chart` (`detail` basic / standard / full), `get_kp_summary`, `get_kp_report` (markdown), `get_significators`, `get_planet_significations`, `get_cuspal_sublord_analysis`, `get_cuspal_interlinks`, `judge_event`, `get_fortuna`, `compare_births`, `get_longevity_facts`, `get_kp_annual_chart`, `get_health_facts`.
- **Dasha / timing:** `get_kp_dasha`, `get_current_kp_dasha`, `get_dasha_significations`, `find_event_periods`, `analyze_joint_period`, `get_fruitful_significators`, `verify_event_day`, `find_double_transit`, `find_stations`, `find_transit_triggers`, `kp_judgment_workflow`.
- **Moment / horary:** `get_ruling_planets`, `get_ruling_planets_timeline`, `cast_horary_chart`, `judge_horary`, `get_transit_snapshot`, `find_boundary_changes`, `get_lagna_changes`, `rectify_birth_time`, `find_muhurta_windows`.
- **Options (`state.options`):** `ayanamsa` (KP_NEW default, KP_OLD, LAHIRI, RAMAN), `node_type` (mean default | true), `significator_method` (kp4level | four_step | six_fold), `include_type_e`, `retrograde_rule` (report_only | ksk_horary_cusp | strict), `cusp_orb_deg` + `cusp_proximity_apply`, `csl_extension` (off | own_bhava | sub_chain | both), `include_outer`, `significator_levels`.
- Schools disagree: every school-dependent rule is an option, the default is pure KP, and each output names the rule used. Details: `mcp/kp/progress.md`.

## Adding a new agent

1. Write the agent (for example `agents/finance.py`). Expose one async context manager that yields an `AgentSpec`
   (see `dasha_plugin()` in `agents/dasha.py`):
   - `name`: short lowercase id, for example `"finance"`
   - `description`: when to use it; **the router reads this**, so be specific about what it covers
   - `run(text, thread_id)`: async function returning the answer as text
2. In `orchestrator.py` import the plugin and add it to the `PLUGINS` list. Nothing else changes: the graph
   gets a node and an edge for it automatically, and both `main.py` and `orchestrator.py` pick it up.

## Rules this project follows

- Packages are never installed by tooling; only `requirements.txt` is edited.
- Secrets in `.env` are never printed.
- `progress.md` is updated after every step.
