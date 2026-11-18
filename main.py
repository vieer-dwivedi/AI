# =====================================================================
# main.py
# In simple words: the command-line "remote control" of the project.
#   python main.py check [--ping]      show config, test the LLM choice from .env
#   python main.py list                list the fake AWS incident scenarios
#   python main.py run ecs_oom         investigate one scenario (or: all)
#   python main.py eval                score the agent against known incidents
#   python main.py serve               start the webhook API (port 8000)
#   python main.py mcp                 start the MCP server (stdio)
# =====================================================================
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so imports work from anywhere

from core.config import get_settings, reload_settings  # noqa: E402


def _setup_logging() -> None:
    logging.basicConfig(level=get_settings().log_level.upper(),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "langchain_core"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_check(args) -> int:
    from core.llm import get_chat_model, is_fake

    s = get_settings()
    print(f"LLM_MODEL        : {s.llm_model}")
    for role in ("triage", "diagnosis", "remediation"):
        if s.model_for(role) != s.llm_model:
            print(f"  {role:<12} override: {s.model_for(role)}")
    print(f"EMBEDDINGS_MODEL : {s.embeddings_model}")
    print(f"CLOUD_BACKEND    : {s.cloud_backend} (region {s.aws_region})")
    print(f"AUTO_REMEDIATE   : {s.auto_remediate}")
    print(f"Slack            : {'webhook' if s.slack_webhook_url else 'dry-run -> out/slack_outbox.jsonl'}")
    try:
        model = get_chat_model()
    except Exception as e:  # noqa: BLE001
        print(f"\n[FAIL] could not build the model: {type(e).__name__}: {e}")
        return 1
    print(f"Model object     : {type(model).__name__}")
    if is_fake(s.llm_model):
        print("\n[NOTE] LLM_MODEL=fake: offline test brain. Set a real model in .env for real reasoning.")
    if args.ping:
        try:
            reply = model.invoke("Reply with the single word OK.")
            print(f"Ping reply       : {str(reply.content)[:80]!r}\n[OK] the LLM answers.")
        except Exception as e:  # noqa: BLE001
            print(f"\n[FAIL] LLM call failed: {type(e).__name__}: {e}")
            return 1
    return 0


def cmd_list(_args) -> int:
    from core.scenarios import incident_from_scenario, list_scenarios, load_scenario

    for n in list_scenarios():
        print(f"{n:<16} {incident_from_scenario(n).title}\n{'':<16} ({load_scenario(n)['description']})")
    return 0


def cmd_run(args) -> int:
    from agents.orchestrator import run_incident
    from core.scenarios import incident_from_scenario, list_scenarios

    names = list_scenarios() if args.scenario == "all" else [args.scenario]
    code = 0
    for name in names:
        os.environ["MOCK_SCENARIO"] = name            # isolate: the fake AWS account only has this one
        reload_settings()
        inc = incident_from_scenario(name)
        print(f"\n=== Investigating {inc.id}: {inc.title}")
        r = run_incident(inc)
        report = get_settings().output_path / "reports" / f"{inc.id}.md"
        print(f"status={r.status}  severity={r.severity}  actions={len(r.proposed_actions)}  "
              f"time={r.duration_seconds}s\nroot cause: {r.root_cause}\nreport: {report}")
        if r.error:
            print(f"errors: {r.error}")
            code = 1
        if args.show:
            print("\n" + r.report_markdown)
    return code


def cmd_eval(args) -> int:
    from evals.run_evals import main as eval_main

    return eval_main(["--min-score", str(args.min_score)])


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("api.webhook:app", host=args.host, port=args.port, log_level="info")
    return 0


def cmd_mcp(args) -> int:
    from mcp_server.server import main as mcp_main

    mcp_main(["--transport", args.transport])
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AI DevOps Incident Response Agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="show config and verify the LLM"); c.add_argument("--ping", action="store_true")
    sub.add_parser("list", help="list scenarios")
    r = sub.add_parser("run", help="investigate a scenario"); r.add_argument("scenario")
    r.add_argument("--show", action="store_true", help="print the full report")
    e = sub.add_parser("eval", help="score against known incidents"); e.add_argument("--min-score", type=float, default=1.0)
    s = sub.add_parser("serve", help="start the webhook API")
    s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000)
    m = sub.add_parser("mcp", help="start the MCP server")
    m.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http"])
    args = p.parse_args(argv)
    _setup_logging()
    return {"check": cmd_check, "list": cmd_list, "run": cmd_run, "eval": cmd_eval,
            "serve": cmd_serve, "mcp": cmd_mcp}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
