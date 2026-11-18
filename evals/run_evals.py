# =====================================================================
# evals/run_evals.py
# In simple words: a report card for the agent. For every known incident in
# evals/test_scenarios/*.json we run the FULL workflow and check the answer
# against what we know is true (right severity? right root cause? found the
# change that caused it? proposed the right fixes? executed nothing?).
# Run it with a different LLM_MODEL in .env to compare models:
#     python main.py eval            (or: python -m evals.run_evals)
# =====================================================================
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.orchestrator import run_incident_async  # noqa: E402
from core.config import get_settings, reload_settings  # noqa: E402
from core.scenarios import incident_from_scenario  # noqa: E402
from memory.short_term import get_section  # noqa: E402

SCENARIO_DIR = Path(__file__).resolve().parent / "test_scenarios"


def check(result, diagnosis: dict, expect: dict) -> dict[str, bool]:
    """Compare one run with the expected facts. Returns {check name: passed}."""
    got_actions = {a["action_type"] for a in result.proposed_actions}
    root = (result.root_cause or "").lower()
    checks = {
        "completed without errors": result.status == "completed" and not result.error,
        "severity": result.severity in expect["severity_in"],
        "root-cause category": diagnosis.get("category") == expect["root_cause_category"],
        "found the trigger change": expect["trigger_contains"] in diagnosis.get("trigger", ""),
        "root-cause wording": any(k in root for k in expect["root_cause_keywords_any"]),
        "used the right runbook": diagnosis.get("runbook") == expect["runbook"],
        "proposed the right fixes": set(expect["action_types_include"]) <= got_actions,
        "executed nothing": all(a["status"] == "proposed" for a in result.proposed_actions),
    }
    if "similar_incident_id" in expect:
        checks["recalled similar past incident"] = any(
            s["id"] == expect["similar_incident_id"] for s in result.similar_incidents)
    return checks


async def run_evals() -> list[dict]:
    rows = []
    for path in sorted(SCENARIO_DIR.glob("*.json")):
        spec = json.loads(path.read_text())
        name = spec["scenario"]
        os.environ["MOCK_SCENARIO"] = name              # isolate the fake AWS account
        reload_settings()
        incident = incident_from_scenario(name)
        result = await run_incident_async(incident)
        diagnosis = get_section(incident.id, "diagnosis") or {}
        checks = check(result, diagnosis, spec["expect"])
        rows.append({"scenario": name, "checks": checks, "passed": sum(checks.values()),
                     "total": len(checks), "seconds": result.duration_seconds})
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--min-score", type=float, default=1.0, help="exit 1 if overall score is lower (0-1)")
    args = p.parse_args(argv)

    rows = asyncio.run(run_evals())
    s = get_settings()
    print(f"\nEval report  (LLM_MODEL={s.llm_model})\n" + "-" * 60)
    for r in rows:
        print(f"{r['scenario']:<18} {r['passed']}/{r['total']}  ({r['seconds']}s)")
        for name, ok in r["checks"].items():
            if not ok:
                print(f"    FAILED: {name}")
    passed, total = sum(r["passed"] for r in rows), sum(r["total"] for r in rows)
    score = passed / total if total else 0.0
    print("-" * 60 + f"\nOverall: {passed}/{total} = {score:.0%}")
    (s.output_path / "eval_report.json").write_text(json.dumps(
        {"llm_model": s.llm_model, "score": score, "scenarios": rows}, indent=2))
    return 0 if score >= args.min_score else 1


if __name__ == "__main__":
    raise SystemExit(main())
