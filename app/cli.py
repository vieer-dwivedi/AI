"""CLI: python -m app.cli run-scenario N (Step 9; Step 10 extends it)."""
import json
import sys
from pathlib import Path

from app.agent import memory
from app.agent.orchestrator import run_case

EMAILS = Path(__file__).resolve().parent.parent / "data" / "emails"


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] != "run-scenario":
        print("usage: python -m app.cli run-scenario <1-6>")
        return 1
    path = EMAILS / f"scenario_{argv[2]}.json"
    if not path.exists():
        print(f"no such scenario: {argv[2]}")
        return 1
    email = json.loads(path.read_text())
    print(f"Scenario {argv[2]}: {email['subject']}\n")
    case_id = f"CASE-S{argv[2]}"
    memory.create_case(case_id, email)
    result = run_case(email, case_id=case_id)
    status = memory.save_result(result, email)
    print(f"\nCase {case_id} saved with status: {status}")
    print(f"\nStopped: {result.stop_reason}")
    c = result.case
    print(f"Usage: {c.steps} steps, {c.tokens_in} in / {c.tokens_out} out tokens, ~${c.cost_usd:.4f}")
    if result.plan:
        print(result.plan.model_dump_json(indent=2))
    else:
        print("NO VALID PLAN: needs a human")
    if result.pending:
        print("\nPENDING APPROVAL:")
        for p in result.pending:
            print(f"  - {p['tool']} {p['args']}  ({p['reason']})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
