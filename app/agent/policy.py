"""Policy engine: ALLOW / NEEDS_APPROVAL / DENY for every write action (Step 8)."""
from enum import Enum
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import yaml
from simpleeval import simple_eval

from app.agent.schemas import CaseState

POLICY_PATH = Path(__file__).resolve().parent.parent.parent / "policy.yaml"


class Decision(str, Enum):
    ALLOW = "allow"
    NEEDS_APPROVAL = "needs_approval"
    DENY = "deny"


@lru_cache(maxsize=1)
def load_policy() -> dict:
    return yaml.safe_load(POLICY_PATH.read_text())


def evaluate_with_reason(tool_name: str, args: dict, case: CaseState,
                         policy: dict | None = None) -> tuple[Decision, str]:
    policy = policy or load_policy()
    limits = policy.get("global", {})
    rule = policy.get("tools", {}).get(tool_name)

    if rule is None:
        return Decision.DENY, f"unknown tool '{tool_name}' (default deny)"
    if case.steps >= limits.get("max_steps", 15):
        return Decision.DENY, "step budget exhausted"
    if case.cost_usd >= limits.get("max_cost_usd_per_case", 0.75):
        return Decision.DENY, "cost budget exhausted"

    if rule.get("require_approval") == "always":
        return Decision.NEEDS_APPROVAL, "this action always needs approval"

    expr = rule.get("require_approval_if")
    if expr:
        if "customer." in expr and case.customer_tier is None:
            return Decision.NEEDS_APPROVAL, "customer tier unknown; failing closed"
        try:
            names = {"args": SimpleNamespace(**args),
                     "customer": SimpleNamespace(tier=case.customer_tier)}
            if bool(simple_eval(expr, names=names)):
                return Decision.NEEDS_APPROVAL, f"rule matched: {expr}"
        except Exception as exc:  # missing arg, wrong type, bad rule: fail closed
            return Decision.NEEDS_APPROVAL, f"could not evaluate rule ({type(exc).__name__}); failing closed"

    return Decision.ALLOW, "within policy"


def evaluate(tool_name: str, args: dict, case: CaseState, policy: dict | None = None) -> Decision:
    return evaluate_with_reason(tool_name, args, case, policy)[0]