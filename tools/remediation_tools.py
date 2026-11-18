# =====================================================================
# tools/remediation_tools.py
# In simple words: how the agent suggests (and, only if allowed, runs) fixes.
#
# SAFETY RULES built into the code (the LLM cannot override them):
#   1. propose_remediation only WRITES DOWN a suggestion. Nothing runs.
#   2. execute_remediation is blocked unless AUTO_REMEDIATE=true in .env.
#   3. HIGH-risk actions are ALWAYS blocked - a human must do them.
#   4. Even when allowed, execution is only SIMULATED on the mock backend,
#      and NOT implemented for real AWS (you run the commands yourself).
# =====================================================================
from __future__ import annotations

from langchain_core.tools import tool

from core.config import get_settings
from memory.short_term import add_action, find_action, get_context
from tools.common import jdump, safe_tool

ALLOWED_RISK = {"low", "medium", "high"}
KNOWN_ACTIONS = {
    "rollback_deployment", "scale_service", "restart_service", "increase_memory",
    "stop_job", "truncate_logs", "reduce_log_level", "terminate_idle_connections",
    "extend_volume", "add_connection_pooling", "other",
}


@tool
@safe_tool
def propose_remediation(incident_id: str, action_type: str, target: str,
                        rationale: str, risk: str = "low", steps: str = "") -> str:
    """Record ONE proposed fix for human approval (this does not run anything).
    action_type: rollback_deployment | scale_service | restart_service |
      increase_memory | stop_job | truncate_logs | reduce_log_level |
      terminate_idle_connections | extend_volume | add_connection_pooling | other.
    target: the resource, e.g. 'ecs:prod-cluster/checkout-api'.
    rationale: why this fixes the root cause (cite evidence).
    risk: low | medium | high.  steps: the exact commands or console steps."""
    risk = risk.lower().strip()
    if risk not in ALLOWED_RISK:
        return jdump({"error": f"risk must be one of {sorted(ALLOWED_RISK)}"})
    if action_type not in KNOWN_ACTIONS:
        return jdump({"error": f"action_type must be one of {sorted(KNOWN_ACTIONS)}"})
    action = add_action(incident_id, {"action_type": action_type, "target": target,
                                      "rationale": rationale, "risk": risk, "steps": steps})
    return jdump({"proposed": action, "note": "Awaiting human approval; nothing was executed."})


@tool
@safe_tool
def list_proposed_actions(incident_id: str) -> str:
    """List remediation actions proposed so far for this incident."""
    return jdump({"actions": get_context(incident_id)["actions"]})


@tool
@safe_tool
def execute_remediation(incident_id: str, action_id: str) -> str:
    """Try to execute a previously proposed action (e.g. 'ACT-1').
    Will be refused unless auto-remediation is enabled AND the action is
    not high risk. Most of the time you should NOT call this: propose only."""
    s = get_settings()
    action = find_action(incident_id, action_id)
    if not action:
        return jdump({"status": "error", "reason": f"unknown action {action_id}"})
    if action["risk"] == "high":
        return jdump({"status": "blocked", "reason": "HIGH risk actions always need a human."})
    if not s.auto_remediate:
        return jdump({"status": "blocked",
                      "reason": "AUTO_REMEDIATE is false: a human must approve and run this."})
    if s.cloud_backend.lower() != "mock":
        return jdump({"status": "not_implemented",
                      "reason": "Automatic execution on real AWS is not implemented. "
                                "Run the proposed steps manually."})
    action["status"] = "executed_simulated"
    return jdump({"status": "executed_simulated", "action": action,
                  "note": "Mock backend: nothing real was changed."})


REMEDIATION_TOOLS = [propose_remediation, list_proposed_actions, execute_remediation]
