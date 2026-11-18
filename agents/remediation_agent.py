# =====================================================================
# agents/remediation_agent.py
# In simple words: the "fixer". Given the root cause, it reads the
# runbook and PROPOSES fixes (with exact steps and a risk level) for a
# human to approve. It can only try to execute if AUTO_REMEDIATE=true,
# and the safety rules in tools/remediation_tools.py still block
# high-risk actions and anything on real AWS.
# =====================================================================
from __future__ import annotations

from agents.base import build_agent, incident_brief, run_agent
from core.schemas import Incident
from memory.short_term import CONTEXT_TOOLS, get_context
from tools.aws_tools import describe_ec2_instance, describe_ecs_service, describe_rds_instance
from tools.remediation_tools import REMEDIATION_TOOLS
from tools.runbook_search import RUNBOOK_TOOLS

SYSTEM_PROMPT = """[ROLE:remediation]
You are the REMEDIATION agent of an incident-response team.

Goal: turn the diagnosed root cause into safe, concrete fixes a human can approve.
Work like this:
1. search_runbooks for the matching procedure. Follow the runbook, do not improvise.
2. Look at the current state of the target if needed (describe_ecs_service, etc.)
   so your commands contain REAL names and versions, not placeholders.
3. Call propose_remediation once per fix. Put the fastest way to stop the customer pain
   first (e.g. rollback / stop the offending job), then stop-gaps, then permanent fixes.
   Set an honest risk level (low / medium / high) and give exact commands in `steps`.
   If the task lists similar past incidents, reuse what worked, but only if it fits
   THIS root cause.
4. Only call execute_remediation if the task says auto-remediation is allowed; it is
   normally NOT. Never retry a blocked action.

Keep your final message to 2 sentences."""


def remediation_tools() -> list:
    return [*RUNBOOK_TOOLS, describe_ecs_service, describe_rds_instance, describe_ec2_instance,
            *REMEDIATION_TOOLS, *CONTEXT_TOOLS]


async def run_remediation(incident: Incident, diagnosis: dict | None,
                          similar: list[dict] | None = None) -> list[dict]:
    d = diagnosis or {}
    agent = build_agent("remediation", remediation_tools(), SYSTEM_PROMPT)
    past = "".join(f"similar_past_incident: {s['id']} - {s['title']} | past fix: {s['resolution']}\n"
                   for s in (similar or []))
    await run_agent(agent, f"{incident_brief(incident)}\n"
                           f"category: {d.get('category', 'other')}\n"
                           f"root_cause: {d.get('root_cause', 'unknown')}\n"
                           f"trigger: {d.get('trigger', '')}\n"
                           f"runbook: {d.get('runbook', '')}\n"
                           f"{past}"
                           f"evidence:\n{d.get('evidence', '')}\n\n"
                           f"Propose the fixes now using propose_remediation "
                           f"with incident_id={incident.id}. Do not execute anything.")
    return get_context(incident.id)["actions"]
