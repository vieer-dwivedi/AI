# =====================================================================
# agents/triage_agent.py
# In simple words: the "first responder". It decides how bad the incident
# is (P1-P4). It only looks at alarms and logs - quickly - and hands in a
# short assessment with submit_triage. It does NOT hunt for the root cause.
# =====================================================================
from __future__ import annotations

from agents.base import build_agent, incident_brief, run_agent
from core.config import get_settings
from core.schemas import Incident
from memory.short_term import CONTEXT_TOOLS, get_section
from tools.aws_tools import (describe_cloudwatch_alarms, describe_ecs_service,
                             get_cloudwatch_metrics, list_log_groups)
from tools.azure_tools import AZURE_TOOLS
from tools.log_analyzer import LOG_TOOLS
from tools.report_tools import TRIAGE_SUBMIT_TOOLS

SYSTEM_PROMPT = """[ROLE:triage]
You are the TRIAGE agent of an incident-response team. You are fast and factual.

Goal: decide how severe this incident is and who is affected.
Work like this:
1. Call describe_cloudwatch_alarms to see what is firing.
2. Call analyze_logs on the incident's log_group (use list_log_groups if none is given)
   to measure the error rate and see the main error.
3. Optionally check one metric or the ECS service state.
4. Call submit_triage EXACTLY ONCE with your impact assessment. The official severity
   P1-P4 is computed from it by company rules - do not invent your own severity.

Rules: do not search for the root cause (another agent does that), do not propose fixes,
never guess numbers - use what the tools return. Keep your final message to 2 sentences."""


def triage_tools() -> list:
    tools = [describe_cloudwatch_alarms, list_log_groups, get_cloudwatch_metrics,
             describe_ecs_service, *LOG_TOOLS, *CONTEXT_TOOLS, *TRIAGE_SUBMIT_TOOLS]
    if get_settings().enable_azure:
        tools += AZURE_TOOLS
    return tools


async def run_triage(incident: Incident) -> dict | None:
    """Run the triage agent; returns the saved triage result (or None if it never submitted)."""
    agent = build_agent("triage", triage_tools(), SYSTEM_PROMPT)
    await run_agent(agent, f"{incident_brief(incident)}\n\n"
                           f"Assess this incident now and finish by calling submit_triage "
                           f"with incident_id={incident.id}.")
    return get_section(incident.id, "triage")
