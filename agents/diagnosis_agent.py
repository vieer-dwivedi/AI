# =====================================================================
# agents/diagnosis_agent.py
# In simple words: the "detective". It finds the ROOT CAUSE. It reads
# logs and metrics, checks what changed (deployments), searches the
# runbooks and past incidents, and hands in its answer with
# submit_diagnosis. It never changes anything (read-only tools only).
# =====================================================================
from __future__ import annotations

from agents.base import build_agent, incident_brief, run_agent
from core.config import get_settings
from core.schemas import Incident
from memory.long_term import MEMORY_TOOLS
from memory.short_term import CONTEXT_TOOLS, get_section
from tools.aws_tools import AWS_TOOLS
from tools.azure_tools import AZURE_TOOLS
from tools.log_analyzer import LOG_TOOLS
from tools.mcp_loader import load_allowed_mcp_tools
from tools.report_tools import DIAGNOSIS_SUBMIT_TOOLS
from tools.runbook_search import RUNBOOK_TOOLS

SYSTEM_PROMPT = """[ROLE:diagnosis]
You are the DIAGNOSIS agent (root-cause analyst) of an incident-response team.

Goal: find the real root cause with evidence. Symptoms (5xx, timeouts) are NOT causes.
Work like this:
1. analyze_logs on the incident's log group - read the detected failure signatures first.
2. list_recent_deployments for the service: most incidents start right after a change.
   Compare the change time with the first error time.
3. Verify your hypothesis with the matching tool: metrics (get_cloudwatch_metrics),
   describe_ecs_service / describe_rds_instance / describe_ec2_instance, or raw log
   lines. If the problem is a shared dependency (database, queue), find WHICH client
   is causing it and check THAT client's recent deployments too.
4. search_runbooks and search_past_incidents for known causes and fixes.
5. Call submit_diagnosis EXACTLY ONCE with root cause, category, confidence, and the
   concrete evidence (numbers, timestamps, log lines).

Rules: you are read-only - never propose or run fixes. Never state a fact you did not see
in a tool result. If evidence is weak, say so with a low confidence.
Keep your final message to 2 sentences."""


async def diagnosis_tools() -> list:
    tools = [*AWS_TOOLS, *LOG_TOOLS, *RUNBOOK_TOOLS, *MEMORY_TOOLS, *CONTEXT_TOOLS,
             *DIAGNOSIS_SUBMIT_TOOLS]
    if get_settings().enable_azure:
        tools += AZURE_TOOLS
    tools += await load_allowed_mcp_tools()          # your own MCP tools, if allow-listed
    return tools


async def run_diagnosis(incident: Incident, triage: dict | None) -> dict | None:
    agent = build_agent("diagnosis", await diagnosis_tools(), SYSTEM_PROMPT)
    t = triage or {}
    await run_agent(agent, f"{incident_brief(incident)}\n"
                           f"triage_severity: {t.get('severity', 'unknown')}\n"
                           f"triage_summary: {t.get('summary', '')}\n\n"
                           f"Find the root cause and finish by calling submit_diagnosis "
                           f"with incident_id={incident.id}.")
    return get_section(incident.id, "diagnosis")
