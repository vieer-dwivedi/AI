# =====================================================================
# tools/report_tools.py
# In simple words: "hand in your answer" tools.
# Instead of hoping the LLM writes its conclusion in a perfect format, we
# give each agent a form to fill in:
#   submit_triage    -> the triage agent fills: impact, summary, error rate
#                       (the severity P1-P4 is then calculated by OUR rules)
#   submit_diagnosis -> the diagnosis agent fills: root cause, evidence...
# The filled form is saved in the short-term notebook, and the orchestrator
# reads it from there. This works the same with ANY LLM.
# =====================================================================
from __future__ import annotations

from langchain_core.tools import tool

from memory.short_term import add_finding, set_section
from tools.common import jdump, safe_tool
from tools.triage_tools import compute_severity

CATEGORIES = [
    "bad_deployment", "memory_exhaustion", "disk_exhaustion", "cpu_saturation",
    "connection_exhaustion", "dependency_failure", "config_change", "capacity",
    "throttling", "other",
]


@tool
@safe_tool
def submit_triage(incident_id: str, user_impact: str, summary: str,
                  affected_service: str = "", environment: str = "prod",
                  error_rate_pct: float = 0.0, revenue_path: bool = False) -> str:
    """Hand in your FINAL triage result (call this exactly once, at the end).
    user_impact: none | minor | degraded | partial_outage | full_outage.
    summary: 1-2 sentences on what is broken and who is affected.
    error_rate_pct: share of failing requests 0-100 (0 if unknown).
    revenue_path: true if checkout / payments / orders are affected.
    The official severity P1-P4 is computed for you from these inputs."""
    sev = compute_severity(user_impact, environment, error_rate_pct, revenue_path)
    section = {**sev, "user_impact": user_impact, "summary": summary,
               "affected_service": affected_service, "environment": environment,
               "error_rate_pct": error_rate_pct, "revenue_path": revenue_path}
    set_section(incident_id, "triage", section)
    add_finding(incident_id, "triage", f"Severity {sev['severity']}: {summary}")
    return jdump({"saved": True, "severity": sev["severity"], "response": sev["response"]})


@tool
@safe_tool
def submit_diagnosis(incident_id: str, root_cause: str, category: str, confidence: float,
                     evidence: str, affected_service: str = "", trigger: str = "",
                     runbook: str = "") -> str:
    """Hand in your FINAL root-cause analysis (call this exactly once, at the end).
    root_cause: one or two sentences naming the real cause.
    category: bad_deployment | memory_exhaustion | disk_exhaustion | cpu_saturation |
      connection_exhaustion | dependency_failure | config_change | capacity |
      throttling | other.
    confidence: 0.0 - 1.0.   evidence: the concrete facts you found (metrics, log
    lines, deployment times), as short lines separated by newlines.
    trigger: what changed right before it started (deploy id, config change...).
    runbook: the matching runbook file name, if you found one."""
    category = category.strip().lower()
    if category not in CATEGORIES:
        category = "other"
    conf = max(0.0, min(1.0, float(confidence)))
    section = {"root_cause": root_cause, "category": category, "confidence": conf,
               "evidence": evidence, "affected_service": affected_service,
               "trigger": trigger, "runbook": runbook}
    set_section(incident_id, "diagnosis", section)
    add_finding(incident_id, "diagnosis", f"Root cause ({category}, {conf:.0%}): {root_cause}")
    return jdump({"saved": True, "category": category, "confidence": conf})


TRIAGE_SUBMIT_TOOLS = [submit_triage]
DIAGNOSIS_SUBMIT_TOOLS = [submit_diagnosis]
