# =====================================================================
# tools/triage_tools.py
# In simple words: a fixed "severity scoring sheet" for triage. The LLM
# judges how bad the impact is, and this tool turns that into P1-P4 using
# the company's rules, so severity is consistent and not a mood.
#   P1 = customers badly hit, act now      P3 = minor, handle in hours
#   P2 = serious but partial               P4 = informational
# =====================================================================
from __future__ import annotations

from langchain_core.tools import tool

from tools.common import jdump, safe_tool

IMPACTS = ["none", "minor", "degraded", "partial_outage", "full_outage"]
SLA = {"P1": "Respond in 5 min, page on-call + incident commander",
       "P2": "Respond in 15 min, page on-call",
       "P3": "Respond in 4 hours, ticket to owning team",
       "P4": "No page, log and review next business day"}


def compute_severity(user_impact: str, environment: str = "prod",
                     error_rate_pct: float = 0.0, revenue_path: bool = False) -> dict:
    impact = user_impact.lower().strip()
    if impact not in IMPACTS:
        raise ValueError(f"user_impact must be one of {IMPACTS}")
    env = environment.lower()
    why = [f"impact={impact}", f"env={env}", f"error_rate={error_rate_pct}%",
           f"revenue_path={revenue_path}"]

    if impact == "full_outage":
        sev = "P1"
    elif impact == "partial_outage":
        sev = "P1" if (revenue_path or error_rate_pct >= 25) else "P2"
    elif impact == "degraded":
        sev = "P2" if (revenue_path or error_rate_pct >= 10) else "P3"
    elif impact == "minor":
        sev = "P3"
    else:
        sev = "P4"

    if env not in {"prod", "production"}:        # non-production never pages at P1/P2
        order = ["P1", "P2", "P3", "P4"]
        sev = order[max(order.index(sev), 2)]
        why.append("non-prod capped at P3")
    return {"severity": sev, "response": SLA[sev], "inputs": why}


@tool
@safe_tool
def severity_rubric(user_impact: str, environment: str = "prod",
                    error_rate_pct: float = 0.0, revenue_path: bool = False) -> str:
    """Convert your impact assessment into an official severity P1-P4.
    user_impact: none | minor | degraded | partial_outage | full_outage.
    environment: prod / staging / dev.  error_rate_pct: share of failing
    requests (0-100).  revenue_path: true if checkout/payments/orders are affected."""
    return jdump(compute_severity(user_impact, environment, error_rate_pct, revenue_path))


TRIAGE_TOOLS = [severity_rubric]
