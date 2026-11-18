# =====================================================================
# agents/orchestrator.py
# In simple words: the "incident commander". It is a LangGraph workflow
# (plan -> execute) that runs the specialists in order and decides the
# route:
#
#   intake -> triage -> (P4/noise? skip) -> diagnosis -> recall -> remediation -> report
#
#  - intake:      remember the incident, tell Slack
#  - triage:      how bad is it?          (triage agent)
#  - diagnosis:   why is it happening?    (diagnosis agent)
#  - recall:      have we seen this ROOT CAUSE before? (long-term memory)
#  - remediation: what should we do?      (remediation agent, proposals only,
#                 it also sees how similar past incidents were fixed)
#  - report:      write the markdown report, save the case to long-term
#                 memory, tell Slack, return an IncidentResult
# If any agent crashes (e.g. wrong API key) the error is recorded and the
# graph jumps straight to the report, so you always get a result.
# =====================================================================
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.diagnosis_agent import run_diagnosis
from agents.remediation_agent import run_remediation
from agents.triage_agent import run_triage
from core.config import get_settings
from core.schemas import Incident, IncidentResult
from memory.long_term import save_incident, search_incidents
from memory.short_term import add_finding, clear_context, get_context
from tools.slack_notifier import send_slack

log = logging.getLogger(__name__)

_SLACK_SEVERITY = {"P1": "critical", "P2": "high", "P3": "warning", "P4": "info"}


class State(TypedDict, total=False):
    """The notebook that travels through the graph."""
    incident: Incident
    t0: float
    triage: dict | None
    diagnosis: dict | None
    actions: list[dict]
    similar: list[dict]
    errors: list[str]
    result: IncidentResult


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
async def _notify(message: str, severity: str, incident_id: str) -> None:
    """Slack must never break an investigation."""
    try:
        await asyncio.to_thread(send_slack, message, severity, incident_id)
    except Exception as e:  # noqa: BLE001
        log.warning("Slack notification failed: %s", e)


def enrich_incident(incident: Incident) -> Incident:
    """Webhook alerts often lack hints. If no log group is known, guess it from the
    service name using the account's log groups, so the agents start with a good lead."""
    if incident.labels.get("log_group") or not incident.service:
        return incident
    try:
        from tools.aws_tools import list_log_groups

        groups = json.loads(list_log_groups.invoke({})).get("log_groups", [])
        match = next((g for g in groups if incident.service in g), None)
        if match:
            incident.labels["log_group"] = match
    except Exception as e:  # noqa: BLE001
        log.info("Could not guess log group: %s", e)
    return incident


def build_report(incident: Incident, triage: dict | None, diagnosis: dict | None,
                 actions: list[dict], similar: list[dict], findings: list[dict],
                 errors: list[str]) -> str:
    """Turn everything the agents found into one readable markdown report."""
    t, d = triage or {}, diagnosis or {}
    L = [f"# Incident report: {incident.title}", "",
         f"- **ID:** {incident.id}   **Service:** {incident.service}   **Env:** {incident.environment}",
         f"- **Source:** {incident.source}   **Alarm:** {incident.alarm_name}   "
         f"**Triggered:** {incident.triggered_at}",
         f"- **Severity:** {t.get('severity', 'not assessed')}"
         + (f" - {t['response']}" if t.get("response") else ""), ""]
    if t:
        L += ["## Impact", t.get("summary", ""), ""]
    if d:
        L += ["## Root cause", f"{d.get('root_cause', '')}", "",
              f"- **Category:** {d.get('category')}   **Confidence:** {d.get('confidence', 0):.0%}",
              f"- **Trigger:** {d.get('trigger') or 'n/a'}",
              f"- **Runbook:** {d.get('runbook') or 'n/a'}", "", "### Evidence"]
        L += [f"- {line.lstrip('-* ').strip()}" for line in str(d.get("evidence", "")).splitlines()
              if line.strip()]
        L.append("")
    if actions:
        L += ["## Proposed remediation (needs human approval)", ""]
        for a in actions:
            L += [f"### {a['action_id']}: {a['action_type']} -> `{a['target']}`  (risk: {a['risk']})",
                  a["rationale"], "", "```", a.get("steps") or "(no steps given)", "```", ""]
    if similar:
        L += ["## Similar past incidents"]
        L += [f"- **{s['id']}** ({s['severity']}, relevance {s['relevance']}): {s['title']} "
              f"- fix: {s['resolution']}" for s in similar]
        L.append("")
    if findings:
        L += ["## Timeline of findings"]
        L += [f"- {f['time']} [{f['agent']}] {f['finding']}" for f in findings]
        L.append("")
    if errors:
        L += ["## Problems during the run"] + [f"- {e}" for e in errors] + [""]
    return "\n".join(L)


# ---------------------------------------------------------------------
# Graph nodes (each one returns only the parts of the state it changes)
# ---------------------------------------------------------------------
async def intake(state: State) -> dict:
    inc = enrich_incident(state["incident"])
    clear_context(inc.id)
    add_finding(inc.id, "orchestrator", f"Incident received: {inc.brief()}")
    await _notify(f"Investigating: {inc.title}", "info", inc.id)
    return {"incident": inc, "similar": [], "errors": [], "t0": time.monotonic()}


async def triage_node(state: State) -> dict:
    inc, errors = state["incident"], list(state["errors"])
    triage = None
    try:
        triage = await run_triage(inc)
        if triage is None:
            errors.append("triage: agent finished without calling submit_triage")
    except Exception as e:  # noqa: BLE001
        log.exception("triage failed")
        errors.append(f"triage: {type(e).__name__}: {e}")
    if triage:
        await _notify(f"Severity {triage['severity']}: {triage['summary']}",
                      _SLACK_SEVERITY[triage["severity"]], inc.id)
    return {"triage": triage, "errors": errors}


def after_triage(state: State) -> str:
    """P4 = informational noise: no point spending an LLM investigation on it."""
    t = state.get("triage")
    if t is None:
        return "report"
    return "report" if t["severity"] == "P4" else "diagnosis"


async def diagnosis_node(state: State) -> dict:
    inc, errors = state["incident"], list(state["errors"])
    diagnosis = None
    try:
        diagnosis = await run_diagnosis(inc, state.get("triage"))
        if diagnosis is None:
            errors.append("diagnosis: agent finished without calling submit_diagnosis")
    except Exception as e:  # noqa: BLE001
        log.exception("diagnosis failed")
        errors.append(f"diagnosis: {type(e).__name__}: {e}")
    if diagnosis:
        await _notify(f"Root cause ({diagnosis['confidence']:.0%} sure): {diagnosis['root_cause']}",
                      "high", inc.id)
    return {"diagnosis": diagnosis, "errors": errors}


def after_diagnosis(state: State) -> str:
    return "recall" if state.get("diagnosis") else "report"


async def recall_node(state: State) -> dict:
    """Search long-term memory using the ROOT CAUSE (the original alert text only
    describes symptoms, which match too many things)."""
    inc, d = state["incident"], state["diagnosis"]
    try:
        hits = search_incidents(f"{d['root_cause']} {d['category']}", k=4)
        similar = [s for s in hits if s["id"] != inc.id and s["relevance"] >= 0.15][:3]
    except Exception as e:  # noqa: BLE001
        log.warning("Past-incident lookup failed: %s", e)
        similar = []
    if similar:
        add_finding(inc.id, "orchestrator", f"Similar past incident: {similar[0]['id']} - {similar[0]['title']}")
    return {"similar": similar}


async def remediation_node(state: State) -> dict:
    inc, errors = state["incident"], list(state["errors"])
    actions: list[dict] = []
    try:
        actions = await run_remediation(inc, state.get("diagnosis"), state.get("similar"))
        if not actions:
            errors.append("remediation: agent proposed no actions")
    except Exception as e:  # noqa: BLE001
        log.exception("remediation failed")
        errors.append(f"remediation: {type(e).__name__}: {e}")
    return {"actions": actions, "errors": errors}


async def report_node(state: State) -> dict:
    inc, triage, diagnosis = state["incident"], state.get("triage"), state.get("diagnosis")
    actions, errors = state.get("actions", []), state.get("errors", [])
    ctx = get_context(inc.id)
    markdown = build_report(inc, triage, diagnosis, actions, state.get("similar", []),
                            ctx["findings"], errors)

    # Save the report to a file next to the other outputs.
    reports = get_settings().output_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{inc.id}.md").write_text(markdown)

    # Long-term memory: remember the case. The fix is only PROPOSED, so we say so.
    if diagnosis:
        try:
            first = actions[0] if actions else None
            save_incident(inc.id, inc.title, (triage or {}).get("severity", ""),
                          diagnosis["root_cause"],
                          f"UNVERIFIED proposal: {first['action_type']} on {first['target']}" if first else "",
                          f"{diagnosis['category']} {inc.service}")
        except Exception as e:  # noqa: BLE001
            errors = [*errors, f"memory: could not save incident ({type(e).__name__}: {e})"]

    status = "failed" if (errors and not diagnosis and not (triage and triage["severity"] == "P4")) \
        else "completed"
    elapsed = round(time.monotonic() - state["t0"], 2)
    result = IncidentResult(
        incident_id=inc.id, status=status, severity=(triage or {}).get("severity"),
        root_cause=(diagnosis or {}).get("root_cause"), report_markdown=markdown,
        proposed_actions=actions, similar_incidents=state.get("similar", []),
        finished_at=datetime.now(timezone.utc).isoformat(), duration_seconds=elapsed,
        error="; ".join(errors) or None)
    summary = (f"Report ready ({elapsed}s). Severity {result.severity or 'n/a'}. "
               f"{len(actions)} fix(es) proposed - awaiting approval." if status == "completed"
               else f"Investigation FAILED: {result.error}")
    await _notify(summary, "info" if status == "completed" else "warning", inc.id)
    return {"result": result, "errors": errors}


# ---------------------------------------------------------------------
# Wire the graph together
# ---------------------------------------------------------------------
def build_graph():
    g = StateGraph(State)
    g.add_node("intake", intake)
    g.add_node("triage", triage_node)
    g.add_node("diagnosis", diagnosis_node)
    g.add_node("recall", recall_node)
    g.add_node("remediation", remediation_node)
    g.add_node("report", report_node)
    g.add_edge(START, "intake")
    g.add_edge("intake", "triage")
    g.add_conditional_edges("triage", after_triage, {"diagnosis": "diagnosis", "report": "report"})
    g.add_conditional_edges("diagnosis", after_diagnosis, {"recall": "recall", "report": "report"})
    g.add_edge("recall", "remediation")
    g.add_edge("remediation", "report")
    g.add_edge("report", END)
    return g.compile()


async def run_incident_async(incident: Incident) -> IncidentResult:
    """Run the full workflow for one incident (use this inside async code like FastAPI)."""
    final: Any = await build_graph().ainvoke({"incident": incident})
    return final["result"]


def run_incident(incident: Incident) -> IncidentResult:
    """Same thing for normal (non-async) code such as scripts."""
    return asyncio.run(run_incident_async(incident))
