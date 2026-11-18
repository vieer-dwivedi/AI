# =====================================================================
# memory/short_term.py
# In simple words: the agent's SHORT-TERM memory = a notebook for ONE
# incident that is being worked on right now. Sub-agents write findings
# into it ("memory is at 99%", "deploy happened at 08:35") and the
# remediation agent writes its proposed fixes into it. When the incident
# is over, the notebook is saved to long-term memory and can be dropped.
# It lives in the program's memory (a dict), guarded by a lock.
# =====================================================================
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import tool

from tools.common import jdump, safe_tool

_lock = threading.Lock()
_store: dict[str, dict[str, Any]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def get_context(incident_id: str) -> dict[str, Any]:
    """Get (or create) the notebook for an incident."""
    with _lock:
        return _store.setdefault(incident_id, {"findings": [], "actions": []})


def add_finding(incident_id: str, agent: str, finding: str) -> dict:
    ctx = get_context(incident_id)
    entry = {"time": _now(), "agent": agent, "finding": finding}
    with _lock:
        ctx["findings"].append(entry)
    return entry


def add_action(incident_id: str, action: dict) -> dict:
    """Store a proposed remediation and give it an id like ACT-1."""
    ctx = get_context(incident_id)
    with _lock:
        action = {**action, "action_id": f"ACT-{len(ctx['actions']) + 1}", "status": "proposed"}
        ctx["actions"].append(action)
    return action


def find_action(incident_id: str, action_id: str) -> dict | None:
    ctx = get_context(incident_id)
    with _lock:
        return next((a for a in ctx["actions"] if a["action_id"] == action_id), None)


def set_section(incident_id: str, key: str, value: Any) -> Any:
    """Save a named result block (e.g. 'triage' or 'diagnosis') in the notebook."""
    ctx = get_context(incident_id)
    with _lock:
        ctx[key] = value
    return value


def get_section(incident_id: str, key: str, default: Any = None) -> Any:
    """Read a named result block back (None if the agent never wrote it)."""
    ctx = get_context(incident_id)
    with _lock:
        return ctx.get(key, default)


def clear_context(incident_id: str | None = None) -> None:
    """Forget one incident (or everything when no id is given)."""
    with _lock:
        if incident_id:
            _store.pop(incident_id, None)
        else:
            _store.clear()


@tool
@safe_tool
def record_finding(incident_id: str, agent: str, finding: str) -> str:
    """Save one short, factual finding in the shared incident notebook
    (e.g. 'MemoryUtilization rose 52%->99% starting 08:35, right after deploy v2.14.0').
    Use your own agent name for `agent`. Keep each finding to one or two sentences."""
    return jdump({"saved": add_finding(incident_id, agent, finding)})


@tool
@safe_tool
def get_incident_context(incident_id: str) -> str:
    """Read everything recorded so far for this incident: findings from all
    agents and any proposed remediation actions."""
    return jdump(get_context(incident_id))


CONTEXT_TOOLS = [record_finding, get_incident_context]
