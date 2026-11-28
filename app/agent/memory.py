"""Case state persistence so cases survive restarts and can wait for approval (Step 10).

Uses the `cases`, `approvals` and `tool_call_log` tables in the ERP database.
"""
import json
import uuid
from datetime import datetime, timezone

from langchain_core.messages import messages_from_dict, messages_to_dict
from sqlalchemy import create_engine, text

from app.agent.policy import load_policy
from app.config import settings

_engine = create_engine(settings.database_url)

SCHEMA = [
    "CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, order_id INTEGER, status TEXT, state_json TEXT)",
    "CREATE TABLE IF NOT EXISTS approvals (id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, tool_name TEXT, "
    "args_json TEXT, status TEXT, comment TEXT)",
    "CREATE TABLE IF NOT EXISTS tool_call_log (id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, tool_name TEXT, "
    "args_json TEXT, result_json TEXT, ts TEXT)",
]


def _init() -> None:
    with _engine.begin() as conn:
        for stmt in SCHEMA:
            conn.execute(text(stmt))


def _rows(sql: str, **params) -> list[dict]:
    _init()
    with _engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(text(sql), params)]


def new_case_id() -> str:
    return f"CASE-{uuid.uuid4().hex[:6].upper()}"


def create_case(case_id: str, email: dict) -> None:
    """Register a case as 'running'. Re-using an id (CLI scenarios) resets it."""
    _init()
    with _engine.begin() as conn:
        conn.execute(text("DELETE FROM approvals WHERE case_id = :c"), {"c": case_id})
        conn.execute(text("INSERT OR REPLACE INTO cases (id, order_id, status, state_json) "
                          "VALUES (:c, NULL, 'running', :s)"),
                     {"c": case_id, "s": json.dumps({"email": email}, default=str)})


def _load_state(case_id: str) -> dict:
    rows = _rows("SELECT state_json FROM cases WHERE id = :c", c=case_id)
    return json.loads(rows[0]["state_json"] or "{}") if rows else {}


def _write(case_id: str, status: str, state: dict, order_id: int | None = None) -> None:
    with _engine.begin() as conn:
        conn.execute(text("UPDATE cases SET status = :s, state_json = :j, "
                          "order_id = COALESCE(:o, order_id) WHERE id = :c"),
                     {"s": status, "j": json.dumps(state, default=str), "o": order_id, "c": case_id})


def set_status(case_id: str, status: str) -> None:
    _write(case_id, status, _load_state(case_id))


def fail_case(case_id: str, error: str) -> None:
    state = _load_state(case_id)
    state["error"] = error
    _write(case_id, "failed", state)


def fail_stale_cases() -> int:
    """On API start, cases still 'running' were cut off by a restart: mark them failed."""
    stale = [r["id"] for r in _rows("SELECT id FROM cases WHERE status = 'running'")]
    for case_id in stale:
        fail_case(case_id, "interrupted: the API restarted while this case was running")
    return len(stale)


def _order_id(trace: list[dict]) -> int | None:
    for t in trace:
        if t.get("tool") == "get_order":
            try:
                return int(t["args"]["order_id"])
            except (KeyError, TypeError, ValueError):
                return None
    return None


def save_result(result, email: dict) -> str:
    """Persist a finished agent run (CaseResult) and queue its pending approvals. Returns the status."""
    min_conf = load_policy().get("global", {}).get("min_confidence_auto_close", 0.8)
    plan = result.plan
    if plan is None:
        status = "needs_human"
    elif result.pending:
        status = "awaiting_approval"
    elif plan.needs_human_reason or plan.confidence < min_conf:
        status = "needs_review"
    else:
        status = "closed"

    state = {
        "email": email,
        "plan": plan.model_dump() if plan else None,
        "pending": result.pending,
        "trace": result.trace,
        "stop_reason": result.stop_reason,
        "usage": {"steps": result.case.steps, "tokens_in": result.case.tokens_in,
                  "tokens_out": result.case.tokens_out, "cost_usd": round(result.case.cost_usd, 4)},
        "customer_tier": result.case.customer_tier,
        "messages": messages_to_dict(result.messages),    # kept so the case can be resumed or replayed
    }
    case_id = result.case.case_id
    _init()
    _write(case_id, status, state, _order_id(result.trace))
    with _engine.begin() as conn:
        for p in result.pending:
            conn.execute(text("INSERT INTO approvals (case_id, tool_name, args_json, status, comment) "
                              "VALUES (:c, :t, :a, 'pending', '')"),
                         {"c": case_id, "t": p["tool"], "a": json.dumps(p["args"], default=str)})
    return status


def load_messages(case_id: str) -> list:
    return messages_from_dict(_load_state(case_id).get("messages", []))


def get_case(case_id: str) -> dict | None:
    rows = _rows("SELECT id, order_id, status FROM cases WHERE id = :c", c=case_id)
    if not rows:
        return None
    state = _load_state(case_id)
    state.pop("messages", None)                           # large; not for API consumers
    executed = [{"tool": r["tool_name"], "args": json.loads(r["args_json"]),
                 "result": json.loads(r["result_json"]), "ts": r["ts"]}
                for r in _rows("SELECT tool_name, args_json, result_json, ts FROM tool_call_log "
                               "WHERE case_id = :c ORDER BY id", c=case_id)]
    return {"case_id": case_id, "order_id": rows[0]["order_id"], "status": rows[0]["status"],
            **state, "approvals": list_approvals(case_id=case_id), "executed_actions": executed}


def list_cases() -> list[dict]:
    """Inbox view: one compact row per case, newest first."""
    out = []
    for r in _rows("SELECT id, order_id, status, state_json FROM cases ORDER BY rowid DESC"):
        state = json.loads(r["state_json"] or "{}")
        plan = state.get("plan") or {}
        out.append({"case_id": r["id"], "order_id": r["order_id"], "status": r["status"],
                    "subject": (state.get("email") or {}).get("subject", ""),
                    "exception_type": plan.get("exception_type", ""),
                    "confidence": plan.get("confidence"),
                    "reason": plan.get("needs_human_reason") or state.get("error") or "",
                    "cost_usd": (state.get("usage") or {}).get("cost_usd")})
    return out


def list_approvals(status: str | None = None, case_id: str | None = None) -> list[dict]:
    sql, params = "SELECT id, case_id, tool_name, args_json, status, comment FROM approvals WHERE 1=1", {}
    if status:
        sql, params["s"] = sql + " AND status = :s", status
    if case_id:
        sql, params["c"] = sql + " AND case_id = :c", case_id
    out = []
    for r in _rows(sql + " ORDER BY id", **params):
        r["args"] = json.loads(r.pop("args_json"))
        out.append(r)
    return out


def get_approval(approval_id: int) -> dict | None:
    rows = [a for a in list_approvals() if a["id"] == approval_id]
    return rows[0] if rows else None


def decide(approval_id: int, approved: bool, comment: str = "") -> bool:
    """Mark a pending approval approved/rejected. False if it was not pending (already decided)."""
    _init()
    with _engine.begin() as conn:
        res = conn.execute(text("UPDATE approvals SET status = :s, comment = :m WHERE id = :i AND status = 'pending'"),
                           {"s": "approved" if approved else "rejected", "m": comment, "i": approval_id})
        return res.rowcount == 1


def pending_count(case_id: str) -> int:
    return len(list_approvals(status="pending", case_id=case_id))


def log_tool_call(case_id: str, tool: str, args: dict, result) -> None:
    _init()
    with _engine.begin() as conn:
        conn.execute(text("INSERT INTO tool_call_log (case_id, tool_name, args_json, result_json, ts) "
                          "VALUES (:c, :t, :a, :r, :ts)"),
                     {"c": case_id, "t": tool, "a": json.dumps(args, default=str),
                      "r": json.dumps(result, default=str), "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")})
