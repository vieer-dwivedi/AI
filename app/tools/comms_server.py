"""MCP server: drafts, emails and other policy-gated write tools (Step 7).

These tools do NOT check approvals. The policy engine (Step 8) gates every call.
"""
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP
from sqlalchemy import create_engine, text

from app.config import settings

mcp = FastMCP("comms")
_engine = create_engine(settings.database_url)

VALID_STATUSES = {"processing", "in_transit", "delivered", "on_hold", "delayed", "cancelled"}

SCHEMA = [
    "CREATE TABLE IF NOT EXISTS drafts (id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, subject TEXT, body TEXT, status TEXT, created_at TEXT)",
    "CREATE TABLE IF NOT EXISTS outbox (id INTEGER PRIMARY KEY AUTOINCREMENT, draft_id INTEGER, sent_at TEXT)",
    "CREATE TABLE IF NOT EXISTS case_notes (id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, note TEXT, created_at TEXT)",
    "CREATE TABLE IF NOT EXISTS reship_orders (id INTEGER PRIMARY KEY AUTOINCREMENT, order_id INTEGER, sku TEXT, qty INTEGER, warehouse TEXT, created_at TEXT)",
    "CREATE TABLE IF NOT EXISTS freight_bookings (id INTEGER PRIMARY KEY AUTOINCREMENT, order_id INTEGER, mode TEXT, origin TEXT, destination TEXT, cost_inr REAL, created_at TEXT)",
    "CREATE TABLE IF NOT EXISTS credit_notes (id INTEGER PRIMARY KEY AUTOINCREMENT, order_id INTEGER, amount_inr REAL, reason TEXT, created_at TEXT)",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _init() -> None:
    with _engine.begin() as conn:
        for stmt in SCHEMA:
            conn.execute(text(stmt))


def _insert(sql: str, **params) -> int:
    """Run an INSERT and return the new row id."""
    _init()
    with _engine.begin() as conn:
        return conn.execute(text(sql), params).lastrowid


def _order_exists(order_id: int) -> bool:
    with _engine.connect() as conn:
        return conn.execute(text("SELECT 1 FROM orders WHERE id = :id"), {"id": order_id}).first() is not None


@mcp.tool()
def draft_customer_email(case_id: str, subject: str, body: str) -> dict:
    """Save an email draft for review. Does not send anything."""
    draft_id = _insert(
        "INSERT INTO drafts (case_id, subject, body, status, created_at) VALUES (:c, :s, :b, 'draft', :t)",
        c=case_id, s=subject, b=body, t=_now())
    return {"draft_id": draft_id, "status": "draft"}


@mcp.tool()
def send_email(draft_id: int) -> dict:
    """Send a saved draft (demo: records it in the outbox, no real email)."""
    _init()
    with _engine.begin() as conn:
        row = conn.execute(text("SELECT status FROM drafts WHERE id = :id"), {"id": draft_id}).first()
        if not row:
            return {"error": "DRAFT_NOT_FOUND", "draft_id": draft_id}
        if row[0] == "sent":
            return {"error": "ALREADY_SENT", "draft_id": draft_id}
        conn.execute(text("UPDATE drafts SET status = 'sent' WHERE id = :id"), {"id": draft_id})
        conn.execute(text("INSERT INTO outbox (draft_id, sent_at) VALUES (:d, :t)"), {"d": draft_id, "t": _now()})
    return {"draft_id": draft_id, "status": "sent"}


@mcp.tool()
def add_case_note(case_id: str, note: str) -> dict:
    """Add an internal note to a case."""
    note_id = _insert("INSERT INTO case_notes (case_id, note, created_at) VALUES (:c, :n, :t)",
                      c=case_id, n=note, t=_now())
    return {"note_id": note_id}


@mcp.tool()
def update_order_status(order_id: int, status: str) -> dict:
    """Set an order's status: processing, in_transit, delivered, on_hold, delayed or cancelled."""
    if status not in VALID_STATUSES:
        return {"error": "INVALID_STATUS", "allowed": sorted(VALID_STATUSES)}
    if not _order_exists(order_id):
        return {"error": "ORDER_NOT_FOUND", "order_id": order_id}
    with _engine.begin() as conn:
        conn.execute(text("UPDATE orders SET status = :s WHERE id = :id"), {"s": status, "id": order_id})
    return {"order_id": order_id, "status": status}


@mcp.tool()
def create_reship_order(order_id: int, sku: str, qty: int, warehouse: str) -> dict:
    """Create a replacement shipment for an order from a given warehouse."""
    if qty <= 0:
        return {"error": "INVALID_QTY"}
    if not _order_exists(order_id):
        return {"error": "ORDER_NOT_FOUND", "order_id": order_id}
    rid = _insert("INSERT INTO reship_orders (order_id, sku, qty, warehouse, created_at) VALUES (:o, :s, :q, :w, :t)",
                  o=order_id, s=sku, q=qty, w=warehouse, t=_now())
    return {"reship_id": rid, "order_id": order_id, "sku": sku, "qty": qty, "warehouse": warehouse}


@mcp.tool()
def book_expedited_freight(order_id: int, mode: str, origin: str, destination: str, cost_inr: float) -> dict:
    """Book expedited freight (air or road) for an order at the quoted cost in INR."""
    if mode not in ("air", "road"):
        return {"error": "INVALID_MODE", "detail": "mode must be air or road"}
    if cost_inr <= 0:
        return {"error": "INVALID_COST"}
    if not _order_exists(order_id):
        return {"error": "ORDER_NOT_FOUND", "order_id": order_id}
    bid = _insert("INSERT INTO freight_bookings (order_id, mode, origin, destination, cost_inr, created_at) "
                  "VALUES (:o, :m, :or, :d, :c, :t)",
                  o=order_id, m=mode, **{"or": origin}, d=destination, c=cost_inr, t=_now())
    return {"booking_id": bid, "order_id": order_id, "mode": mode, "cost_inr": cost_inr}


@mcp.tool()
def issue_credit_note(order_id: int, amount_inr: float, reason: str) -> dict:
    """Issue a credit note in INR against an order."""
    if amount_inr <= 0:
        return {"error": "INVALID_AMOUNT"}
    if not _order_exists(order_id):
        return {"error": "ORDER_NOT_FOUND", "order_id": order_id}
    cid = _insert("INSERT INTO credit_notes (order_id, amount_inr, reason, created_at) VALUES (:o, :a, :r, :t)",
                  o=order_id, a=amount_inr, r=reason, t=_now())
    return {"credit_note_id": cid, "order_id": order_id, "amount_inr": amount_inr}


if __name__ == "__main__":
    mcp.run()