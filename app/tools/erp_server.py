"""MCP server: get_order, get_customer, check_inventory, calculate_penalty (Step 4)."""
from mcp.server.fastmcp import FastMCP
from sqlalchemy import create_engine, text

from app.config import settings

mcp = FastMCP("erp")
_engine = create_engine(settings.database_url)


def compute_penalty(order_value: float, delay_days: int, grace_days: int,
                    pct_per_day: float, cap_pct: float) -> float:
    """Pure deterministic penalty math. Never left to the LLM."""
    billable_days = max(0, delay_days - grace_days)
    return round(min(order_value * pct_per_day / 100 * billable_days,
                     order_value * cap_pct / 100), 2)


def _rows(sql: str, **params) -> list[dict]:
    with _engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(text(sql), params)]


@mcp.tool()
def get_order(order_id: int) -> dict:
    """Order header, lines and shipment tracking id."""
    order = _rows("SELECT * FROM orders WHERE id = :id", id=order_id)
    if not order:
        return {"error": "ORDER_NOT_FOUND", "order_id": order_id}
    lines = _rows("SELECT l.sku, l.qty, p.weight_kg FROM order_lines l "
                  "JOIN products p ON p.sku = l.sku WHERE l.order_id = :id", id=order_id)
    total_weight = round(sum(l["qty"] * l["weight_kg"] for l in lines), 1)
    ship = _rows("SELECT tracking_id, carrier FROM shipments WHERE order_id = :id", id=order_id)
    return {**order[0], "lines": lines, "total_weight_kg": total_weight,
            "shipment": ship[0] if ship else None}


@mcp.tool()
def get_customer(customer_id: str) -> dict:
    """Customer tier, payment terms, contact email and country."""
    rows = _rows("SELECT * FROM customers WHERE id = :id", id=customer_id)
    return rows[0] if rows else {"error": "CUSTOMER_NOT_FOUND", "customer_id": customer_id}


@mcp.tool()
def check_inventory(sku: str, warehouse: str | None = None) -> dict:
    """On-hand, allocated and available stock per warehouse."""
    sql = "SELECT warehouse, on_hand, allocated FROM inventory WHERE sku = :sku"
    params = {"sku": sku}
    if warehouse:
        sql += " AND warehouse = :wh"
        params["wh"] = warehouse
    rows = _rows(sql, **params)
    if not rows:
        return {"error": "SKU_NOT_FOUND", "sku": sku}
    for r in rows:
        r["available"] = r["on_hand"] - r["allocated"]
    return {"sku": sku, "stock": rows}


@mcp.tool()
def calculate_penalty(contract_id: str, delay_days: int, order_value: float) -> dict:
    """Contract late-delivery penalty in INR for a given delay."""
    rows = _rows("SELECT * FROM contracts WHERE id = :id", id=contract_id)
    if not rows:
        return {"error": "CONTRACT_NOT_FOUND", "contract_id": contract_id}
    c = rows[0]
    penalty = compute_penalty(order_value, delay_days, c["grace_days"], c["pct_per_day"], c["cap_pct"])
    return {"contract_id": contract_id, "delay_days": delay_days, "grace_days": c["grace_days"],
            "pct_per_day": c["pct_per_day"], "cap_pct": c["cap_pct"], "penalty_inr": penalty}


if __name__ == "__main__":
    mcp.run()