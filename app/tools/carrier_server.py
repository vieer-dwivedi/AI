"""MCP server: get_shipment_status, get_freight_quote (Step 5)."""
import time

import httpx
from mcp.server.fastmcp import FastMCP

from app.config import settings

mcp = FastMCP("carrier")

TIMEOUT_S = 5.0
MAX_ATTEMPTS = 3
BACKOFF_S = 0.3


def _call(method: str, path: str, **kwargs) -> tuple[int | None, dict]:
    """HTTP call with timeout and retries on 5xx / network errors.
    Returns (status_code, json). status_code None means the carrier was unreachable."""
    url = settings.carrier_api_url + path
    last_status = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = httpx.request(method, url, timeout=TIMEOUT_S, **kwargs)
            last_status = resp.status_code
            if resp.status_code < 500:
                return resp.status_code, resp.json()
        except httpx.HTTPError:
            last_status = None
        if attempt < MAX_ATTEMPTS - 1:
            time.sleep(BACKOFF_S * (attempt + 1))
    return last_status, {}


@mcp.tool()
def get_shipment_status(tracking_id: str) -> dict:
    """Live shipment status, ETA, last event and delay reason from the carrier."""
    status, data = _call("GET", f"/shipments/{tracking_id}")
    if status == 200:
        return data
    if status == 404:
        return {"error": "TRACKING_NOT_FOUND", "tracking_id": tracking_id}
    return {"error": "CARRIER_UNAVAILABLE", "tracking_id": tracking_id,
            "detail": "Carrier API failed after retries. Escalate to a human."}


@mcp.tool()
def get_freight_quote(origin: str, destination: str, weight_kg: float, mode: str | None = None) -> dict:
    """Freight quotes (price in INR, ETA in days). mode: road | air | sea, or omit for all three."""
    if mode not in (None, "road", "air", "sea"):
        return {"error": "INVALID_MODE", "detail": "mode must be road, air or sea"}
    body = {"origin": origin, "destination": destination, "weight_kg": weight_kg}
    if mode:
        body["mode"] = mode
    status, data = _call("POST", "/quotes", json=body)
    if status == 200:
        return data
    return {"error": "CARRIER_UNAVAILABLE",
            "detail": "Quote service failed after retries. Escalate to a human."}


if __name__ == "__main__":
    mcp.run()