"""Mock third-party carrier REST API (Step 3).

Simulates latency, random 503 errors, and fixed answers for demo scenarios.
"""
import asyncio
import os
import random
import re
from datetime import date, timedelta
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Mock Carrier API")

TODAY = date(2026, 11, 1)           # same fixed date as data/seed.py
FAIL_RATE = float(os.getenv("CARRIER_FAIL_RATE", "0.1"))

# Fixed answers so demo scenarios behave predictably (never randomly fail)
SCENARIOS = {
    "TRK-48213": {
        "status": "delayed",
        "eta": TODAY + timedelta(days=9),
        "last_event": "Container held at Nhava Sheva port",
        "delay_reason": "port congestion",
        "delay_days": 6,
    },
    "TRK-48220": {
        "status": "delivered",
        "eta": TODAY - timedelta(days=1),
        "last_event": "Delivered, signed by 'R. Kumar'",
        "delay_reason": None,
        "delay_days": 0,
    },
}

# price = base + per_kg * weight ; air tuned so 4000 kg is about INR 86,000
RATES = {
    "road": {"base": 2000, "per_kg": 4.0, "eta_days": 6},
    "air": {"base": 8000, "per_kg": 19.5, "eta_days": 2},
    "sea": {"base": 3000, "per_kg": 1.5, "eta_days": 18},
}


async def simulate_network(allow_failure: bool = True):
    await asyncio.sleep(random.uniform(0.1, 0.4))
    if allow_failure and random.random() < FAIL_RATE:
        raise HTTPException(status_code=503, detail="Carrier service temporarily unavailable")


@app.get("/shipments/{tracking_id}")
async def get_shipment(tracking_id: str):
    await simulate_network(allow_failure=tracking_id not in SCENARIOS)

    if tracking_id in SCENARIOS:
        return {"tracking_id": tracking_id, **SCENARIOS[tracking_id], "eta": SCENARIOS[tracking_id]["eta"].isoformat()}

    m = re.fullmatch(r"TRK-(\d{5})", tracking_id)
    if not m or m.group(1) == "00000":
        raise HTTPException(status_code=404, detail="TRACKING_NOT_FOUND")

    # deterministic: same ID always gives the same answer
    n = int(m.group(1))
    return {
        "tracking_id": tracking_id,
        "status": "in_transit",
        "eta": (TODAY + timedelta(days=n % 14)).isoformat(),
        "last_event": "Departed regional hub",
        "delay_reason": None,
        "delay_days": 0,
    }


class QuoteRequest(BaseModel):
    origin: str
    destination: str
    weight_kg: float
    mode: Literal["road", "air", "sea"] | None = None


@app.post("/quotes")
async def get_quotes(req: QuoteRequest):
    await simulate_network()
    modes = [req.mode] if req.mode else list(RATES)
    return {
        "origin": req.origin,
        "destination": req.destination,
        "quotes": [
            {
                "mode": m,
                "price_inr": round(RATES[m]["base"] + RATES[m]["per_kg"] * req.weight_kg, 2),
                "eta_days": RATES[m]["eta_days"],
            }
            for m in modes
        ],
    }