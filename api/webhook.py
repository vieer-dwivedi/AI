# =====================================================================
# api/webhook.py
# In simple words: the "front door". PagerDuty, Prometheus Alertmanager or
# an AWS CloudWatch alarm (via SNS) sends an HTTP POST here. We turn it
# into our standard Incident, answer immediately ("202 accepted") and run
# the investigation in the background. You fetch the result later with
# GET /incidents/<id>.
#
#   POST /webhook/pagerduty      PagerDuty v3 webhook
#   POST /webhook/alertmanager   Prometheus Alertmanager
#   POST /webhook/cloudwatch     CloudWatch alarm through SNS
#   POST /webhook/manual         our own simple JSON (for curl tests)
#   GET  /incidents, /incidents/{id}, /health
# If WEBHOOK_SECRET is set in .env, callers must send  X-Webhook-Secret.
# =====================================================================
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request

from agents.orchestrator import run_incident_async
from core.config import get_settings
from core.schemas import Incident

log = logging.getLogger(__name__)
app = FastAPI(title="Incident Response Agent", version="1.0")

# incident id -> {"status": queued|running|completed|failed, "result": {...}}
INCIDENTS: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------
# Security: optional shared secret
# ---------------------------------------------------------------------
async def require_secret(request: Request) -> None:
    secret = get_settings().webhook_secret
    if secret and not hmac.compare_digest(request.headers.get("x-webhook-secret", ""), secret):
        raise HTTPException(status_code=401, detail="invalid or missing X-Webhook-Secret")


# ---------------------------------------------------------------------
# Normalisers: every source's JSON -> our Incident
# ---------------------------------------------------------------------
_METRIC_WORDS = {"5xx", "4xx", "high", "low", "cpu", "memory", "disk", "used", "connections", "errors",
                 "latency", "utilization", "throttles", "alarm"}


def service_from_alarm(alarm_name: str) -> str:
    """'checkout-api-5xx-high' -> 'checkout-api' (drop trailing metric words)."""
    parts = alarm_name.split("-")
    while len(parts) > 1 and parts[-1].lower() in _METRIC_WORDS:
        parts.pop()
    return "-".join(parts)


def _short(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


def from_pagerduty(body: dict) -> list[Incident]:
    event = body.get("event") or {}
    if event.get("event_type") not in (None, "incident.triggered"):
        return []                                           # ignore acknowledged / resolved events
    data = event.get("data") or {}
    if not data.get("id"):
        raise ValueError("PagerDuty payload has no event.data.id")
    service = (data.get("service") or {}).get("summary") or ""
    return [Incident(id=f"PD-{data['id']}", source="pagerduty", title=data.get("title", "PagerDuty incident"),
                     description=data.get("description") or data.get("title", ""),
                     service=service or None, triggered_at=event.get("occurred_at") or data.get("created_at") or
                     Incident.model_fields["triggered_at"].default_factory(), raw=body)]


def from_alertmanager(body: dict) -> list[Incident]:
    out = []
    for a in body.get("alerts", []):
        if a.get("status") != "firing":
            continue
        lab, ann = a.get("labels", {}), a.get("annotations", {})
        name = lab.get("alertname", "alert")
        out.append(Incident(
            id=f"AM-{a.get('fingerprint') or _short(json.dumps(lab, sort_keys=True))}", source="alertmanager",
            title=ann.get("summary") or name, description=ann.get("description", ""),
            service=lab.get("service") or lab.get("job") or service_from_alarm(name),
            environment=lab.get("env") or lab.get("environment") or "prod", alarm_name=name,
            triggered_at=a.get("startsAt") or Incident.model_fields["triggered_at"].default_factory(),
            labels={k: v for k, v in lab.items() if k in ("log_group", "ecs_cluster", "db_instance", "instance_id")},
            raw=a))
    return out


def from_cloudwatch(body: dict) -> list[Incident]:
    msg = body.get("Message")
    alarm = json.loads(msg) if isinstance(msg, str) else (msg or body)       # SNS wraps it as a string
    if alarm.get("NewStateValue") != "ALARM":
        return []
    name = alarm["AlarmName"]
    dims = {d["name"]: d["value"] for d in (alarm.get("Trigger", {}).get("Dimensions") or [])}
    service = dims.get("ServiceName") or dims.get("DBInstanceIdentifier") or service_from_alarm(name)
    labels = {}
    if "ClusterName" in dims:
        labels["ecs_cluster"] = dims["ClusterName"]
    if "InstanceId" in dims:
        labels["instance_id"] = dims["InstanceId"]
    if "DBInstanceIdentifier" in dims:
        labels["db_instance"] = dims["DBInstanceIdentifier"]
    return [Incident(id=f"CW-{name}-{_short(alarm.get('StateChangeTime', ''))}", source="cloudwatch",
                     title=alarm.get("AlarmDescription") or name,
                     description=alarm.get("NewStateReason", ""), service=service, alarm_name=name,
                     triggered_at=alarm.get("StateChangeTime") or Incident.model_fields["triggered_at"].default_factory(),
                     labels=labels, raw=alarm)]


def from_manual(body: dict) -> list[Incident]:
    body = dict(body)
    body.setdefault("id", f"MAN-{_short(json.dumps(body, sort_keys=True))}")
    body.setdefault("source", "manual")
    return [Incident(**body)]


NORMALISERS = {"pagerduty": from_pagerduty, "alertmanager": from_alertmanager,
               "cloudwatch": from_cloudwatch, "manual": from_manual}


# ---------------------------------------------------------------------
# Background job + endpoints
# ---------------------------------------------------------------------
async def _investigate(incident: Incident) -> None:
    INCIDENTS[incident.id]["status"] = "running"
    try:
        result = await run_incident_async(incident)
        INCIDENTS[incident.id].update(status=result.status, result=result.model_dump())
    except Exception as e:  # noqa: BLE001
        log.exception("investigation crashed")
        INCIDENTS[incident.id].update(status="failed", result={"error": f"{type(e).__name__}: {e}"})


@app.post("/webhook/{source}", status_code=202, dependencies=[Depends(require_secret)])
async def receive(source: str, request: Request, background: BackgroundTasks) -> dict:
    if source not in NORMALISERS:
        raise HTTPException(404, f"unknown source '{source}'. Use one of {sorted(NORMALISERS)}")
    try:
        body = await request.json()
        incidents = NORMALISERS[source](body)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - bad payloads are the caller's problem, not a 500
        raise HTTPException(422, f"could not parse {source} payload: {type(e).__name__}: {e}")
    accepted, skipped = [], []
    for inc in incidents:
        if inc.id in INCIDENTS and INCIDENTS[inc.id]["status"] in ("queued", "running"):
            skipped.append(inc.id)                           # duplicate page while still investigating
            continue
        INCIDENTS[inc.id] = {"status": "queued", "result": None}
        background.add_task(_investigate, inc)
        accepted.append(inc.id)
    return {"accepted": accepted, "skipped_duplicates": skipped,
            "ignored": not incidents}


@app.get("/incidents")
async def list_incidents() -> dict:
    return {i: {"status": v["status"]} for i, v in INCIDENTS.items()}


@app.get("/incidents/{incident_id}")
async def get_incident(incident_id: str) -> dict:
    if incident_id not in INCIDENTS:
        raise HTTPException(404, "unknown incident")
    return {"incident_id": incident_id, **INCIDENTS[incident_id]}


@app.get("/health")
async def health() -> dict:
    s = get_settings()
    return {"status": "ok", "llm_model": s.llm_model, "cloud_backend": s.cloud_backend,
            "auto_remediate": s.auto_remediate, "open_incidents": sum(
                v["status"] in ("queued", "running") for v in INCIDENTS.values())}
