# ---------------------------------------------------------------------
# tests/test_api_mcp_evals.py
# Checks the three "outer layers": the webhook API (each alert format,
# security, bad input, background run), the MCP server (a real stdio
# client talks to it) + the allow-list that guards external MCP tools,
# and the eval harness.
# ---------------------------------------------------------------------
import asyncio
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api.webhook as wh
from core.config import reload_settings
from evals.run_evals import run_evals

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def client():
    wh.INCIDENTS.clear()
    with TestClient(wh.app) as c:
        yield c


def wait_done(client, incident_id, tries=100):
    import time
    for _ in range(tries):
        body = client.get(f"/incidents/{incident_id}").json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.2)
    raise AssertionError("investigation did not finish")


# ---------------- webhook normalisers ----------------
def test_service_from_alarm_name():
    assert wh.service_from_alarm("checkout-api-5xx-high") == "checkout-api"
    assert wh.service_from_alarm("orders-db-connections-high") == "orders-db"
    assert wh.service_from_alarm("app-server-disk-used-high") == "app-server"


def test_alertmanager_only_firing_alerts():
    body = {"alerts": [
        {"status": "firing", "fingerprint": "abc", "labels": {"alertname": "HighErrors", "service": "payments-service"},
         "annotations": {"summary": "errors high"}},
        {"status": "resolved", "labels": {"alertname": "Other"}}]}
    incs = wh.from_alertmanager(body)
    assert [i.id for i in incs] == ["AM-abc"] and incs[0].service == "payments-service"


def test_cloudwatch_sns_message_is_unwrapped_and_ok_state_ignored():
    alarm = {"AlarmName": "checkout-api-memory-high", "NewStateValue": "ALARM", "NewStateReason": "x",
             "StateChangeTime": "2026-11-18T08:54:00.000+0000",
             "Trigger": {"Dimensions": [{"name": "ClusterName", "value": "prod-cluster"},
                                        {"name": "ServiceName", "value": "checkout-api"}]}}
    inc = wh.from_cloudwatch({"Message": json.dumps(alarm)})[0]
    assert inc.service == "checkout-api" and inc.labels["ecs_cluster"] == "prod-cluster"
    assert wh.from_cloudwatch({"Message": json.dumps({**alarm, "NewStateValue": "OK"})}) == []


def test_pagerduty_ignores_resolved():
    base = {"event": {"event_type": "incident.resolved", "data": {"id": "P1"}}}
    assert wh.from_pagerduty(base) == []
    base["event"]["event_type"] = "incident.triggered"
    assert wh.from_pagerduty(base)[0].id == "PD-P1"


# ---------------- API behaviour ----------------
def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_unknown_source_and_bad_payload(client):
    assert client.post("/webhook/nope", json={}).status_code == 404
    r = client.post("/webhook/pagerduty", json={"event": {"data": {}}})
    assert r.status_code == 422
    assert client.post("/webhook/manual", content=b"not json").status_code == 422


def test_secret_is_enforced(client, monkeypatch):
    monkeypatch.setenv("WEBHOOK_SECRET", "s3cret"); reload_settings()
    assert client.post("/webhook/manual", json={"title": "x"}).status_code == 401
    ok = client.post("/webhook/manual", json={"title": "x", "service": "payments-service"},
                     headers={"X-Webhook-Secret": "s3cret"})
    assert ok.status_code == 202


def test_manual_alert_is_investigated_in_background(client, monkeypatch):
    monkeypatch.setenv("MOCK_SCENARIO", ""); reload_settings()
    r = client.post("/webhook/manual", json={"id": "INC-API-1", "title": "High 5xx on payments-service",
                                              "service": "payments-service",
                                              "alarm_name": "payments-service-5xx-high"})
    assert r.status_code == 202 and r.json()["accepted"] == ["INC-API-1"]
    body = wait_done(client, "INC-API-1")
    assert body["status"] == "completed"
    assert body["result"]["severity"] in ("P1", "P2")
    assert "bad_deployment" in body["result"]["report_markdown"]
    assert client.get("/incidents").json()["INC-API-1"]["status"] == "completed"


def test_duplicate_page_while_running_is_skipped(client):
    wh.INCIDENTS["INC-DUP"] = {"status": "running", "result": None}
    r = client.post("/webhook/manual", json={"id": "INC-DUP", "title": "again"})
    assert r.json()["skipped_duplicates"] == ["INC-DUP"] and r.json()["accepted"] == []


def test_unknown_incident_404(client):
    assert client.get("/incidents/NOPE").status_code == 404


# ---------------- MCP server + allow-list ----------------
def _stdio_servers():
    return {"irs": {"transport": "stdio", "command": sys.executable, "args": ["-m", "mcp_server.server"],
                    "cwd": str(ROOT)}}


async def test_mcp_server_speaks_mcp_and_runs_tools():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"], cwd=str(ROOT))
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            names = {t.name for t in (await s.list_tools()).tools}
            assert {"server_status", "alarms", "logs_summary", "runbooks", "investigate_scenario"} <= names
            out = await s.call_tool("runbooks", {"query": "ECS tasks OOMKilled exit code 137"})
            assert "ecs-oom-killed.md" in out.content[0].text


async def test_allowlist_blocks_unlisted_external_tools(monkeypatch):
    from tools.mcp_loader import load_allowed_mcp_tools

    monkeypatch.setenv("MCP_SERVERS_JSON", json.dumps(_stdio_servers()))
    monkeypatch.setenv("MCP_TOOL_ALLOWLIST", "")
    reload_settings()
    assert await load_allowed_mcp_tools() == []                       # no allow-list -> nothing attached
    monkeypatch.setenv("MCP_TOOL_ALLOWLIST", "server_status")
    reload_settings()
    tools = await load_allowed_mcp_tools()
    assert [t.name for t in tools] == ["server_status"]               # only the listed one
    out = await tools[0].ainvoke({})                                   # MCP adapters return content blocks
    text = out if isinstance(out, str) else "".join(b.get("text", "") for b in out)
    assert "auto_remediate=False" in text


async def test_broken_mcp_server_does_not_break_the_run(monkeypatch):
    from tools.mcp_loader import load_allowed_mcp_tools

    monkeypatch.setenv("MCP_SERVERS_JSON", json.dumps(
        {"bad": {"transport": "stdio", "command": "/no/such/binary", "args": []}}))
    monkeypatch.setenv("MCP_TOOL_ALLOWLIST", "anything")
    reload_settings()
    assert await load_allowed_mcp_tools() == []


# ---------------- evals ----------------
async def test_eval_harness_scores_perfect_with_offline_model():
    rows = await run_evals()
    assert len(rows) == 4
    failed = {r["scenario"]: [k for k, ok in r["checks"].items() if not ok] for r in rows}
    assert all(r["passed"] == r["total"] for r in rows), failed
