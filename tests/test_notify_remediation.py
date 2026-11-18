# ---------------------------------------------------------------------
# tests/test_notify_remediation.py
# Checks Slack dry-run + real-webhook path (mocked), the severity rubric,
# and - most important - the SAFETY RULES of remediation.
# ---------------------------------------------------------------------
import json

import pytest

from core.config import get_settings, reload_settings
from memory import short_term
from tools import slack_notifier
from tools.remediation_tools import execute_remediation, list_proposed_actions, propose_remediation
from tools.triage_tools import compute_severity, severity_rubric


def j(tool, **kw):
    return json.loads(tool.invoke(kw))


# ---------------- Slack ----------------
def test_slack_dry_run_writes_outbox():
    r = slack_notifier.send_slack("DB connections maxed", "critical", "INC-7")
    assert r["delivered"] == "outbox"
    line = json.loads((get_settings().output_path / "slack_outbox.jsonl").read_text().splitlines()[-1])
    assert "[INC-7]" in line["text"] and ":rotating_light:" in line["text"]


def test_slack_real_webhook_path(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.example.test/xyz"); reload_settings()
    sent = {}

    class FakeResp:
        status_code = 200
        def raise_for_status(self): pass

    monkeypatch.setattr(slack_notifier.httpx, "post",
                        lambda url, json, timeout: sent.update(url=url, body=json) or FakeResp())
    out = j(slack_notifier.post_slack_update, message="hi", severity="info", incident_id="INC-8")
    assert out["delivered"] == "slack" and sent["url"].endswith("/xyz") and "hi" in sent["body"]["text"]


def test_slack_failure_is_reported_not_raised(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.example.test/xyz"); reload_settings()
    def boom(*a, **k): raise RuntimeError("network down")
    monkeypatch.setattr(slack_notifier.httpx, "post", boom)
    assert "network down" in j(slack_notifier.post_slack_update, message="x")["error"]


# ---------------- Severity ----------------
@pytest.mark.parametrize("impact,env,err,rev,expected", [
    ("full_outage", "prod", 100, False, "P1"),
    ("partial_outage", "prod", 40, False, "P1"),
    ("partial_outage", "prod", 5, False, "P2"),
    ("partial_outage", "prod", 5, True, "P1"),
    ("degraded", "prod", 2, False, "P3"),
    ("degraded", "prod", 2, True, "P2"),
    ("none", "prod", 0, False, "P4"),
    ("full_outage", "staging", 100, True, "P3"),       # non-prod never above P3
])
def test_severity_rules(impact, env, err, rev, expected):
    assert compute_severity(impact, env, err, rev)["severity"] == expected


def test_severity_bad_input_is_reported():
    assert "user_impact must be one of" in j(severity_rubric, user_impact="catastrophic")["error"]


# ---------------- Remediation safety ----------------
def propose(risk="low", **kw):
    return j(propose_remediation, incident_id="INC-1", action_type="rollback_deployment",
             target="ecs:prod-cluster/checkout-api", rationale="memory up after deploy",
             risk=risk, steps="aws ecs update-service ...", **kw)


@pytest.fixture(autouse=True)
def clean():
    short_term.clear_context(); yield; short_term.clear_context()


def test_propose_records_but_does_not_execute():
    out = propose()
    assert out["proposed"]["action_id"] == "ACT-1" and out["proposed"]["status"] == "proposed"
    assert len(j(list_proposed_actions, incident_id="INC-1")["actions"]) == 1


def test_validation_of_risk_and_action_type():
    assert "risk must be" in propose(risk="yolo")["error"]
    assert "action_type must be" in j(propose_remediation, incident_id="i", action_type="rm -rf /",
                                      target="x", rationale="y")["error"]


def test_execute_blocked_by_default():
    propose()
    r = j(execute_remediation, incident_id="INC-1", action_id="ACT-1")
    assert r["status"] == "blocked" and "AUTO_REMEDIATE" in r["reason"]
    assert short_term.find_action("INC-1", "ACT-1")["status"] == "proposed"      # untouched


def test_execute_simulated_when_enabled_on_mock(monkeypatch):
    monkeypatch.setenv("AUTO_REMEDIATE", "true"); reload_settings()
    propose()
    r = j(execute_remediation, incident_id="INC-1", action_id="ACT-1")
    assert r["status"] == "executed_simulated"


def test_high_risk_always_blocked_even_when_enabled(monkeypatch):
    monkeypatch.setenv("AUTO_REMEDIATE", "true"); reload_settings()
    propose(risk="high")
    assert j(execute_remediation, incident_id="INC-1", action_id="ACT-1")["status"] == "blocked"


def test_real_aws_execution_is_not_implemented(monkeypatch):
    monkeypatch.setenv("AUTO_REMEDIATE", "true"); monkeypatch.setenv("CLOUD_BACKEND", "aws"); reload_settings()
    propose()
    assert j(execute_remediation, incident_id="INC-1", action_id="ACT-1")["status"] == "not_implemented"


def test_unknown_action_id():
    assert j(execute_remediation, incident_id="INC-1", action_id="ACT-99")["status"] == "error"
