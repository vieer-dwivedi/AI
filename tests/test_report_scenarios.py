# ---------------------------------------------------------------------
# tests/test_report_scenarios.py
# Checks the "hand in your answer" tools and the scenario -> alert builder.
# ---------------------------------------------------------------------
import json

import pytest

from core.scenarios import incident_from_scenario, list_scenarios
from memory.short_term import clear_context, get_section
from tools.report_tools import submit_diagnosis, submit_triage


def run(tool, **kw):
    return json.loads(tool.invoke(kw))


def test_all_scenarios_become_incidents():
    names = list_scenarios()
    assert set(names) == {"ecs_oom", "rds_connections", "ec2_disk_full", "bad_deploy_npe"}
    for n in names:
        inc = incident_from_scenario(n)
        assert inc.service and inc.labels["log_group"] and inc.alarm_name


def test_alert_does_not_leak_root_cause():
    text = (incident_from_scenario("ecs_oom").description + incident_from_scenario("ecs_oom").title).lower()
    assert "oom" not in text and "memory" not in text


def test_unknown_scenario():
    with pytest.raises(ValueError):
        incident_from_scenario("nope")


def test_submit_triage_computes_severity():
    clear_context("T1")
    r = run(submit_triage, incident_id="T1", user_impact="partial_outage", summary="checkout failing",
            affected_service="checkout-api", error_rate_pct=30, revenue_path=True)
    assert r["severity"] == "P1"
    assert get_section("T1", "triage")["severity"] == "P1"


def test_submit_triage_bad_impact_is_reported_not_raised():
    r = run(submit_triage, incident_id="T2", user_impact="catastrophic", summary="x")
    assert "error" in r


def test_submit_diagnosis_clamps_and_normalises():
    r = run(submit_diagnosis, incident_id="T3", root_cause="oom", category="MEMORY_EXHAUSTION",
            confidence=7, evidence="mem 99%")
    assert r["confidence"] == 1.0 and r["category"] == "memory_exhaustion"
    r = run(submit_diagnosis, incident_id="T3", root_cause="?", category="weird", confidence=0.2, evidence="-")
    assert r["category"] == "other"
