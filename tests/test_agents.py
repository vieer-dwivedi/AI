# ---------------------------------------------------------------------
# tests/test_agents.py
# Runs each specialist agent for real (deep agent + LangGraph loop + tools
# + mock AWS data) with the offline model. Checks the agents REACH THE
# RIGHT CONCLUSION from the tool results, not just that nothing crashed.
# ---------------------------------------------------------------------
import pytest

from agents.diagnosis_agent import run_diagnosis
from agents.remediation_agent import run_remediation
from agents.triage_agent import run_triage
from core.config import reload_settings
from core.scenarios import incident_from_scenario
from memory.short_term import clear_context

EXPECTED = {   # scenario -> (root-cause category, a word that must be in the trigger, action types)
    "ecs_oom": ("memory_exhaustion", "d-8K2LQ9ZPA", {"rollback_deployment", "increase_memory"}),
    "rds_connections": ("connection_exhaustion", "reporting-job", {"stop_job", "add_connection_pooling"}),
    "ec2_disk_full": ("disk_exhaustion", "cfg-OPS-5521", {"reduce_log_level", "extend_volume"}),
    "bad_deploy_npe": ("bad_deployment", "d-PAY380XQ", {"rollback_deployment"}),
}


@pytest.fixture(autouse=True)
def _fresh():
    clear_context()
    yield
    clear_context()


def use_scenario(monkeypatch, name):
    monkeypatch.setenv("MOCK_SCENARIO", name)
    reload_settings()
    return incident_from_scenario(name)


@pytest.mark.parametrize("name", EXPECTED)
async def test_triage_agent(monkeypatch, name):
    inc = use_scenario(monkeypatch, name)
    triage = await run_triage(inc)
    assert triage is not None, "triage agent never called submit_triage"
    assert triage["severity"] in {"P1", "P2", "P3"}
    assert triage["affected_service"] == inc.service


@pytest.mark.parametrize("name", EXPECTED)
async def test_diagnosis_agent_finds_root_cause(monkeypatch, name):
    category, trigger_word, _ = EXPECTED[name]
    inc = use_scenario(monkeypatch, name)
    diag = await run_diagnosis(inc, None)
    assert diag is not None, "diagnosis agent never called submit_diagnosis"
    assert diag["category"] == category
    assert trigger_word in diag["trigger"]
    assert diag["confidence"] >= 0.8 and diag["evidence"]


@pytest.mark.parametrize("name", EXPECTED)
async def test_remediation_agent_proposes_but_never_executes(monkeypatch, name):
    category, trigger_word, wanted = EXPECTED[name]
    inc = use_scenario(monkeypatch, name)
    diag = await run_diagnosis(inc, None)
    actions = await run_remediation(inc, diag)
    assert wanted <= {a["action_type"] for a in actions}
    assert all(a["status"] == "proposed" for a in actions)
    assert not any("<" in a["steps"] and "task-definition>" in a["steps"] for a in actions), \
        "commands should use real names, not placeholders"
