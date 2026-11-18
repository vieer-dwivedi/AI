# ---------------------------------------------------------------------
# tests/test_aws_tools.py
# Checks the mock "AWS account" and the AWS tools: alarms, logs, metrics,
# ECS/RDS/EC2 details, deployments, and that errors never crash a tool.
# ---------------------------------------------------------------------
import json

import pytest

from core.config import reload_settings
from tools import cloud_backend
from tools.aws_tools import (AWS_TOOLS, describe_cloudwatch_alarms, describe_ec2_instance,
                             describe_ecs_service, describe_rds_instance, get_cloudwatch_logs,
                             get_cloudwatch_metrics, list_log_groups, list_recent_deployments)
from tools.azure_tools import azure_aks_pods, azure_list_alerts, azure_query_logs


def run(tool, **kw):
    return json.loads(tool.invoke(kw))


def test_backend_loads_all_scenarios():
    b = cloud_backend.get_backend()
    assert set(b.scenarios) == {"ecs_oom", "rds_connections", "ec2_disk_full", "bad_deploy_npe"}


def test_scenario_filter(monkeypatch):
    monkeypatch.setenv("MOCK_SCENARIO", "ecs_oom"); reload_settings()
    assert list(cloud_backend.get_backend().scenarios) == ["ecs_oom"]
    assert run(describe_cloudwatch_alarms)["count"] == 2


def test_unknown_scenario_is_clear_error(monkeypatch):
    monkeypatch.setenv("MOCK_SCENARIO", "nope"); reload_settings()
    with pytest.raises(ValueError, match="Unknown MOCK_SCENARIO"):
        cloud_backend.get_backend()


def test_tool_schemas_are_llm_friendly():
    # every tool needs a name + description, or the LLM can't use it
    for t in AWS_TOOLS:
        assert t.name and len(t.description) > 20
    assert set(describe_cloudwatch_alarms.args) == {"alarm_name", "state"}


def test_alarms_all_and_by_name():
    assert run(describe_cloudwatch_alarms)["count"] == 6            # 6 firing across 4 scenarios
    one = run(describe_cloudwatch_alarms, alarm_name="payments-service-5xx")
    assert one["count"] == 1 and one["alarms"][0]["state"] == "ALARM"
    assert run(describe_cloudwatch_alarms, state="OK")["count"] >= 4


def test_logs_filtering():
    r = run(get_cloudwatch_logs, log_group="/ecs/checkout-api", filter_pattern="OutOfMemoryError", minutes=60)
    assert r["matched_events"] > 5 and "OutOfMemoryError" in r["events"][0]["message"]
    none = run(get_cloudwatch_logs, log_group="/ecs/checkout-api", filter_pattern="nonexistent-xyz")
    assert none["matched_events"] == 0
    # exclusion + OR syntax
    r2 = run(get_cloudwatch_logs, log_group="/ecs/checkout-api", filter_pattern="?ERROR ?WARN -503", minutes=60)
    assert all("503" not in e["message"] for e in r2["events"])


def test_time_window_is_respected():
    last5 = run(get_cloudwatch_logs, log_group="/ecs/checkout-api", minutes=5, max_events=500)
    last60 = run(get_cloudwatch_logs, log_group="/ecs/checkout-api", minutes=60, max_events=500)
    assert 0 < last5["matched_events"] < last60["matched_events"]


def test_metric_summary_shows_memory_rising():
    r = run(get_cloudwatch_metrics, namespace="AWS/ECS", metric_name="MemoryUtilization",
            dimensions_json='{"ClusterName":"prod-cluster","ServiceName":"checkout-api"}', minutes=60)
    assert r["trend"] == "rising" and r["max"] > 95 and len(r["samples"]) >= 5


def test_metric_not_found_is_graceful():
    r = run(get_cloudwatch_metrics, namespace="AWS/ECS", metric_name="Nope", dimensions_json="{}")
    assert "no datapoints" in r["result"]


def test_service_descriptions():
    ecs = run(describe_ecs_service, cluster="prod-cluster", service="checkout-api")
    assert ecs["desired"] == 4 and ecs["running"] == 2 and ecs["task_definition"].endswith(":42")
    assert run(describe_rds_instance, db_instance_id="orders-db")["Engine"] == "postgres"
    assert run(describe_ec2_instance, instance_id="i-0a1b2c3d4e5f67890")["state"] == "running"
    assert "error" in run(describe_rds_instance, db_instance_id="ghost")


def test_deployments_newest_first():
    d = run(list_recent_deployments, service="checkout-api")
    assert d["count"] == 2 and "v2.14.0" in d["deployments"][0]["image"]


def test_errors_never_crash_a_tool():
    r = run(get_cloudwatch_logs, log_group="/does/not/exist")      # backend raises -> JSON error
    assert "error" in r and "not found" in r["error"]
    assert "log_groups" in run(list_log_groups)


def test_azure_mock_tools():
    assert run(azure_list_alerts)["count"] == 1
    assert run(azure_aks_pods)["pods"][0]["last_state"] == "OOMKilled"
    assert run(azure_query_logs, contains="137")["matched"] == 1
