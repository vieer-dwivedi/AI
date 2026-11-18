# ---------------------------------------------------------------------
# tests/test_log_analyzer.py
# Checks the log analyser finds the RIGHT failure signature in each of the
# 4 sample incidents, groups similar messages, and finds the spike time.
# ---------------------------------------------------------------------
import json

import pytest

from tools.log_analyzer import _normalize, _parse, analyze_events, analyze_logs
from tools.cloud_backend import _match_pattern


def run_analyze(group, minutes=60):
    return json.loads(analyze_logs.invoke({"log_group": group, "minutes": minutes}))


@pytest.mark.parametrize("group,expected_first_cause", [
    ("/ecs/checkout-api", "out_of_memory"),
    ("/aws/rds/instance/orders-db/postgresql", "db_connection_exhaustion"),
    ("/ecs/orders-service", "db_connection_exhaustion"),
    ("/ec2/app-server/app.log", "disk_full"),
    ("/ecs/payments-service", "null_pointer_exception"),
])
def test_detects_root_cause_signature(group, expected_first_cause):
    rep = run_analyze(group)
    assert rep["signatures"][0]["name"] == expected_first_cause
    assert rep["signatures"][0]["kind"] == "cause"       # causes always rank above symptoms


def test_report_has_timing_for_ecs_oom():
    rep = run_analyze("/ecs/checkout-api")
    # release happened at 08:35; errors must start after it, spike before the alarm (09:00)
    assert "2026-11-18T08:35" <= rep["first_error_time"] < "2026-11-18T09:00"
    assert rep["error_spike_start"] and rep["error_spike_start"] > "2026-11-18T08:35"
    assert rep["level_counts"]["ERROR"] > 20 and rep["total_events"] > 200


def test_bad_deploy_has_single_dominant_exception():
    rep = run_analyze("/ecs/payments-service")
    assert rep["top_exceptions"][0]["name"] == "NullPointerException"
    top = rep["top_error_messages"][0]
    assert top["count"] > 50 and "NullPointerException" in top["example"]
    assert len(rep["top_error_messages"]) == 1           # all errors collapse into ONE pattern


def test_healthy_logs_have_no_signature():
    # only the quiet first 50 minutes of checkout-api (before the deploy at -25m)
    from tools.cloud_backend import get_backend
    ev = get_backend().filter_log_events("/ecs/checkout-api", minutes=90)["events"]
    quiet = [e for e in ev if e["timestamp"] < ev[-1]["timestamp"] - 30 * 60_000]
    rep = analyze_events(quiet)
    assert rep["signatures"] == [] and rep["first_error_time"] is None


def test_empty_input_is_safe():
    assert analyze_events([])["total_events"] == 0


def test_parse_json_and_plain():
    assert _parse('{"level":"warn","message":"hi"}') == ("WARN", "hi")
    lvl, _ = _parse("2026-11-18 08:41:12 UTC:10.0.1.2(5):app@db:[12]:FATAL:  slots reserved")
    assert lvl == "FATAL"
    assert _parse("some random line")[0] == "UNKNOWN"


def test_normalize_groups_similar_messages():
    a = _normalize("request failed after 31ms from 10.0.0.5 id=9f3a1c77aa12bb")
    b = _normalize("request failed after 877ms from 10.0.9.200 id=1122334455667788")
    assert a == b


def test_log_filter_syntax():
    assert _match_pattern("Error: Disk FULL", "disk full")
    assert _match_pattern("alpha beta", "?gamma ?beta")
    assert not _match_pattern("alpha beta", "alpha -beta")
    assert _match_pattern("a b c", '"b c"') and not _match_pattern("a c b", '"b c"')
