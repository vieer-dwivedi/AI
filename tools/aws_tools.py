# =====================================================================
# tools/aws_tools.py
# In simple words: the AWS "hands" of the agent. Each function below is a
# LangChain tool the LLM can call (CloudWatch alarms, logs, metrics, and
# ECS / RDS / EC2 details). They ask the backend (mock or real AWS) for
# raw data and boil it down to short JSON, so the model gets facts
# without drowning in thousands of lines.
# =====================================================================
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone

from langchain_core.tools import tool

from tools.cloud_backend import _dims_to_dict, get_backend
from tools.common import fit_list, jdump, safe_tool


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------
# Tool 1: which alarms are firing?
# ---------------------------------------------------------------------
@tool
@safe_tool
def describe_cloudwatch_alarms(alarm_name: str = "", state: str = "ALARM") -> str:
    """List CloudWatch alarms. Use alarm_name (partial match) to focus on one
    alarm, and state ALARM / OK / INSUFFICIENT_DATA (empty string = any state).
    Returns name, state, metric, threshold and the reason the alarm fired."""
    alarms = get_backend().describe_alarms(alarm_name, state)
    return jdump({"count": len(alarms), "alarms": [{
        "name": a["AlarmName"], "state": a["StateValue"],
        "since": a["StateUpdatedTimestamp"], "namespace": a["Namespace"],
        "metric": a["MetricName"], "dimensions": _dims_to_dict(a["Dimensions"]),
        "stat": a["Statistic"], "threshold": a["Threshold"],
        "operator": a["ComparisonOperator"], "reason": a["StateReason"],
        "description": a.get("AlarmDescription", ""),
    } for a in alarms]})


# ---------------------------------------------------------------------
# Tool 2: list the log groups that exist
# ---------------------------------------------------------------------
@tool
@safe_tool
def list_log_groups() -> str:
    """List the CloudWatch log group names available to search."""
    return jdump({"log_groups": get_backend().list_log_groups()})


# ---------------------------------------------------------------------
# Tool 3: read raw log lines (the latest N that match)
# ---------------------------------------------------------------------
@tool
@safe_tool
def get_cloudwatch_logs(log_group: str, filter_pattern: str = "",
                        minutes: int = 30, max_events: int = 25) -> str:
    """Fetch the most recent raw log events from a CloudWatch log group.
    filter_pattern examples: 'ERROR', '?ERROR ?FATAL', 'timeout -healthcheck'.
    For a big-picture summary of many lines, prefer the analyze_logs tool."""
    r = get_backend().filter_log_events(log_group, filter_pattern, minutes)
    evs = r["events"]
    shown = evs[-max_events:]
    rows = [{"time": _iso(e["timestamp"]), "stream": e["logStreamName"].split("/")[-1][:8],
             "message": e["message"][:240]} for e in shown]

    # If it is too big, drop the OLDEST lines until it fits (stays valid JSON).
    def build(items):
        return {"log_group": log_group, "window_minutes": minutes,
                "matched_events": len(evs), "showing_latest": len(items), "events": items}

    return jdump(fit_list(build, rows, 7500), limit=8000)


# ---------------------------------------------------------------------
# Tool 4: metrics, summarised (min/max/avg/trend + a few sample points)
# ---------------------------------------------------------------------
@tool
@safe_tool
def get_cloudwatch_metrics(namespace: str, metric_name: str,
                           dimensions_json: str = "{}", minutes: int = 60) -> str:
    """Get a CloudWatch metric summary (min, max, average, latest, trend, when
    the peak happened and a few sample points). dimensions_json is a JSON
    object text such as '{"ClusterName":"prod-cluster","ServiceName":"checkout-api"}'.
    Example namespaces: AWS/ECS, AWS/RDS, AWS/EC2, AWS/ApplicationELB, CWAgent."""
    dims = json.loads(dimensions_json or "{}")
    s = get_backend().get_metric_series(namespace, metric_name, dims, minutes)
    if not s or not s["values"]:
        return jdump({"namespace": namespace, "metric": metric_name,
                      "result": "no datapoints found for these dimensions/time window"})
    vals, ts = s["values"], s["timestamps"]
    third = max(1, len(vals) // 3)
    first_avg, last_avg = statistics.mean(vals[:third]), statistics.mean(vals[-third:])
    change = last_avg - first_avg
    trend = ("rising" if change > 0.15 * max(abs(first_avg), 1e-9) else
             "falling" if change < -0.15 * max(abs(first_avg), 1e-9) else "flat")
    peak_i = max(range(len(vals)), key=vals.__getitem__)
    step = max(1, len(vals) // 10)
    return jdump({
        "namespace": namespace, "metric": metric_name, "dimensions": s["dimensions"],
        "stat": s["stat"], "unit": s["unit"], "window_minutes": minutes,
        "min": round(min(vals), 2), "max": round(max(vals), 2),
        "avg": round(statistics.mean(vals), 2), "latest": vals[-1],
        "trend": trend, "peak_time": _iso(ts[peak_i]),
        "start_of_window_avg": round(first_avg, 2), "end_of_window_avg": round(last_avg, 2),
        "samples": [{"t": _iso(ts[i]), "v": vals[i]} for i in range(0, len(vals), step)]})


# ---------------------------------------------------------------------
# Tool 5-7: describe ECS service / RDS instance / EC2 instance
# ---------------------------------------------------------------------
@tool
@safe_tool
def describe_ecs_service(cluster: str, service: str) -> str:
    """Describe an ECS service: desired/running/pending counts, deployments
    (with task definition revisions and failed task counts) and recent events."""
    s = get_backend().describe_ecs_service(cluster, service)
    if not s:
        return jdump({"error": f"ECS service {service} not found in cluster {cluster}"})
    return jdump({
        "service": s["serviceName"], "status": s["status"],
        "desired": s["desiredCount"], "running": s["runningCount"],
        "pending": s["pendingCount"], "task_definition": s["taskDefinition"].split("/")[-1],
        "deployments": [{"status": d["status"], "task_definition": d["taskDefinition"].split("/")[-1],
                         "rollout": d.get("rolloutState"), "failed_tasks": d.get("failedTasks"),
                         "created": d["createdAt"]} for d in s["deployments"]],
        "recent_events": [{"time": e["createdAt"], "message": e["message"]} for e in s["events"][:8]]})


@tool
@safe_tool
def describe_rds_instance(db_instance_id: str) -> str:
    """Describe an RDS database instance: engine, class, status, Multi-AZ,
    storage and pending modifications."""
    r = get_backend().describe_rds_instance(db_instance_id)
    if not r:
        return jdump({"error": f"RDS instance {db_instance_id} not found"})
    return jdump({k: r.get(k) for k in (
        "DBInstanceIdentifier", "DBInstanceClass", "Engine", "EngineVersion",
        "DBInstanceStatus", "MultiAZ", "AllocatedStorage", "StorageType",
        "PendingModifiedValues", "LatestRestorableTime")})


@tool
@safe_tool
def describe_ec2_instance(instance_id: str) -> str:
    """Describe an EC2 instance: type, state, IP, tags and attached volumes."""
    i = get_backend().describe_ec2_instance(instance_id)
    if not i:
        return jdump({"error": f"EC2 instance {instance_id} not found"})
    return jdump({"instance_id": i["InstanceId"], "type": i["InstanceType"],
                  "state": i["State"]["Name"], "private_ip": i.get("PrivateIpAddress"),
                  "launched": i.get("LaunchTime"),
                  "tags": {t["Key"]: t["Value"] for t in i.get("Tags", [])},
                  "volumes": [b["Ebs"]["VolumeId"] for b in i.get("BlockDeviceMappings", [])]})


# ---------------------------------------------------------------------
# Tool 8: recent deployments / config changes (very often THE root cause)
# ---------------------------------------------------------------------
@tool
@safe_tool
def list_recent_deployments(service: str, limit: int = 5) -> str:
    """List recent deployments or config changes for a service/host, newest
    first, with time, version, author and change description. Always check
    this: most incidents start right after a change."""
    deps = get_backend().list_deployments(service, limit)
    return jdump({"service": service, "count": len(deps), "deployments": deps})


AWS_TOOLS = [
    describe_cloudwatch_alarms, list_log_groups, get_cloudwatch_logs,
    get_cloudwatch_metrics, describe_ecs_service, describe_rds_instance,
    describe_ec2_instance, list_recent_deployments,
]
