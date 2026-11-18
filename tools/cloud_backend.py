# =====================================================================
# tools/cloud_backend.py
# In simple words: the "data source" behind the AWS tools. There are two:
#
#   MockCloudBackend - reads the fake AWS data in sample_data/scenarios.
#                      Safe, free, works offline.  (CLOUD_BACKEND=mock)
#   AwsCloudBackend  - calls the real AWS APIs with boto3.
#                      (CLOUD_BACKEND=aws, needs AWS credentials)
#
# Both return data in the SAME shape (the real boto3 shape), so the tools
# on top never care which one is behind them.
# =====================================================================
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from core.config import get_settings


# ---------------------------------------------------------------------
# Simple log filter: a tiny subset of CloudWatch filter syntax
#   word1 word2   -> both words must appear      (AND)
#   ?word1 ?word2 -> at least one must appear     (OR)
#   -word         -> word must NOT appear
#   "exact phrase"-> quoted phrase
# The real AWS backend passes the pattern straight to CloudWatch instead.
# ---------------------------------------------------------------------
def _match_pattern(message: str, pattern: str) -> bool:
    if not pattern.strip():
        return True
    msg = message.lower()
    terms = [t[0] or t[1] for t in re.findall(r'"([^"]+)"|(\S+)', pattern)]
    must = [t.lower() for t in terms if not t.startswith(("?", "-"))]
    anyof = [t[1:].lower() for t in terms if t.startswith("?")]
    never = [t[1:].lower() for t in terms if t.startswith("-") and len(t) > 1]
    if any(t not in msg for t in must):
        return False
    if anyof and not any(t in msg for t in anyof):
        return False
    return not any(t in msg for t in never)


def _dims_to_dict(dimensions: list[dict]) -> dict[str, str]:
    return {d["Name"]: d["Value"] for d in dimensions}


# =====================================================================
#                           MOCK BACKEND
# =====================================================================
class MockCloudBackend:
    """Pretend AWS account built from sample_data/scenarios/*.json."""

    def __init__(self, scenario: str = ""):
        folder = get_settings().sample_data_path / "scenarios"
        files = sorted(folder.glob("*.json"))
        if not files:
            raise FileNotFoundError(
                f"No sample data in {folder}. Run: python sample_data/generate_sample_data.py")
        self.scenarios: dict[str, dict] = {}
        for f in files:
            data = json.loads(f.read_text())
            if scenario and data["name"] != scenario:
                continue
            self.scenarios[data["name"]] = data
        if scenario and not self.scenarios:
            names = [f.stem for f in files]
            raise ValueError(f"Unknown MOCK_SCENARIO '{scenario}'. Choose from {names}")

    # -- small lookups ------------------------------------------------
    def _each(self, key: str):
        for sc in self.scenarios.values():
            for item in sc.get(key, []):
                yield sc, item

    # -- alarms ---------------------------------------------------------
    def describe_alarms(self, alarm_name: str = "", state: str = "") -> list[dict]:
        out = []
        for _, a in self._each("alarms"):
            if alarm_name and alarm_name.lower() not in a["AlarmName"].lower():
                continue
            if state and a["StateValue"].upper() != state.upper():
                continue
            out.append(a)
        return out

    # -- logs -----------------------------------------------------------
    def list_log_groups(self) -> list[str]:
        return sorted({g for sc in self.scenarios.values() for g in sc["log_groups"]})

    def filter_log_events(self, log_group: str, filter_pattern: str = "",
                          minutes: int = 60, limit: int = 2000) -> dict:
        for sc in self.scenarios.values():
            if log_group in sc["log_groups"]:
                end = sc["anchor_ms"]            # "now" in the sample world
                start = end - minutes * 60_000
                events = [e for e in sc["log_groups"][log_group]
                          if start <= e["timestamp"] <= end
                          and _match_pattern(e["message"], filter_pattern)]
                events.sort(key=lambda e: e["timestamp"])
                return {"log_group": log_group, "events": events[-limit:],
                        "window_start_ms": start, "window_end_ms": end}
        raise ValueError(f"Log group '{log_group}' not found. Known: {self.list_log_groups()}")

    # -- metrics --------------------------------------------------------
    def get_metric_series(self, namespace: str, metric: str,
                          dimensions: dict[str, str], minutes: int = 60) -> dict | None:
        for sc, m in self._each("metrics"):
            if m["namespace"] != namespace or m["metric"] != metric:
                continue
            if any(m["dimensions"].get(k) != v for k, v in dimensions.items()):
                continue
            end = sc["anchor_ms"]
            start = end - minutes * 60_000
            pts = [(t, v) for t, v in zip(m["timestamps"], m["values"]) if start <= t <= end]
            return {"namespace": namespace, "metric": metric, "dimensions": m["dimensions"],
                    "stat": m["stat"], "period": m["period"], "unit": m["unit"],
                    "timestamps": [p[0] for p in pts], "values": [p[1] for p in pts]}
        return None

    # -- ECS / RDS / EC2 ------------------------------------------------
    def describe_ecs_service(self, cluster: str, service: str) -> dict | None:
        for _, s in self._each("ecs_services"):
            if s["serviceName"] == service and cluster in s["clusterArn"]:
                return s
        return None

    def describe_rds_instance(self, db_id: str) -> dict | None:
        return next((r for _, r in self._each("rds_instances")
                     if r["DBInstanceIdentifier"] == db_id), None)

    def describe_ec2_instance(self, instance_id: str) -> dict | None:
        return next((i for _, i in self._each("ec2_instances")
                     if i["InstanceId"] == instance_id), None)

    def list_deployments(self, service: str, limit: int = 5) -> list[dict]:
        items = [d for _, d in self._each("deployments") if d["service"] == service]
        items.sort(key=lambda d: d["createdAt"], reverse=True)
        return items[:limit]


# =====================================================================
#                            REAL AWS BACKEND
# (Written against the boto3 docs. Not exercised against a live AWS
#  account in this project's tests - try it on a sandbox account first.)
# =====================================================================
class AwsCloudBackend:
    def __init__(self, region: str):
        import boto3  # imported here so mock mode never needs AWS setup

        self.region = region
        self.logs = boto3.client("logs", region_name=region)
        self.cw = boto3.client("cloudwatch", region_name=region)
        self.ecs = boto3.client("ecs", region_name=region)
        self.rds = boto3.client("rds", region_name=region)
        self.ec2 = boto3.client("ec2", region_name=region)

    def describe_alarms(self, alarm_name: str = "", state: str = "") -> list[dict]:
        kw: dict[str, Any] = {"AlarmTypes": ["MetricAlarm"]}
        if alarm_name:
            kw["AlarmNamePrefix"] = alarm_name
        if state:
            kw["StateValue"] = state.upper()
        out = []
        for page in self.cw.get_paginator("describe_alarms").paginate(**kw):
            out.extend(page.get("MetricAlarms", []))
        return out

    def list_log_groups(self) -> list[str]:
        names = []
        for page in self.logs.get_paginator("describe_log_groups").paginate():
            names += [g["logGroupName"] for g in page["logGroups"]]
        return sorted(names)

    def filter_log_events(self, log_group: str, filter_pattern: str = "",
                          minutes: int = 60, limit: int = 2000) -> dict:
        end = int(time.time() * 1000)
        start = end - minutes * 60_000
        events: list[dict] = []
        kw: dict[str, Any] = {"logGroupName": log_group, "startTime": start, "endTime": end}
        if filter_pattern:
            kw["filterPattern"] = filter_pattern
        for page in self.logs.get_paginator("filter_log_events").paginate(**kw):
            events.extend(page["events"])
            if len(events) >= limit:
                break
        return {"log_group": log_group, "events": events[:limit],
                "window_start_ms": start, "window_end_ms": end}

    def get_metric_series(self, namespace: str, metric: str,
                          dimensions: dict[str, str], minutes: int = 60) -> dict | None:
        import datetime as dt

        end = dt.datetime.now(dt.timezone.utc)
        resp = self.cw.get_metric_data(
            MetricDataQueries=[{
                "Id": "m1",
                "MetricStat": {
                    "Metric": {"Namespace": namespace, "MetricName": metric,
                               "Dimensions": [{"Name": k, "Value": v} for k, v in dimensions.items()]},
                    "Period": 60, "Stat": "Average"}}],
            StartTime=end - dt.timedelta(minutes=minutes), EndTime=end)
        r = resp["MetricDataResults"][0]
        if not r["Values"]:
            return None
        pairs = sorted(zip(r["Timestamps"], r["Values"]))
        return {"namespace": namespace, "metric": metric, "dimensions": dimensions,
                "stat": "Average", "period": 60, "unit": "",
                "timestamps": [int(t.timestamp() * 1000) for t, _ in pairs],
                "values": [v for _, v in pairs]}

    def describe_ecs_service(self, cluster: str, service: str) -> dict | None:
        r = self.ecs.describe_services(cluster=cluster, services=[service])
        return r["services"][0] if r["services"] else None

    def describe_rds_instance(self, db_id: str) -> dict | None:
        r = self.rds.describe_db_instances(DBInstanceIdentifier=db_id)
        return r["DBInstances"][0] if r["DBInstances"] else None

    def describe_ec2_instance(self, instance_id: str) -> dict | None:
        r = self.ec2.describe_instances(InstanceIds=[instance_id])
        res = r["Reservations"]
        return res[0]["Instances"][0] if res else None

    def list_deployments(self, service: str, limit: int = 5) -> list[dict]:
        """Best effort: look through every ECS cluster for the service and
        return its deployments (newest first)."""
        out: list[dict] = []
        for c in self.ecs.list_clusters()["clusterArns"]:
            try:
                svcs = self.ecs.describe_services(cluster=c, services=[service])["services"]
            except Exception:  # noqa: BLE001
                continue
            for s in svcs:
                for d in s.get("deployments", []):
                    out.append({"service": service, "deploymentId": d["id"],
                                "status": d.get("rolloutState", d["status"]),
                                "createdAt": str(d["createdAt"]),
                                "taskDefinition": d["taskDefinition"].split("/")[-1],
                                "image": "see task definition", "author": "n/a",
                                "commit": "n/a", "description": "ECS deployment"})
        out.sort(key=lambda d: d["createdAt"], reverse=True)
        return out[:limit]


# ---------------------------------------------------------------------
# Factory: pick the backend from .env (cached so JSON loads only once).
# ---------------------------------------------------------------------
_cache: dict[tuple, Any] = {}


def get_backend():
    s = get_settings()
    key = (s.cloud_backend.lower(), s.mock_scenario, s.aws_region)
    if key not in _cache:
        if key[0] == "mock":
            _cache[key] = MockCloudBackend(s.mock_scenario)
        elif key[0] == "aws":
            _cache[key] = AwsCloudBackend(s.aws_region)
        else:
            raise ValueError(f"CLOUD_BACKEND must be 'mock' or 'aws', got '{key[0]}'")
    return _cache[key]
