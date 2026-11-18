# =====================================================================
# core/scenarios.py
# In simple words: turns one of our fake AWS "scenarios" into the alert
# that would have woken somebody up. The alert text only says WHAT fired
# (like a real CloudWatch alarm) - it never reveals the root cause, so the
# agent has to actually investigate.
# =====================================================================
from __future__ import annotations

import json

from core.config import get_settings
from core.schemas import Incident

# What the on-call person would see in the page, per scenario.
_ALERTS = {
    "ecs_oom": dict(
        alarm="checkout-api-5xx-high", service="checkout-api",
        title="High 5xx error rate on checkout-api",
        labels={"log_group": "/ecs/checkout-api", "ecs_cluster": "prod-cluster"}),
    "rds_connections": dict(
        alarm="orders-service-5xx-high", service="orders-service",
        title="High 5xx error rate on orders-service",
        labels={"log_group": "/ecs/orders-service", "ecs_cluster": "prod-cluster",
                "db_instance": "orders-db"}),
    "ec2_disk_full": dict(
        alarm="app-server-disk-used-high", service="app-server",
        title="Disk usage critical on app-server",
        labels={"log_group": "/ec2/app-server/app.log", "instance_id": "i-0a1b2c3d4e5f67890"}),
    "bad_deploy_npe": dict(
        alarm="payments-service-5xx-high", service="payments-service",
        title="High 5xx error rate on payments-service",
        labels={"log_group": "/ecs/payments-service", "ecs_cluster": "prod-cluster"}),
}


def list_scenarios() -> list[str]:
    folder = get_settings().sample_data_path / "scenarios"
    return sorted(p.stem for p in folder.glob("*.json"))


def load_scenario(name: str) -> dict:
    path = get_settings().sample_data_path / "scenarios" / f"{name}.json"
    if not path.exists():
        raise ValueError(f"Unknown scenario '{name}'. Available: {list_scenarios()}")
    return json.loads(path.read_text())


def incident_from_scenario(name: str) -> Incident:
    """Build the Incident (the 'page') for a scenario."""
    data = load_scenario(name)
    alert = _ALERTS.get(name)
    if alert is None:
        raise ValueError(f"No alert template for scenario '{name}'")
    alarm = next(a for a in data["alarms"] if a["AlarmName"] == alert["alarm"])
    return Incident(
        id=f"INC-{name.upper().replace('_', '-')}",
        source="cloudwatch",
        title=alert["title"],
        description=f"{alarm['AlarmDescription']}. {alarm['StateReason']}",
        service=alert["service"],
        environment="prod",
        alarm_name=alert["alarm"],
        triggered_at=alarm["StateUpdatedTimestamp"],
        labels=alert["labels"],
        raw={"alarm": alarm},
    )
