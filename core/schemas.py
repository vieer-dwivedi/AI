# =====================================================================
# core/schemas.py
# In simple words: the "shape" of the data that flows through the system.
#  - Incident: a normalised alert (no matter if it came from PagerDuty,
#    Alertmanager, CloudWatch or was typed by hand).
#  - IncidentResult: what the agent produces at the end.
# =====================================================================
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Incident(BaseModel):
    """One alert, in a common format."""

    id: str
    source: str = "manual"  # pagerduty | alertmanager | cloudwatch | manual
    title: str
    description: str = ""
    service: str | None = None
    environment: str = "prod"
    alarm_name: str | None = None
    triggered_at: str = Field(default_factory=_now_iso)
    # Free-form hints from the alert: log_group, ecs_cluster, db_instance...
    labels: dict[str, str] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)

    def brief(self) -> str:
        """A short, human-readable summary (used in prompts and Slack)."""
        return f"[{self.id}] {self.title} (service={self.service}, env={self.environment})"


class IncidentResult(BaseModel):
    """The final output of one agent run."""

    incident_id: str
    status: str = "completed"  # completed | failed
    severity: str | None = None  # P1..P4
    root_cause: str | None = None
    report_markdown: str = ""
    proposed_actions: list[dict[str, Any]] = Field(default_factory=list)
    similar_incidents: list[dict[str, Any]] = Field(default_factory=list)
    started_at: str = Field(default_factory=_now_iso)
    finished_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None
