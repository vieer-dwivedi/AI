# =====================================================================
# tools/slack_notifier.py
# In simple words: lets the agent tell humans what is happening.
#  - If SLACK_WEBHOOK_URL is set in .env -> posts to that Slack channel.
#  - If it is empty -> writes the message to out/slack_outbox.jsonl
#    instead (a safe "dry run", nothing leaves your machine).
# =====================================================================
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import httpx
from langchain_core.tools import tool

from core.config import get_settings
from tools.common import jdump, safe_tool

log = logging.getLogger(__name__)

_ICON = {"critical": ":rotating_light:", "high": ":red_circle:", "warning": ":warning:",
         "info": ":information_source:", "resolved": ":white_check_mark:"}


def send_slack(message: str, severity: str = "info", incident_id: str = "") -> dict:
    """Send (or dry-run) one Slack message. Plain function so the pipeline
    can call it directly as well as the LLM tool below."""
    s = get_settings()
    icon = _ICON.get(severity.lower(), ":speech_balloon:")
    prefix = f"[{incident_id}] " if incident_id else ""
    text = f"{icon} {prefix}{message}"

    if s.slack_webhook_url:
        r = httpx.post(s.slack_webhook_url, json={"text": text}, timeout=10)
        r.raise_for_status()
        return {"delivered": "slack", "status_code": r.status_code}

    # No webhook configured -> save to a local file.
    outbox = s.output_path / "slack_outbox.jsonl"
    record = {"time": datetime.now(timezone.utc).isoformat(), "incident_id": incident_id,
              "severity": severity, "text": text}
    with outbox.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return {"delivered": "outbox", "file": str(outbox)}


@tool
@safe_tool
def post_slack_update(message: str, severity: str = "info", incident_id: str = "") -> str:
    """Post a short status update to the incident Slack channel.
    severity: critical | high | warning | info | resolved.
    Keep messages brief (what happened, impact, what is being done)."""
    return jdump(send_slack(message, severity, incident_id))


NOTIFY_TOOLS = [post_slack_update]
