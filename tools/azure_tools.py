# =====================================================================
# tools/azure_tools.py
# In simple words: the Azure side (Azure Monitor alerts + AKS pods).
# NOTE: this is MOCK-ONLY for now. It reads sample_data/azure.json so you
# can see how a second cloud plugs in. To make it real, replace the three
# function bodies with calls to azure-monitor-query / the Kubernetes API.
# It is only loaded when ENABLE_AZURE=true in .env.
# =====================================================================
from __future__ import annotations

import json

from langchain_core.tools import tool

from core.config import get_settings
from tools.common import jdump, safe_tool


def _data() -> dict:
    return json.loads((get_settings().sample_data_path / "azure.json").read_text())


@tool
@safe_tool
def azure_list_alerts(state: str = "Fired") -> str:
    """List Azure Monitor alerts (state: Fired / Resolved / empty for all)."""
    alerts = [a for a in _data()["alerts"] if not state or a["state"].lower() == state.lower()]
    return jdump({"count": len(alerts), "alerts": alerts})


@tool
@safe_tool
def azure_aks_pods(namespace: str = "default") -> str:
    """List AKS pods in a namespace with status, restart count and last state."""
    pods = _data()["aks_pods"].get(namespace, [])
    return jdump({"namespace": namespace, "pods": pods})


@tool
@safe_tool
def azure_query_logs(contains: str = "") -> str:
    """Search recent Azure container log lines containing a text."""
    lines = [l for l in _data()["logs"] if contains.lower() in l.lower()]
    return jdump({"matched": len(lines), "lines": lines})


AZURE_TOOLS = [azure_list_alerts, azure_aks_pods, azure_query_logs]
