# =====================================================================
# mcp_server/server.py
# In simple words: exposes this project's abilities as an MCP server, so
# ANY MCP client (Claude Desktop, an IDE, another agent) can use them:
# read-only cloud queries, runbook search, past incidents, and a tool that
# runs the whole investigation.
#
# This is also the place to plug in YOUR existing MCP framework: add your
# own @mcp.tool() functions below, or keep your server separate and point
# the agent at it with MCP_SERVERS_JSON + MCP_TOOL_ALLOWLIST in .env.
#
# Run it:  python main.py mcp                       (stdio)
#          python main.py mcp --transport streamable-http
# Everything here is READ-ONLY except investigate_scenario, which only
# writes a report file and proposes fixes (nothing is executed).
# =====================================================================
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from core.config import get_settings  # noqa: E402
from memory.long_term import search_past_incidents  # noqa: E402
from tools.aws_tools import (describe_cloudwatch_alarms, get_cloudwatch_logs,  # noqa: E402
                             get_cloudwatch_metrics, list_recent_deployments)
from tools.log_analyzer import analyze_logs  # noqa: E402
from tools.runbook_search import search_runbooks  # noqa: E402

mcp = FastMCP("incident-response")


@mcp.tool()
def server_status() -> str:
    """Show how this incident-response server is configured (LLM, cloud backend, safety flags)."""
    s = get_settings()
    return (f"llm_model={s.llm_model} cloud_backend={s.cloud_backend} region={s.aws_region} "
            f"auto_remediate={s.auto_remediate}")


@mcp.tool()
def alarms(alarm_name: str = "", state: str = "ALARM") -> str:
    """List CloudWatch alarms (default: only the ones firing)."""
    return describe_cloudwatch_alarms.invoke({"alarm_name": alarm_name, "state": state})


@mcp.tool()
def logs_summary(log_group: str, filter_pattern: str = "", minutes: int = 60) -> str:
    """Analyse a log group: error patterns, first error time, detected failure signatures."""
    return analyze_logs.invoke({"log_group": log_group, "filter_pattern": filter_pattern, "minutes": minutes})


@mcp.tool()
def raw_logs(log_group: str, filter_pattern: str = "", minutes: int = 30, max_events: int = 25) -> str:
    """Fetch the latest raw log lines from a log group."""
    return get_cloudwatch_logs.invoke({"log_group": log_group, "filter_pattern": filter_pattern,
                                       "minutes": minutes, "max_events": max_events})


@mcp.tool()
def metric(namespace: str, metric_name: str, dimensions_json: str = "{}", minutes: int = 60) -> str:
    """Summarise a CloudWatch metric (min/max/avg/trend/peak time)."""
    return get_cloudwatch_metrics.invoke({"namespace": namespace, "metric_name": metric_name,
                                          "dimensions_json": dimensions_json, "minutes": minutes})


@mcp.tool()
def recent_deployments(service: str, limit: int = 5) -> str:
    """Recent deployments / config changes for a service, newest first."""
    return list_recent_deployments.invoke({"service": service, "limit": limit})


@mcp.tool()
def runbooks(query: str, k: int = 3) -> str:
    """Search the internal runbooks by describing the symptoms."""
    return search_runbooks.invoke({"query": query, "k": k})


@mcp.tool()
def past_incidents(query: str, k: int = 3) -> str:
    """Search long-term memory for similar past incidents and how they were fixed."""
    return search_past_incidents.invoke({"query": query, "k": k})


@mcp.tool()
async def investigate_scenario(scenario: str) -> str:
    """Run the full triage -> diagnosis -> remediation workflow on a sample scenario
    (mock backend only) and return the markdown report. Fixes are only PROPOSED."""
    from agents.orchestrator import run_incident_async
    from core.scenarios import incident_from_scenario

    if get_settings().cloud_backend.lower() != "mock":
        return "investigate_scenario only works with CLOUD_BACKEND=mock"
    result = await run_incident_async(incident_from_scenario(scenario))
    return result.report_markdown


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http"])
    args = p.parse_args(argv)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
