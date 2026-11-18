# =====================================================================
# tools/mcp_loader.py
# In simple words: lets you plug OTHER MCP servers (yours or third-party)
# into the agent. Two safety locks:
#   1. Servers come from MCP_SERVERS_JSON in .env (nothing is hard-coded).
#   2. Only tools whose names are in MCP_TOOL_ALLOWLIST are given to the
#      agent. Everything else the server offers is ignored.
# No servers configured / no allow-list = this returns an empty list.
# =====================================================================
from __future__ import annotations

import logging

from core.config import get_settings

log = logging.getLogger(__name__)


async def load_allowed_mcp_tools() -> list:
    """Connect to the configured MCP servers and return only allow-listed tools."""
    s = get_settings()
    if not s.mcp_servers or not s.mcp_allowed_tools:
        return []
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        tools = await MultiServerMCPClient(s.mcp_servers).get_tools()
    except Exception as e:  # noqa: BLE001 - a broken plugin must not kill the incident run
        log.warning("Could not load MCP tools (%s: %s) - continuing without them", type(e).__name__, e)
        return []
    allowed = s.mcp_allowed_tools
    skipped = sorted(t.name for t in tools if t.name not in allowed)
    if skipped:
        log.info("MCP tools NOT attached (not in MCP_TOOL_ALLOWLIST): %s", skipped)
    return [t for t in tools if t.name in allowed]
