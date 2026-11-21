"""Entry point for the KP (Krishnamurti Paddhati) MCP server.

    python mcp/kp/main.py --http                     # separate server: http://127.0.0.1:8002/mcp (agents connect here)
    python mcp/kp/main.py --http --port 9002         # other port
    python mcp/kp/main.py --http --host 0.0.0.0      # reachable from other machines (see note below)
    python mcp/kp/main.py                            # stdio (for MCP clients that launch the server themselves)

Defaults can also come from the environment: KP_MCP_HOST, KP_MCP_PORT.
Note: with --host 0.0.0.0 the server has no authentication; only do that on a trusted network.

Run it as a script (not `python -m`), so the folder name `mcp` does not shadow the MCP SDK.
"""

import argparse
import os

from kp_server import mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="KP (Krishnamurti Paddhati) MCP server: Swiss Ephemeris + local KP tables")
    parser.add_argument("--http", action="store_true", help="serve over streamable-http (separate server) instead of stdio")
    parser.add_argument("--host", default=os.getenv("KP_MCP_HOST", "127.0.0.1"), help="bind address (http only)")
    parser.add_argument("--port", type=int, default=int(os.getenv("KP_MCP_PORT", "8002")), help="port (http only)")
    args = parser.parse_args()

    if args.http:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        print(f"KP MCP server: http://{args.host}:{args.port}{mcp.settings.streamable_http_path}", flush=True)
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
