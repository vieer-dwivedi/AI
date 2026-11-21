"""Entry point for the DBA MCP server (Dasha-Bhukti-Antara timing).

    python mcp/DBA/main.py --http                # http://127.0.0.1:8003/mcp
    python mcp/DBA/main.py --http --port 9003
    python mcp/DBA/main.py                       # stdio

Defaults from the environment: DBA_MCP_HOST, DBA_MCP_PORT. Run as a script, not `python -m`.
"""
import argparse
import os

from dba_server import mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="DBA MCP server")
    parser.add_argument("--http", action="store_true", help="serve over streamable-http instead of stdio")
    parser.add_argument("--host", default=os.getenv("DBA_MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("DBA_MCP_PORT", "8003")))
    args = parser.parse_args()
    if args.http:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        print(f"DBA MCP server: http://{args.host}:{args.port}{mcp.settings.streamable_http_path}", flush=True)
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
