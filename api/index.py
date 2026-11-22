"""Vercel entrypoint: exposes the ASGI app (Resume Studio MCP)."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server import app as _app  # noqa: E402


async def _missing(scope, receive, send):
    if scope["type"] == "http":
        body = b"RESUME_SECRET env var (16+ chars) is not set. Add it in Vercel -> Settings -> Environment Variables and redeploy."
        await send({"type": "http.response.start", "status": 500, "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": body})


app = _app or _missing
