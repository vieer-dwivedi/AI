# =====================================================================
# tools/common.py
# In simple words: two tiny helpers used by every tool.
#  - jdump(): turn Python data into compact JSON text (LLMs read text).
#  - safe_tool(): if a tool crashes, return {"error": "..."} instead of
#    blowing up the whole agent run. The agent can then try something else.
# =====================================================================
from __future__ import annotations

import functools
import json
import logging
from typing import Any, Callable

log = logging.getLogger(__name__)


def jdump(obj: Any, limit: int = 6000) -> str:
    """JSON-encode and cut to `limit` characters so one tool can't flood
    the model's context window."""
    text = json.dumps(obj, default=str, separators=(",", ":"))
    if len(text) > limit:
        # Keep the answer VALID JSON: wrap the cut-off text instead of
        # chopping the JSON in half.
        return json.dumps({"truncated": True, "original_chars": len(text),
                           "partial_json": text[:limit]}, separators=(",", ":"))
    return text


def fit_list(build: Callable[[list], Any], items: list, limit: int = 6000) -> Any:
    """Build a result from `items`, dropping the OLDEST items until it fits
    in `limit` characters. Used for log lines so output stays valid JSON."""
    items = list(items)
    while items and len(json.dumps(build(items), default=str, separators=(",", ":"))) > limit:
        items = items[max(1, len(items) // 5):] if len(items) > 1 else []
    return build(items)


def safe_tool(fn: Callable[..., str]) -> Callable[..., str]:
    """Decorator: catch any error and report it as JSON text."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - we really want everything
            log.exception("tool %s failed", fn.__name__)
            return jdump({"error": f"{type(e).__name__}: {e}", "tool": fn.__name__})

    return wrapper
