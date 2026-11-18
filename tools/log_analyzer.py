# =====================================================================
# tools/log_analyzer.py
# In simple words: a smart "log reader". Instead of giving the LLM
# thousands of log lines, this turns them into a short report:
#   - how many lines of each level (ERROR / WARN / INFO ...)
#   - the most common error messages (numbers/ids replaced, so
#     "took 31ms" and "took 87ms" count as the same message)
#   - when the errors first appeared and when they spiked
#   - known failure "signatures" (out of memory, DB connections, disk full)
# The core function analyze_events() is plain Python (easy to test).
# analyze_logs is the tool wrapper the agent calls.
# =====================================================================
from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone

from langchain_core.tools import tool

from tools.cloud_backend import get_backend
from tools.common import jdump, safe_tool

# ---------------------------------------------------------------------
# Known failure patterns. "cause" = points at a root cause,
# "symptom" = just the visible effect (5xx, timeouts) so we rank
# causes first.
# ---------------------------------------------------------------------
SIGNATURES = [
    ("out_of_memory", "cause",
     r"OutOfMemoryError|OOMKilled|exit code 137|killed due to memory",
     "Process or container ran out of memory (heap/limit too small or memory growth/leak)."),
    ("db_connection_exhaustion", "cause",
     r"remaining connection slots|Connection is not available|too many connections|HikariPool.*timed out|SQLTransientConnectionException",
     "Database connection limit reached; some client holds or opens too many connections."),
    ("disk_full", "cause",
     r"No space left on device|Errno 28|filesystem is full",
     "Disk is full; usually runaway logs or missing log rotation."),
    ("null_pointer_exception", "cause",
     r"NullPointerException",
     "Application bug (null dereference), typically introduced by a recent code change."),
    ("certificate_problem", "cause",
     r"certificate.*(expired|invalid|verify failed)",
     "TLS certificate expired or invalid."),
    ("dns_failure", "cause",
     r"Name or service not known|UnknownHostException|NXDOMAIN",
     "DNS resolution failure."),
    ("throttling", "cause",
     r"ThrottlingException|TooManyRequestsException|Rate exceeded",
     "Requests are being throttled by an AWS service or quota."),
    ("gc_pressure", "symptom",
     r"GC overhead|full GC",
     "Heavy garbage collection, often a precursor to OutOfMemory."),
    ("timeouts", "symptom", r"timed out|timeout|deadline exceeded",
     "Calls are timing out; look at the dependency being called."),
    ("http_5xx", "symptom", r'"status": ?5\d\d|\b5\d\d (?:Internal|Service|Bad)',
     "Server errors returned to clients."),
]
_SIG_RE = [(n, c, re.compile(p, re.I), h) for n, c, p, h in SIGNATURES]

_LEVEL_RE = re.compile(r"\b(FATAL|CRITICAL|ERROR|WARN(?:ING)?|INFO|DEBUG)\b")
_EXC_RE = re.compile(r"\b((?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:Exception|Error))\b")

# Replace changing details so identical messages group together.
_NORMALIZE = [
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?(?: UTC)?"), "<ts>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<ip>"),
    (re.compile(r"\b[0-9a-f]{12,}\b"), "<id>"),
    (re.compile(r"\(\d{4,6}\)|\[\d{3,6}\]"), "<pid>"),
    (re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)?"), "<n>"),
]


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(message: str) -> tuple[str, str]:
    """Return (level, text). Works for JSON logs and plain-text logs."""
    msg = message.strip()
    if msg.startswith("{"):
        try:
            d = json.loads(msg)
            text = str(d.get("message", msg))
            if d.get("exception") and d["exception"] not in text:
                text += f" [{d['exception']}]"
            return str(d.get("level", "UNKNOWN")).upper(), text
        except json.JSONDecodeError:
            pass
    m = _LEVEL_RE.search(msg)
    level = m.group(1).upper() if m else "UNKNOWN"
    return ("WARN" if level.startswith("WARN") else level), msg


def _normalize(text: str) -> str:
    for rx, rep in _NORMALIZE:
        text = rx.sub(rep, text)
    return text[:220]


# ---------------------------------------------------------------------
# The main function: events in -> structured report out.
# `events` use the CloudWatch shape: {"timestamp": ms, "message": str}.
# ---------------------------------------------------------------------
def analyze_events(events: list[dict]) -> dict:
    if not events:
        return {"total_events": 0, "note": "no events matched"}

    levels: Counter = Counter()
    err_templates: dict[str, dict] = {}
    warn_templates: Counter = Counter()
    sig_hits: dict[str, dict] = {}
    exceptions: Counter = Counter()
    per_minute: dict[int, int] = defaultdict(int)
    first_error_ms: int | None = None

    for e in sorted(events, key=lambda x: x["timestamp"]):
        ts, raw = e["timestamp"], e["message"]
        level, text = _parse(raw)

        # Which known signatures does this line match?
        matched = [(n, c, h) for n, c, rx, h in _SIG_RE if rx.search(raw)]
        is_error = level in {"ERROR", "FATAL", "CRITICAL"} or any(
            c == "cause" for _, c, _ in matched)
        levels["ERROR" if is_error and level not in {"FATAL", "CRITICAL"} else level] += 1

        for n, c, h in matched:
            hit = sig_hits.setdefault(n, {"name": n, "kind": c, "hint": h, "count": 0,
                                          "first_seen_ms": ts, "example": raw[:200]})
            hit["count"] += 1
        for ex in _EXC_RE.findall(raw):
            exceptions[ex.split(".")[-1]] += 1

        if is_error:
            first_error_ms = first_error_ms or ts
            per_minute[ts // 60_000] += 1
            t = err_templates.setdefault(_normalize(text), {
                "count": 0, "first_seen_ms": ts, "last_seen_ms": ts, "example": text[:200]})
            t["count"] += 1
            t["last_seen_ms"] = ts
        elif level == "WARN":
            warn_templates[_normalize(text)] += 1

    # ---- error timeline & spike start --------------------------------
    # Count errors per minute over the WHOLE window (including the quiet
    # time before trouble), so "normal" is measured from the quiet part.
    spike_start = None
    timeline = []
    if per_minute:
        lo = min(e["timestamp"] for e in events) // 60_000
        hi = max(e["timestamp"] for e in events) // 60_000
        counts = [per_minute.get(m, 0) for m in range(lo, hi + 1)]
        base = statistics.median(counts[: max(1, len(counts) // 3)])
        threshold = max(3, base * 3 + 2)
        for i in range(len(counts)):
            window = counts[i:i + 3]
            if len(window) == 3 and all(c >= threshold for c in window):
                spike_start = _iso((lo + i) * 60_000)
                break
        # Timeline: start just before the first error, ~10 sample points.
        first_i = max(0, min(per_minute) - lo - 2)
        view = counts[first_i:]
        step = max(1, len(view) // 10)
        timeline = [{"minute": _iso((lo + first_i + i) * 60_000), "errors": view[i]}
                    for i in range(0, len(view), step)]

    signatures = sorted(sig_hits.values(), key=lambda s: (s["kind"] != "cause", -s["count"]))
    for s in signatures:
        s["first_seen"] = _iso(s.pop("first_seen_ms"))

    top_errors = sorted(err_templates.items(), key=lambda kv: -kv[1]["count"])[:5]
    return {
        "total_events": len(events),
        "window": {"from": _iso(events[0]["timestamp"]), "to": _iso(events[-1]["timestamp"])},
        "level_counts": dict(levels),
        "first_error_time": _iso(first_error_ms) if first_error_ms else None,
        "error_spike_start": spike_start,
        "top_error_messages": [{
            "count": v["count"], "pattern": k, "example": v["example"],
            "first_seen": _iso(v["first_seen_ms"]), "last_seen": _iso(v["last_seen_ms"]),
        } for k, v in top_errors],
        "top_warnings": [{"count": c, "pattern": p} for p, c in warn_templates.most_common(3)],
        "top_exceptions": [{"name": n, "count": c} for n, c in exceptions.most_common(5)],
        "signatures": signatures,
        "error_timeline": timeline,
    }


# ---------------------------------------------------------------------
# The tool the agent calls: fetch logs, then analyse them.
# ---------------------------------------------------------------------
@tool
@safe_tool
def analyze_logs(log_group: str, filter_pattern: str = "", minutes: int = 60) -> str:
    """Analyse a CloudWatch log group over the last N minutes and return a
    compact report: level counts, top error patterns, first error time, error
    spike start, top exceptions and detected failure signatures (out of memory,
    DB connection exhaustion, disk full, NullPointerException, ...).
    Use this FIRST on a service's logs before reading raw lines."""
    r = get_backend().filter_log_events(log_group, filter_pattern, minutes, limit=5000)
    report = analyze_events(r["events"])
    report["log_group"] = log_group
    report["window_minutes"] = minutes
    return jdump(report, limit=7000)


LOG_TOOLS = [analyze_logs]
