"""Per-case JSONL traces and OpenTelemetry spans (Step 12).

- Tracer.event(...) appends one JSON line to traces/<case_id>.jsonl (audit trail).
- span(...) wraps work in an OpenTelemetry span. Spans do nothing until an exporter is configured;
  set OTEL_CONSOLE=1 to print them, or configure an OTLP exporter in setup_otel().
"""
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from opentelemetry import trace

TRACE_DIR = Path(__file__).resolve().parent.parent.parent / "traces"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def setup_otel() -> None:
    if os.getenv("OTEL_CONSOLE") != "1":
        return
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)


setup_otel()
_otel = trace.get_tracer("dispatch-agent")


@contextmanager
def span(name: str, **attrs):
    """Usage: with span("tool.call", tool="get_order"): ..."""
    with _otel.start_as_current_span(name) as s:
        for k, v in attrs.items():
            s.set_attribute(k, v if isinstance(v, (str, int, float, bool)) else str(v))
        yield s


def _path(case_id: str) -> Path:
    if not _SAFE_ID.match(case_id):                      # case ids come from URLs: block path tricks
        raise ValueError("invalid case id")
    return TRACE_DIR / f"{case_id}.jsonl"


class Tracer:
    def __init__(self, case_id: str, reset: bool = False):
        TRACE_DIR.mkdir(exist_ok=True)
        self.path = _path(case_id)
        if reset:
            self.path.unlink(missing_ok=True)

    def event(self, type_: str, **data) -> None:
        line = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "type": type_, **data}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, default=str) + "\n")


def read_trace(case_id: str) -> list[dict]:
    try:
        path = _path(case_id)
    except ValueError:
        return []
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
