from prometheus_client import Counter, Histogram

AGENT_REQUESTS = Counter(
    "agent_requests_total",
    "Total agent requests"
)

AGENT_ERRORS = Counter(
    "agent_errors_total",
    "Agent errors",
    ["stage"]
)

AGENT_STAGE_DURATION = Histogram(
    "agent_stage_duration_seconds",
    "Agent stage duration",
    ["stage"]
)

AGENT_TASKS_QUEUED = Counter(
    "agent_tasks_queued_total",
    "Total tasks queued"
)
