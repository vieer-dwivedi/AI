import os
import json
import logging
from azure.storage.queue import QueueClient
from app.metrics.agent_metrics import AGENT_ERRORS

logger = logging.getLogger("queue-client")

# -------- Env validation --------
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
QUEUE_NAME = os.getenv("AZURE_QUEUE_NAME")

if not CONNECTION_STRING:
    raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING not set")

if not QUEUE_NAME:
    raise RuntimeError("AZURE_QUEUE_NAME not set")


# -------- Client init (once) --------
queue_client = QueueClient.from_connection_string(
    conn_str=CONNECTION_STRING,
    queue_name=QUEUE_NAME
)


def push_job(job: dict):
    """
    Push a single executor-ready task to Azure Queue.
    """
    try:
        payload = json.dumps(job)
        queue_client.send_message(payload)

        logger.info(
            "Queue push successful",
            extra={
                "queue": QUEUE_NAME,
                "workflow_id": job.get("workflow_id"),
                "task_index": job.get("task_index"),
            }
        )

    except Exception:
        AGENT_ERRORS.labels(stage="queue").inc()
        logger.exception("Queue push failed")
        raise
