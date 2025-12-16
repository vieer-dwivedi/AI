import logging

# ---- SILENCE AZURE SDK LOGS ----
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

import app.logging_config  # keep this
from app.queue.consumer import poll_queue

if __name__ == "__main__":
    poll_queue()
