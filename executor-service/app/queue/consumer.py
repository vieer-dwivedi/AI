import json
import time
from pprint import pprint
from azure.storage.queue import QueueClient
from app.config import (
    AZURE_STORAGE_CONNECTION_STRING,
    AZURE_QUEUE_NAME,
    POLL_INTERVAL,
)
from app.dispatcher import dispatch

# ---- Create client ----
queue_client = QueueClient.from_connection_string(
    conn_str=AZURE_STORAGE_CONNECTION_STRING,
    queue_name=AZURE_QUEUE_NAME,
)

# ---- Print ONCE when connected ----
def print_connected_once():
    try:
        queue_client.get_queue_properties()
        print(f"Connected to Azure Queue: {AZURE_QUEUE_NAME}")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Azure Queue: {e}")

# ---- Poller ----
def poll_queue():
    print_connected_once()  # 👈 ONE TIME ONLY

    while True:
        messages = queue_client.receive_messages(
            messages_per_page=1,
            visibility_timeout=60,
        )

        found = False
        for page in messages.by_page():
            for msg in page:
                found = True
                task = json.loads(msg.content)

                success = dispatch(task)

                if success:
                    queue_client.delete_message(msg.id, msg.pop_receipt)

                    print("MESSAGE RECEIVED AND PASSED:")
                    pprint(task)
                    print("-" * 50)

        if not found:
            time.sleep(POLL_INTERVAL)
