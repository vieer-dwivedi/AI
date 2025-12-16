import os
import sys
from azure.storage.queue import QueueClient


def main():
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    queue_name = os.getenv("AZURE_QUEUE_NAME", "test")

    if not connection_string:
        print("❌ AZURE_STORAGE_CONNECTION_STRING not set")
        sys.exit(1)

    if not queue_name:
        print("❌ AZURE_QUEUE_NAME not set")
        sys.exit(1)

    print("🔍 Testing Azure Queue connection...")
    print(f"Queue name: {queue_name}")

    try:
        queue_client = QueueClient.from_connection_string(
            conn_str=connection_string,
            queue_name=queue_name
        )

        # Send test message
        send_resp = queue_client.send_message("queue-connection-test")
        print("✅ Message sent successfully")

        # Peek message
        messages = queue_client.peek_messages(max_messages=1)
        for msg in messages:
            print("✅ Peeked message content:", msg.content)

        print("🎉 Azure Queue connection is WORKING")

    except Exception as e:
        print("❌ Azure Queue connection FAILED")
        print(e)
        sys.exit(2)


if __name__ == "__main__":
    main()
