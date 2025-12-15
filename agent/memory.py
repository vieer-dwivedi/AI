from collections import deque
from config.settings import ENABLE_MEMORY, MEMORY_WINDOW


class Memory:
    """
    Simple short-term memory for agents.
    Stores the last N messages.
    """

    def __init__(self, window_size: int = MEMORY_WINDOW):
        self.enabled = ENABLE_MEMORY
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def add(self, role: str, content: str):
        """
        Add a message to memory.
        """
        if not self.enabled:
            return

        self.buffer.append({
            "role": role,
            "content": content
        })

    def get(self):
        """
        Get memory messages as a list.
        """
        if not self.enabled:
            return []

        return list(self.buffer)

    def clear(self):
        """
        Clear memory.
        """
        self.buffer.clear()