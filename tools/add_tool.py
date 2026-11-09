
from langchain.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b