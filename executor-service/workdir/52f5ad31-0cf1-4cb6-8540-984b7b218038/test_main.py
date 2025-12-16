import pytest
from main import greet

def test_greet_returns_hello_world():
    """Tests that the greet function returns the correct string."""
    assert greet() == "Hello World"