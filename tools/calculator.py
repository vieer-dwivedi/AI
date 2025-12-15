import math
import operator

# Allowed operators
_ALLOWED_OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "**": operator.pow,
}

# Allowed math functions
_ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
}


def calculator(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    Supported:
      - +, -, *, /, //, %, **
      - basic math functions (sqrt, log, sin, etc.)
    """

    try:
        # Replace unsafe names
        safe_globals = {"__builtins__": {}}
        safe_locals = {}

        # Add allowed operators & functions
        safe_locals.update(_ALLOWED_FUNCTIONS)

        # Evaluate expression safely
        result = eval(expression, safe_globals, safe_locals)
        return str(result)

    except Exception as e:
        return f"Calculation error: {e}"
