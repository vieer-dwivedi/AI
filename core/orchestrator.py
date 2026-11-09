from typing import Literal

from langchain.messages import SystemMessage, ToolMessage
from langgraph.graph import END

from core.state import MessagesState
from tools.registry import model_with_tools, tools_by_name
from core.prompts import SYSTEM_PROMPT


from langgraph.runtime import Runtime
from core.state import MessagesState, Context
import re

def llm_call(state: MessagesState, runtime: Runtime[Context]):
    """LLM decides whether to call a tool and uses long-term memory."""

    # Get user identity
    user_id = runtime.context.user_id

    # Define memory namespace
    namespace = (user_id, "memories")

    # Get latest user message
    user_message = state["messages"][-1].content

    # Save user information
    match = re.search(
        r"my name is ([a-zA-Z ]+)",
        user_message,
        re.IGNORECASE
    )

    if match:
        name = match.group(1).strip()

        runtime.store.put(
            namespace,
            "user_name",
            {"name": name}
        )

    # Retrieve saved memory
    memory = runtime.store.get(
        namespace,
        "user_name"
    )

    saved_name = memory.value["name"] if memory else "Unknown"

    # Add memory to system prompt
    system_message = SystemMessage(
        content=f"""
        {SYSTEM_PROMPT.content}

        User's saved information:
        Name: {saved_name}
        """
    )

    # Invoke LLM with tools and memory
    response = model_with_tools.invoke(
        [system_message] + state["messages"]
    )

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def tool_node(state: MessagesState):
    """Execute requested tools."""

    results = []

    for tool_call in state["messages"][-1].tool_calls:

        tool = tools_by_name[tool_call["name"]]

        observation = tool.invoke(tool_call["args"])

        results.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"]
            )
        )

    return {"messages": results}


def should_continue(
    state: MessagesState
) -> Literal["tool_node", "__end__"]:

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tool_node"

    return END