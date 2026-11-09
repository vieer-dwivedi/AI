from langchain.messages import HumanMessage
from core.graph import agent
from core.state import Context

config = {
    "configurable": {
        "thread_id": "conversation-1"
    }
}

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    result = agent.invoke(
        {
            "messages": [HumanMessage(content=user_input)]
        },
        config=config,
        context=Context(user_id="user-123")
    )

    for message in result["messages"]:
        message.pretty_print()