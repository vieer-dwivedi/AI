import operator
from typing_extensions import TypedDict, Annotated
from langchain.messages import AnyMessage


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int



from dataclasses import dataclass

@dataclass
class Context:
    user_id: str