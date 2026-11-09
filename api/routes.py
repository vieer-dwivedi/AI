from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from uuid import uuid4

from langchain.messages import HumanMessage
from core.graph import agent
from core.state import Context

router = APIRouter()


class ChatRequest(BaseModel):
    model: Optional[str] = "langgraph-agent"
    messages: list[dict]
    user: Optional[str] = "default-user"


def extract_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block["text"] if isinstance(block, dict) else str(block)
            for block in content
            if (isinstance(block, dict) and block.get("type") == "text") or isinstance(block, str)
        )
    return str(content)


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):

    try:
        # Get latest user message
        user_message = next(
            msg["content"]
            for msg in reversed(request.messages)
            if msg["role"] == "user"
        )

        # Conversation ID
        thread_id = request.user or str(uuid4())

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        # Invoke LangGraph agent
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content=user_message)
                ]
            },
            config=config,
            context=Context(user_id=thread_id)
        )

        # Extract tool calls from intermediate messages
        tool_calls_log = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_log.append(f"🔧 `{tc['name']}` → `{tc['args']}`")

        # Extract final AI response
        answer = extract_content(result["messages"][-1].content)

        # Prepend tool calls if any
        if tool_calls_log:
            tools_summary = "\n".join(tool_calls_log)
            answer = f"{tools_summary}\n\n---\n\n{answer}"

        return {
            "id": f"chatcmpl-{uuid4().hex}",
            "object": "chat.completion",
            "created": 0,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer
                    },
                    "finish_reason": "stop"
                }
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "langgraph-agent",
                "object": "model",
                "created": 0,
                "owned_by": "local",
                "name": "LangGraph Agent"
            }
        ]
    }