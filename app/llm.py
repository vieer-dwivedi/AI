"""Provider-agnostic model factory. The rest of the code never names a vendor."""
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from app.config import settings

load_dotenv()


def get_model(kind: str = "main", **kwargs) -> BaseChatModel:
    """Return a chat model. kind: 'main' (reasoning) or 'triage' (cheap/fast)."""
    spec = settings.llm_triage_model if kind == "triage" else settings.llm_model
    spec = spec or settings.llm_model
    if not spec:
        raise RuntimeError("Set LLM_MODEL in .env, e.g. LLM_MODEL=provider:model-name")
    if spec.startswith("ollama:"):
        kwargs.setdefault("num_ctx", settings.llm_num_ctx)
    return init_chat_model(spec, **kwargs)


def as_text(content) -> str:
    """Normalize message content, which is a str or a list of blocks depending on provider."""
    if isinstance(content, str):
        return content
    return "".join(
        b.get("text", "") if isinstance(b, dict) else str(b) for b in content
    )