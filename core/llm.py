# =====================================================================
# core/llm.py
# In simple words: ONE place that builds the chat model.
# It reads LLM_MODEL (and friends) from .env and calls LangChain's
# init_chat_model(), which can create a model for almost any provider
# (Anthropic, OpenAI, Google, Ollama, Groq, Bedrock...).
# Change the .env value -> a different LLM. No code change needed.
# =====================================================================
from __future__ import annotations

import logging

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from core.config import get_settings

log = logging.getLogger(__name__)


def is_fake(spec: str) -> bool:
    """'fake' or 'fake:anything' means: use the offline test model."""
    return spec.strip().lower().split(":")[0] == "fake"


def get_chat_model(role: str | None = None) -> BaseChatModel:
    """Build the chat model for a role (triage / diagnosis / remediation).

    role=None uses the main LLM_MODEL. A role with its own override in .env
    (e.g. TRIAGE_LLM_MODEL) uses that instead - handy to run a cheap model
    for triage and a stronger one for diagnosis.
    """
    s = get_settings()
    spec = s.model_for(role)

    if not spec:
        raise ValueError("LLM_MODEL is empty. Set it in .env, e.g. openai:gpt-4o")

    # Offline scripted model, only for tests / demos without an API key.
    if is_fake(spec):
        from core.fake_llm import FakeIncidentModel

        log.warning("Using the OFFLINE FAKE model (LLM_MODEL=%s). Not a real LLM!", spec)
        return FakeIncidentModel()

    # Only pass options that were actually set, so provider defaults apply.
    kwargs: dict = dict(s.extra_llm_kwargs)
    if s.llm_temperature is not None:
        kwargs["temperature"] = s.llm_temperature
    if s.llm_max_tokens:
        kwargs["max_tokens"] = s.llm_max_tokens
    if s.llm_base_url:
        kwargs["base_url"] = s.llm_base_url
    if s.llm_api_key:
        kwargs["api_key"] = s.llm_api_key
    if s.llm_provider:
        kwargs["model_provider"] = s.llm_provider

    try:
        return init_chat_model(spec, **kwargs)
    except ImportError as e:  # provider package not installed
        provider = spec.split(":")[0]
        raise ImportError(
            f"Provider package for '{provider}' is missing: {e}. "
            f"Install the matching langchain-<provider> package (see requirements.txt)."
        ) from e
