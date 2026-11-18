# ---------------------------------------------------------------------
# tests/test_config_llm.py
# Checks: (1) .env values reach the settings, (2) the LLM is chosen ONLY
# by env vars, and different providers can be swapped without code edits.
# No network calls are made - models are created but never invoked.
# ---------------------------------------------------------------------
import pytest

from core.config import reload_settings
from core.llm import get_chat_model, is_fake


def test_defaults_and_empty_values(monkeypatch):
    monkeypatch.setenv("LLM_MAX_TOKENS", "")        # empty -> None, no crash
    monkeypatch.setenv("LLM_TEMPERATURE", "")
    s = reload_settings()
    assert s.llm_max_tokens is None and s.llm_temperature is None
    assert s.cloud_backend == "mock" and s.auto_remediate is False


def test_role_override(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("DIAGNOSIS_LLM_MODEL", "anthropic:claude-sonnet-4-5")
    s = reload_settings()
    assert s.model_for("triage") == "openai:gpt-4o"            # falls back
    assert s.model_for("diagnosis") == "anthropic:claude-sonnet-4-5"


def test_fake_model_from_env():
    assert is_fake("fake") and is_fake("fake:scripted") and not is_fake("openai:gpt-4o")


def test_openai_from_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.3")
    reload_settings()
    m = get_chat_model()
    assert type(m).__name__ == "ChatOpenAI"
    assert m.model_name == "gpt-4o-mini" and m.temperature == 0.3


def test_anthropic_from_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "anthropic:claude-sonnet-4-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    reload_settings()
    assert type(get_chat_model()).__name__ == "ChatAnthropic"


def test_base_url_for_openai_compatible_servers(monkeypatch):
    # e.g. vLLM / LM Studio / any OpenAI-compatible endpoint
    monkeypatch.setenv("LLM_MODEL", "openai:my-local-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("LLM_API_KEY", "not-needed")
    reload_settings()
    m = get_chat_model()
    assert "localhost:8000" in str(m.openai_api_base)


def test_empty_model_is_an_error(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "  ")
    reload_settings()
    with pytest.raises(ValueError):
        get_chat_model()
