# ---------------------------------------------------------------------
# tests/conftest.py
# In simple words: before EVERY test, reset the settings so tests never
# touch your real .env, real Slack, real AWS or your real Chroma folder.
# ---------------------------------------------------------------------
import pytest

from core.config import reload_settings


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_MODEL", "fake")
    monkeypatch.setenv("CLOUD_BACKEND", "mock")
    monkeypatch.setenv("MOCK_SCENARIO", "")
    monkeypatch.setenv("AUTO_REMEDIATE", "false")
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "")
    monkeypatch.setenv("WEBHOOK_SECRET", "")
    monkeypatch.setenv("EMBEDDINGS_MODEL", "local-hash")
    monkeypatch.setenv("MCP_SERVERS_JSON", "")
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    reload_settings()
    yield
    reload_settings()
