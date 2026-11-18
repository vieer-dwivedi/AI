# =====================================================================
# core/config.py
# In simple words: this file reads the ".env" file and turns it into one
# "settings" object that every other file can import. No hard-coded
# values live anywhere else in the project - everything comes from here.
# =====================================================================
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The project's root folder (one level above "core/").
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env into the real environment. LangChain provider packages look
# for keys like OPENAI_API_KEY in os.environ, so this step is required.
# override=False means: a variable already set in your shell wins.
load_dotenv(BASE_DIR / ".env", override=False)


def _abs(path_str: str) -> Path:
    """Turn a relative path from .env (like ./data) into an absolute one."""
    p = Path(path_str)
    return p if p.is_absolute() else (BASE_DIR / p).resolve()


class Settings(BaseSettings):
    """Every setting, with a safe default. Names match the .env keys."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"), extra="ignore", case_sensitive=False
    )

    # ---- LLM -----------------------------------------------------------
    llm_model: str = "fake"
    llm_provider: str = ""
    llm_temperature: float | None = 0.0
    llm_max_tokens: int | None = None
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_extra_kwargs: str = "{}"
    triage_llm_model: str = ""
    diagnosis_llm_model: str = ""
    remediation_llm_model: str = ""

    # ---- Embeddings ----------------------------------------------------
    embeddings_model: str = "local-hash"

    # ---- Cloud ---------------------------------------------------------
    cloud_backend: str = "mock"
    aws_region: str = "us-east-1"
    mock_scenario: str = ""
    enable_azure: bool = False

    # ---- Safety / notifications ---------------------------------------
    auto_remediate: bool = False
    slack_webhook_url: str = ""
    webhook_secret: str = ""

    # ---- Storage -------------------------------------------------------
    chroma_dir: str = "./data/chroma"
    output_dir: str = "./out"
    log_level: str = "INFO"

    # ---- MCP -----------------------------------------------------------
    mcp_servers_json: str = ""
    mcp_tool_allowlist: str = ""

    # An empty value in .env (e.g. "LLM_MAX_TOKENS=") should mean "not set",
    # not crash while converting "" into a number.
    @field_validator("llm_temperature", "llm_max_tokens", mode="before")
    @classmethod
    def _empty_to_none(cls, v):
        return None if isinstance(v, str) and v.strip() == "" else v

    # ---- helpers -------------------------------------------------------
    def model_for(self, role: str | None = None) -> str:
        """Pick the model spec for a role (triage/diagnosis/remediation).
        If that role has no override, fall back to the main LLM_MODEL."""
        override = getattr(self, f"{role}_llm_model", "") if role else ""
        return (override or self.llm_model).strip()

    @property
    def extra_llm_kwargs(self) -> dict:
        """LLM_EXTRA_KWARGS parsed from JSON text into a dict."""
        raw = (self.llm_extra_kwargs or "").strip()
        return json.loads(raw) if raw else {}

    @property
    def chroma_path(self) -> Path:
        return _abs(self.chroma_dir)

    @property
    def output_path(self) -> Path:
        p = _abs(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def sample_data_path(self) -> Path:
        return BASE_DIR / "sample_data"

    @property
    def runbook_path(self) -> Path:
        return BASE_DIR / "runbooks"

    @property
    def mcp_allowed_tools(self) -> set[str]:
        """Names of external MCP tools the agent may use (comma separated)."""
        return {n.strip() for n in self.mcp_tool_allowlist.split(",") if n.strip()}

    @property
    def mcp_servers(self) -> dict:
        raw = (self.mcp_servers_json or "").strip()
        return json.loads(raw) if raw else {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Create the settings once and reuse them everywhere."""
    return Settings()


def reload_settings() -> Settings:
    """Forget the cached settings (used by tests after changing env vars)."""
    get_settings.cache_clear()
    return get_settings()
