"""Central settings, loaded from environment / .env (Step 1)."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5-5"
    anthropic_triage_model: str = "claude-haiku-5-5"
    max_steps: int = 15
    max_cost_usd_per_case: float = 0.75
    database_url: str = "sqlite:///data/erp.db"
    carrier_api_url: str = "http://localhost:8001"
    chroma_path: str = "./data/chroma"


settings = Settings()
