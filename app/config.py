"""Central settings, loaded from environment / .env."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    llm_model: str = ""
    llm_triage_model: str = ""
    embedding_model: str = "st:all-MiniLM-L6-v2"
    llm_num_ctx: int = 8192  # Ollama context window (its default of 4096 cuts off our prompt)
    max_steps: int = 15
    max_cost_usd_per_case: float = 0.75
    max_tokens_per_case: int = 60000
    price_in_usd_per_mtok: float = 3.0    # set to your model's input price
    price_out_usd_per_mtok: float = 15.0  # set to your model's output price
    database_url: str = "sqlite:///data/erp.db"
    carrier_api_url: str = "http://localhost:8001"
    chroma_path: str = "./data/chroma"


settings = Settings()