import requests, logging
from app.config import LLM_BASE_URL

logger = logging.getLogger("llm-client")

class LLMClient:
    def generate(self, prompt: str) -> str:
        resp = requests.post(
            f"{LLM_BASE_URL}/generate",
            json={"prompt": prompt},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        for k in ("result","text","output","content"):
            if k in data:
                return data[k]
        raise RuntimeError(f"Bad LLM response: {data}")
