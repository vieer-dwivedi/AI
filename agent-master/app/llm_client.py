import os
import logging
import requests

logger = logging.getLogger("llm-client")

LLM_BASE_URL = os.getenv("LLM_BASE_URL")

if not LLM_BASE_URL:
    raise RuntimeError("LLM_BASE_URL environment variable is not set")


class LLMClient:
    def __init__(self):
        self.base_url = LLM_BASE_URL.rstrip("/")

    def generate(self, prompt: str) -> str:
        logger.info("LLM request started")

        resp = requests.post(
            f"{self.base_url}/generate",
            json={"prompt": prompt},
            timeout=60
        )

        resp.raise_for_status()
        data = resp.json()

        logger.debug("Raw LLM response", extra={"data": data})

        # ---- Robust extraction ----
        if "result" in data:
            text = data["result"]
        elif "text" in data:
            text = data["text"]
        elif "output" in data:
            text = data["output"]
        elif "content" in data:
            text = data["content"]
        elif "choices" in data:  # OpenAI-style
            text = data["choices"][0]["message"]["content"]
        else:
            raise ValueError(
                f"Unsupported LLM response format: {data}"
            )

        if not isinstance(text, str) or not text.strip():
            raise ValueError("LLM returned empty response")

        logger.info("LLM request successful")
        return text.strip()
