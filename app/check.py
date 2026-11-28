"""Sanity check: config loads and the configured LLM responds."""
import sys

from app.config import settings
from app.llm import as_text, get_model


def main() -> int:
    if not settings.llm_model:
        print("❌ LLM_MODEL is not set in .env (format: provider:model-name)")
        return 1

    print(f"Model: {settings.llm_model}")
    model = get_model()
    resp = model.invoke("Reply with exactly: DispatchAgent ready")
    print("✅ LLM says:", as_text(resp.content))

    usage = resp.usage_metadata or {}
    print(f"Tokens: {usage.get('input_tokens', '?')} in / {usage.get('output_tokens', '?')} out")
    return 0


if __name__ == "__main__":
    sys.exit(main())