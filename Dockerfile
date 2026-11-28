FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.10 /uv /uvx /bin/

WORKDIR /srv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# LLM provider extra (anthropic | openai | google | ollama) and optional RAG deps
ARG LLM_EXTRA=anthropic
ARG WITH_RAG=false

# Install dependencies first so Docker caches this layer
COPY pyproject.toml uv.lock .python-version ./
RUN if [ "$WITH_RAG" = "true" ]; then \
      uv sync --frozen --no-dev --extra "$LLM_EXTRA" --extra rag; \
    else \
      uv sync --frozen --no-dev --extra "$LLM_EXTRA"; \
    fi

COPY . .
ENV PATH="/srv/.venv/bin:$PATH"
EXPOSE 8000 8001 8501
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
