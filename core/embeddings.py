# =====================================================================
# core/embeddings.py
# In simple words: turns text into numbers (vectors) so we can search by
# meaning. Used by the runbook search and the past-incident memory.
#
# Chosen from .env (EMBEDDINGS_MODEL):
#   local-hash            -> a small built-in method. Free, offline, no
#                            downloads. Matches on shared words/phrases.
#   openai:text-embedding-3-small (or any provider:model)
#                         -> a real embedding model through LangChain's
#                            init_embeddings(). Better at synonyms.
# =====================================================================
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from langchain_core.embeddings import Embeddings

from core.config import get_settings

_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was",
         "were", "it", "this", "that", "with", "as", "at", "by", "be", "from", "if", "then"}


class HashingEmbeddings(Embeddings):
    """Bag of words + word pairs, hashed into a fixed-size vector.
    Deterministic (md5, not Python's randomised hash) so saved vectors stay valid."""

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        words = [w for w in re.findall(r"[a-z0-9]+", text.lower())
                 if len(w) > 1 and w not in _STOP]
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]   # unigrams + bigrams

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok, n in Counter(self._tokens(text)).items():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            sign = 1.0 if (h >> 64) & 1 else -1.0
            vec[h % self.dim] += sign * (1.0 + math.log(n))
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def embeddings_spec() -> str:
    return get_settings().embeddings_model.strip() or "local-hash"


def get_embeddings() -> Embeddings:
    """Build the embeddings object named in .env."""
    spec = embeddings_spec()
    if spec.lower() in {"local-hash", "local", "hash"}:
        return HashingEmbeddings()
    from langchain.embeddings import init_embeddings  # needs the provider package

    return init_embeddings(spec)


def spec_slug() -> str:
    """Short id of the embeddings choice. Vectors from different embedding
    models are NOT compatible, so each model gets its own Chroma collection."""
    return hashlib.md5(embeddings_spec().encode()).hexdigest()[:8]
