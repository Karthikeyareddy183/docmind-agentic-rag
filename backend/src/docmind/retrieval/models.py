"""Embedding and reranking models behind small interfaces.

Embedding model ids:
- ``local:<hf-id>``   sentence-transformers on CPU (e.g. ``local:BAAI/bge-m3``)
- ``litellm:<model>`` hosted API through LiteLLM (e.g. ``litellm:gemini/gemini-embedding-001``)

Local models are loaded once per process and run in a worker thread so they never block the
event loop.
"""

import asyncio
import logging
import threading
from functools import lru_cache
from typing import Any, Protocol

import litellm

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    model_id: str
    dimension: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class LocalEmbedder:
    def __init__(self, hf_id: str, batch_size: int = 16) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_id = f"local:{hf_id}"
        self.batch_size = batch_size
        self._model = SentenceTransformer(hf_id, device="cpu", trust_remote_code=False)
        self._lock = threading.Lock()
        prompts = getattr(self._model, "prompts", {}) or {}
        # Instruction-tuned models (Qwen3, Arctic, E5...) define query/document prompts
        self._query_prompt = "query" if prompts.get("query") else None
        self._doc_prompt = "document" if prompts.get("document") else None
        getter = (
            getattr(self._model, "get_embedding_dimension", None)
            or self._model.get_sentence_embedding_dimension
        )
        dim = getter()
        self.dimension = int(dim or 0)

    def _encode(self, texts: list[str], prompt_name: str | None) -> list[list[float]]:
        with self._lock:
            vectors = self._model.encode(
                texts,
                batch_size=self.batch_size,
                prompt_name=prompt_name,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return [v.tolist() for v in vectors]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, texts, self._doc_prompt)

    async def embed_query(self, text: str) -> list[float]:
        return (await asyncio.to_thread(self._encode, [text], self._query_prompt))[0]


class LiteLLMEmbedder:
    """Hosted embeddings through LiteLLM, batched (Gemini accepts up to 100 inputs per call)."""

    def __init__(self, model: str, dimension: int, batch_size: int = 100, retries: int = 5) -> None:
        self.model_id = f"litellm:{model}"
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size
        self.retries = retries

    async def _call(self, inputs: list[str], task: str) -> list[list[float]]:
        for attempt in range(self.retries + 1):
            try:
                response = await litellm.aembedding(
                    model=self.model,
                    input=inputs,
                    dimensions=self.dimension,
                    task_type=task,  # Gemini: RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY
                )
                return [_normalize(item["embedding"]) for item in response.data]
            except (
                litellm.exceptions.RateLimitError,
                litellm.exceptions.ServiceUnavailableError,
                litellm.exceptions.InternalServerError,
                litellm.exceptions.APIConnectionError,
                litellm.exceptions.Timeout,
            ) as exc:
                # A daily quota resets in hours: retrying only delays the caller's fallback
                if attempt == self.retries or "PerDay" in str(exc):
                    raise
                await asyncio.sleep(min(30.0, 2.0 * 2**attempt))
        raise AssertionError("unreachable")

    async def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(await self._call(texts[i : i + self.batch_size], task))
        return out

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, "RETRIEVAL_DOCUMENT") if texts else []

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text], "RETRIEVAL_QUERY"))[0]


def _normalize(vector: list[float]) -> list[float]:
    norm = sum(x * x for x in vector) ** 0.5 or 1.0
    return [x / norm for x in vector]


@lru_cache(maxsize=4)
def get_embedder(model_id: str, dimension: int, batch_size: int = 16) -> Embedder:
    kind, _, name = model_id.partition(":")
    if kind == "local":
        embedder: Embedder = LocalEmbedder(name, batch_size=batch_size)
    elif kind == "litellm":
        embedder = LiteLLMEmbedder(name, dimension, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown embedding model id {model_id!r} (use local:... or litellm:...)")
    if embedder.dimension != dimension:
        raise ValueError(
            f"{model_id} produces {embedder.dimension}-d vectors but the index expects {dimension}"
        )
    return embedder


class Reranker:
    """Cross-encoder relevance scores in 0..1 (sigmoid over the model's logit)."""

    def __init__(self, hf_id: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model_id = hf_id
        self._model = CrossEncoder(hf_id, device="cpu", max_length=512)
        self._lock = threading.Lock()

    def _score(self, query: str, passages: list[str]) -> list[float]:
        with self._lock:
            scores: Any = self._model.predict(
                [(query, p) for p in passages], batch_size=8, show_progress_bar=False
            )
        values = [float(s) for s in scores]
        if values and (min(values) < 0 or max(values) > 1):  # raw logits → probabilities
            import math

            values = [1 / (1 + math.exp(-v)) for v in values]
        return values

    async def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return await asyncio.to_thread(self._score, query, passages)


@lru_cache(maxsize=2)
def get_reranker(hf_id: str) -> Reranker:
    return Reranker(hf_id)
