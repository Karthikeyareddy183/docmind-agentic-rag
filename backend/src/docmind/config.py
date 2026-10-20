"""Application settings, loaded from environment variables (prefix ``DOCMIND_``) and ``.env``.

Provider API keys (``GEMINI_API_KEY``, ``OPENAI_API_KEY``, ...) keep their standard names and are
read by LiteLLM directly; :func:`load_provider_keys` copies them from ``.env`` into the process
environment so that works without exporting them in the shell.
"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(".env")

PROVIDER_KEY_NAMES = (
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "GROQ_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "HF_HOME",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCMIND_",
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = Field(default="development", description="development | test | production")

    # ─── Database ────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://docmind:docmind@localhost:5432/docmind"
    db_pool_size: int = 5
    db_echo: bool = False
    # Managed Postgres (e.g. Supabase) requires TLS
    db_ssl: bool = False
    # Set when connecting through a transaction-mode pooler (Supabase port 6543 / PgBouncer),
    # which cannot hold asyncpg's server-side prepared statements across transactions
    db_transaction_pooler: bool = False
    # Schema holding the app's tables. Tests use a separate schema in the same database.
    db_schema: str = "public"

    # ─── Auth ────────────────────────────────────────────────────
    jwt_secret: SecretStr = SecretStr("change-me-in-production-please-32b")
    jwt_algorithm: str = "HS256"
    access_token_ttl_minutes: int = 60

    # ─── Uploads ─────────────────────────────────────────────────
    storage_dir: Path = Path("./data/uploads")
    max_upload_mb: int = 50
    max_pages: int = 200

    # ─── Ingestion ───────────────────────────────────────────────
    # Run the job worker inside the API process (handy for local dev / single-box demo)
    run_worker_in_api: bool = True
    worker_poll_seconds: float = 2.0
    job_lease_seconds: int = 900
    chunk_max_tokens: int = 450
    # PDFs longer than this skip the (slow, ~3 s/page on CPU) layout model and read the PDF's
    # own text layer instead (~0.01 s/page); pages without text are OCR'd. Shorter documents
    # keep full layout analysis for tables and figures.
    fast_parse_min_pages: int = 30
    # A page with fewer extracted characters than this is treated as a scan and OCR'd
    ocr_min_chars: int = 50
    # Chunks are written in batches so progress moves and memory stays flat
    index_batch_size: int = 200
    # Background embedding (stage 2): chunks per batch, and how long to keep waiting out
    # provider rate limits before giving up (the document stays keyword-searchable meanwhile)
    embed_batch_chunks: int = 48
    embed_max_wait_seconds: int = 7200
    # The embed job heartbeats after every batch, so a short lease means a crashed worker's
    # job is picked up again within ~2 minutes instead of the full ingestion lease
    embed_lease_seconds: int = 120
    describe_figures: bool = True
    summarize_tables: bool = True

    # ─── Embeddings / reranking ──────────────────────────────────
    # "local:<hf-model-id>" (CPU, sentence-transformers) or "litellm:<model>" (hosted API)
    # Hosted Gemini embeddings: best retrieval in eval/reports/retrieval_bakeoff.md and ~80x faster
    # than bge-m3 on CPU for long passages (0.04 s vs 3 s per chunk). For fully offline use set
    # local:BAAI/bge-m3 (same 1024 dimensions; re-index documents after switching).
    embedding_model: str = "litellm:gemini/gemini-embedding-001"
    embedding_dimension: int = 1024
    # Inputs per embedding call (local batch size, or per hosted API request; keep hosted
    # requests small so free-tier tokens-per-minute limits are not exceeded by one request)
    embedding_batch_size: int = 16
    reranker_model: str = "BAAI/bge-reranker-base"
    # Off by default: on this corpus + CPU the reranker added ~2.7 s/query without improving
    # recall over weighted hybrid retrieval (eval/reports/fusion_sweep.md). Enable on GPU.
    rerank_enabled: bool = False

    # ─── Retrieval ───────────────────────────────────────────────
    # "hybrid" (dense + keyword, RRF) or "dense" (vector only; the naive-RAG baseline)
    retrieval_mode: str = "hybrid"
    dense_k: int = 40
    keyword_k: int = 40
    rrf_k: int = 60
    # Weighted RRF: how much each ranking contributes (keyword < dense when distractors share
    # vocabulary with the gold documents, e.g. same-template reports from other years/companies)
    dense_weight: float = 1.0
    keyword_weight: float = 0.25  # tuned in eval/fusion_sweep.py
    rerank_candidates: int = 24
    context_chunks: int = 6
    # Passages scoring below this (cosine, or reranker score when enabled) are not shown to
    # the LLM; if nothing passes, answer "not found" without an LLM call
    min_relevance: float = 0.35

    # ─── LLMs (LiteLLM model strings) ────────────────────────────
    llm_model: str = "gemini/gemini-3.5-flash"
    llm_fast_model: str = "gemini/gemini-3.5-flash-lite"
    vision_model: str = "gemini/gemini-3.5-flash"
    llm_timeout_seconds: float = 60.0
    # Cap on output tokens per call: bounds cost and latency (providers may default very high)
    llm_max_output_tokens: int = 2048
    llm_max_retries: int = 4
    agent_max_tool_rounds: int = 3
    # "agent": LangGraph planner + tools + fact-check. "single": one search, one answer (baseline)
    agent_mode: str = "agent"

    # ─── HTTP ────────────────────────────────────────────────────
    cors_origins: list[str] = ["http://localhost:3000"]
    log_level: str = "INFO"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


def load_provider_keys(env_file: Path = ENV_FILE) -> None:
    """Expose provider keys from ``.env`` to LiteLLM/HF without overriding the real environment."""
    values = dotenv_values(env_file) if env_file.exists() else {}
    for name in PROVIDER_KEY_NAMES:
        value = values.get(name)
        if value and name not in os.environ:
            os.environ[name] = value


@lru_cache
def get_settings() -> Settings:
    load_provider_keys()
    return Settings()
