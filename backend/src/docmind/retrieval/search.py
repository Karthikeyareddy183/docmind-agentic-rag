"""Hybrid retrieval over the chunks table.

dense (pgvector HNSW, cosine) + keyword (Postgres full-text, weighted tsvector)
  → Reciprocal Rank Fusion → optional cross-encoder rerank → top chunks.

Every query is filtered by ``owner_id`` (and optionally document ids) in SQL, so one user's
search can never touch another user's chunks.
"""

import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from docmind.config import Settings
from docmind.db.models import Chunk, Document, DocumentStatus
from docmind.retrieval.models import Embedder, Reranker

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    document_title: str
    chunk_type: str
    content: str
    context_header: str
    section_path: str
    page_start: int
    page_end: int
    bboxes: list[dict[str, Any]]
    data: dict[str, Any]
    dense_rank: int | None = None
    keyword_rank: int | None = None
    rrf_score: float = 0.0
    similarity: float = 0.0  # cosine similarity between query and chunk embeddings
    has_vector: bool = True  # False while two-stage indexing has not embedded this chunk yet
    rerank_score: float | None = None

    @property
    def score(self) -> float:
        """Best available relevance estimate in 0..1.

        Cross-encoder score if reranking ran; else cosine similarity; for chunks that are not
        embedded yet, a keyword-rank-based estimate (rank 1 -> 0.5, decaying) so strong
        keyword matches still reach the answer while background embedding catches up.
        """
        if self.rerank_score is not None:
            return self.rerank_score
        if not self.has_vector and self.keyword_rank is not None:
            return 0.5 / (1 + 0.1 * (self.keyword_rank - 1))
        return max(0.0, min(1.0, self.similarity))


@dataclass
class SearchResult:
    chunks: list[RetrievedChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)
    candidates: int = 0

    @property
    def top_score(self) -> float:
        return self.chunks[0].score if self.chunks else 0.0


def _scoped(
    stmt: Select[Any], owner_id: uuid.UUID, document_ids: Sequence[uuid.UUID] | None
) -> Select[Any]:
    stmt = stmt.where(Chunk.owner_id == owner_id)
    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))
    return stmt


def rrf_fuse(
    rankings: Sequence[Sequence[uuid.UUID]], k: int, weights: Sequence[float] | None = None
) -> dict[uuid.UUID, float]:
    """(Weighted) Reciprocal Rank Fusion: sum of weight / (k + rank) over the rankings."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[uuid.UUID, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank + 1)
    return scores


class HybridSearcher:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        embedder: Embedder,
        reranker: Reranker | None,
    ) -> None:
        self.session = session
        self.settings = settings
        self.embedder = embedder
        self.reranker = reranker

    async def _dense(
        self, owner_id: uuid.UUID, vector: list[float], document_ids: Sequence[uuid.UUID] | None
    ) -> list[uuid.UUID]:
        distance = Chunk.embedding.cosine_distance(vector)
        # Vectors from different embedding models live in different spaces: never mix them
        stmt = _scoped(select(Chunk.id), owner_id, document_ids).where(
            Chunk.embedding.is_not(None), Chunk.embedding_model == self.embedder.model_id
        )
        rows = await self.session.scalars(stmt.order_by(distance).limit(self.settings.dense_k))
        return list(rows)

    async def _keyword(
        self, owner_id: uuid.UUID, query: str, document_ids: Sequence[uuid.UUID] | None
    ) -> list[uuid.UUID]:
        # websearch_to_tsquery ANDs terms; for natural-language questions OR is far more useful
        tsquery = func.to_tsquery(
            "english",
            func.replace(func.querytree(func.websearch_to_tsquery("english", query)), "&", "|"),
        )
        rank = func.ts_rank_cd(Chunk.tsv, tsquery, 32)  # 32: normalise rank to 0..1
        stmt = _scoped(select(Chunk.id), owner_id, document_ids).where(Chunk.tsv.op("@@")(tsquery))
        rows = await self.session.scalars(stmt.order_by(rank.desc()).limit(self.settings.keyword_k))
        return list(rows)

    async def _load(
        self, ids: Sequence[uuid.UUID], vector: list[float] | None
    ) -> dict[uuid.UUID, RetrievedChunk]:
        if not ids:
            return {}
        model_id = self.embedder.model_id if vector is not None else None
        rows = await self.session.execute(
            select(Chunk, Document.filename, Document.title)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.id.in_(ids))
        )
        out: dict[uuid.UUID, RetrievedChunk] = {}
        for chunk, filename, title in rows:
            out[chunk.id] = RetrievedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                filename=filename,
                document_title=title or filename,
                chunk_type=chunk.chunk_type.value,
                content=chunk.content,
                context_header=chunk.context_header,
                section_path=chunk.section_path,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                bboxes=chunk.bboxes,
                data=chunk.data,
                # Embeddings are unit-normalised, so the dot product is the cosine similarity
                similarity=(
                    float(sum(a * b for a, b in zip(vector, chunk.embedding, strict=True)))
                    if vector is not None
                    and chunk.embedding is not None
                    and chunk.embedding_model == model_id
                    else 0.0
                ),
                # Without a query vector every chunk is scored like an unembedded one
                has_vector=chunk.embedding is not None and chunk.embedding_model == model_id,
            )
        return out

    async def search(
        self,
        owner_id: uuid.UUID,
        query: str,
        *,
        document_ids: Sequence[uuid.UUID] | None = None,
        limit: int | None = None,
        rerank: bool | None = None,
    ) -> SearchResult:
        limit = limit or self.settings.context_chunks
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        vector: list[float] | None
        try:
            vector = await self.embedder.embed_query(query)
        except Exception:
            # e.g. a hosted embedding quota is exhausted: degrade to keyword search, don't fail
            logger.warning("Query embedding failed; keyword-only search", exc_info=True)
            vector = None
        timings["embed_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        dense = await self._dense(owner_id, vector, document_ids) if vector is not None else []
        keyword = (
            await self._keyword(owner_id, query, document_ids)
            if self.settings.retrieval_mode == "hybrid" or vector is None
            else []
        )
        timings["search_ms"] = (time.perf_counter() - t0) * 1000

        fused = rrf_fuse(
            [dense, keyword],
            self.settings.rrf_k,
            [self.settings.dense_weight, self.settings.keyword_weight],
        )
        ordered = sorted(fused, key=lambda cid: -fused[cid])[: self.settings.rerank_candidates]
        chunks = await self._load(ordered, vector)
        dense_pos = {cid: i + 1 for i, cid in enumerate(dense)}
        keyword_pos = {cid: i + 1 for i, cid in enumerate(keyword)}
        candidates = []
        for cid in ordered:
            if cid in chunks:
                c = chunks[cid]
                c.rrf_score = fused[cid]
                c.dense_rank = dense_pos.get(cid)
                c.keyword_rank = keyword_pos.get(cid)
                candidates.append(c)

        use_rerank = self.settings.rerank_enabled if rerank is None else rerank
        if use_rerank and self.reranker and candidates:
            t0 = time.perf_counter()
            scores = await self.reranker.score(
                query, [f"{c.context_header}\n{c.content}" for c in candidates]
            )
            timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
            for c, s in zip(candidates, scores, strict=True):
                c.rerank_score = s
            candidates.sort(key=lambda c: -(c.rerank_score or 0.0))

        return SearchResult(
            chunks=candidates[:limit], timings_ms=timings, candidates=len(candidates)
        )


async def ready_document_ids(
    session: AsyncSession, owner_id: uuid.UUID, document_ids: Sequence[uuid.UUID] | None = None
) -> list[uuid.UUID]:
    stmt = select(Document.id).where(
        Document.owner_id == owner_id, Document.status == DocumentStatus.READY
    )
    if document_ids:
        stmt = stmt.where(Document.id.in_(document_ids))
    return list(await session.scalars(stmt))
