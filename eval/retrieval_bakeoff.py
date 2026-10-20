"""Offline retrieval bake-off: embedding models × retrieval strategy × reranker, on this machine.

Parses + enriches the corpus once (cached), then measures for every answerable question whether a
chunk from the gold document/page lands in the top-k, plus per-query latency on CPU.

    python eval/retrieval_bakeoff.py                # full grid
    python eval/retrieval_bakeoff.py --quick        # one embedder, all rerankers

Writes eval/reports/retrieval_bakeoff.md and .json.
"""

import argparse
import asyncio
import json
import pickle
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from docmind.config import get_settings  # noqa: E402
from docmind.ingestion.chunker import ChunkDraft, chunk_document  # noqa: E402
from docmind.ingestion.enrich import document_overview, enrich_chunks  # noqa: E402
from docmind.ingestion.parser import parse_file  # noqa: E402
from docmind.llm.gateway import LLMGateway, Usage  # noqa: E402
from docmind.retrieval.models import LiteLLMEmbedder, LocalEmbedder, Reranker  # noqa: E402

CORPUS = ROOT / "eval" / "corpus" / "files"
QUESTIONS = ROOT / "eval" / "datasets" / "questions.jsonl"
REPORTS = ROOT / "eval" / "reports"
CACHE = Path("D:/docmind-env/tmp/bakeoff_chunks.pkl")

EMBEDDERS = [
    "local:BAAI/bge-small-en-v1.5",
    "local:BAAI/bge-base-en-v1.5",
    "local:BAAI/bge-m3",
    "local:Qwen/Qwen3-Embedding-0.6B",
    "litellm:gemini/gemini-embedding-001",
]
RERANKERS = [
    None,
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-v2-m3",
]
TOP_K = 5
CANDIDATES = 24
RRF_K = 60


@dataclass
class IndexedChunk:
    doc: str
    draft: ChunkDraft


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower())


async def build_chunks() -> list[IndexedChunk]:
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())  # noqa: S301 (our own cache file)
    settings = get_settings()
    llm, usage = LLMGateway(settings), Usage()
    out: list[IndexedChunk] = []
    for path in sorted(CORPUS.iterdir()):
        parsed = await asyncio.to_thread(parse_file, path)
        overview = await document_overview(parsed, path.name, llm, settings, usage)
        drafts = chunk_document(parsed, overview.title, settings.chunk_max_tokens)
        await enrich_chunks(drafts, overview.title, llm, settings, usage)
        out += [IndexedChunk(path.name, d) for d in drafts]
        print(f"  parsed {path.name}: {len(drafts)} chunks, title={overview.title!r}")
    print(f"  enrichment usage: {usage.as_dict()}")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(out))
    return out


def load_questions() -> list[dict]:
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line]
    return [q for q in rows if q["type"] != "unanswerable"]


def is_hit(chunk: IndexedChunk, q: dict) -> bool:
    docs = q["doc"].split("|")
    pages = set(q["pages"])
    return chunk.doc in docs and (
        not pages
        or any(p in pages for p in range(chunk.draft.page_start, chunk.draft.page_end + 1))
    )


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: -scores[i])


def metrics(ranked: list[int], chunks: list[IndexedChunk], q: dict) -> dict[str, float]:
    hits = [i for i, idx in enumerate(ranked[:10]) if is_hit(chunks[idx], q)]
    first = hits[0] if hits else None
    return {
        "r1": float(first is not None and first < 1),
        "r3": float(first is not None and first < 3),
        "r5": float(first is not None and first < TOP_K),
        "mrr": 1.0 / (first + 1) if first is not None else 0.0,
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        "recall@1": statistics.mean(r["r1"] for r in rows),
        "recall@3": statistics.mean(r["r3"] for r in rows),
        "recall@5": statistics.mean(r["r5"] for r in rows),
        "mrr@10": statistics.mean(r["mrr"] for r in rows),
    }


async def main(quick: bool) -> None:
    from rank_bm25 import BM25Okapi

    print("Building chunks (cached after first run)...")
    chunks = await build_chunks()
    questions = load_questions()
    texts = [c.draft.index_text for c in chunks]
    bm25 = BM25Okapi([_tokens(t) for t in texts])
    print(f"{len(chunks)} chunks, {len(questions)} answerable questions\n")

    embedders = EMBEDDERS[2:3] if quick else EMBEDDERS
    results: list[dict] = []
    runs: dict[str, list[dict]] = {}
    for model_id in embedders:
        kind, _, name = model_id.partition(":")
        t0 = time.perf_counter()
        embedder = LocalEmbedder(name) if kind == "local" else LiteLLMEmbedder(name, 1024)
        load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        doc_vecs = await embedder.embed_documents(texts)
        index_s = time.perf_counter() - t0
        await embedder.embed_query("warm up")

        per_q: list[dict] = []
        for q in questions:
            t0 = time.perf_counter()
            qv = await embedder.embed_query(q["question"])
            embed_ms = (time.perf_counter() - t0) * 1000
            sims = [sum(a * b for a, b in zip(qv, dv, strict=True)) for dv in doc_vecs]
            dense = sorted(range(len(chunks)), key=lambda i: -sims[i])[:40]
            kw = bm25.get_scores(_tokens(q["question"]))
            keyword = [i for i in sorted(range(len(chunks)), key=lambda i: -kw[i]) if kw[i] > 0][
                :40
            ]
            per_q.append(
                {
                    "q": q,
                    "dense": dense,
                    "keyword": keyword,
                    "hybrid": rrf([dense, keyword]),
                    "embed_ms": embed_ms,
                }
            )
        runs[model_id] = per_q

        for strategy in ("dense", "keyword", "hybrid"):
            if (
                strategy == "keyword"
                and results
                and any(r["strategy"] == "keyword" for r in results)
            ):
                continue  # BM25 does not depend on the embedder
            m = [metrics(p[strategy], chunks, p["q"]) for p in per_q]
            results.append(
                {
                    "embedder": model_id if strategy != "keyword" else "—",
                    "strategy": strategy,
                    "reranker": None,
                    **summarize(m),
                    "query_ms": statistics.median(p["embed_ms"] for p in per_q)
                    if strategy != "keyword"
                    else 0.0,
                    "index_s": index_s if strategy != "keyword" else 0.0,
                    "load_s": load_s,
                    "dim": embedder.dimension if strategy != "keyword" else 0,
                    "misses": [p["q"]["id"] for p, x in zip(per_q, m, strict=True) if not x["r5"]],
                }
            )
            r = results[-1]
            print(
                f"{r['embedder']:42s} {strategy:8s} R@1={r['recall@1']:.3f} R@3={r['recall@3']:.3f} R@5={r['recall@5']:.3f} MRR={r['mrr@10']:.3f} q={r['query_ms']:.0f}ms"
            )

    hybrids = [r for r in results if r["strategy"] == "hybrid"]
    best = max(hybrids, key=lambda r: (r["mrr@10"], r["recall@1"]))["embedder"]
    print()
    print(f"Reranker sweep on the best hybrid embedder: {best}")
    per_q = runs[best]
    base = next(r for r in hybrids if r["embedder"] == best)
    for rr_id in RERANKERS[1:]:
        reranker = Reranker(rr_id)
        await reranker.score("warm up", ["warm up"])
        m, lat = [], []
        for p in per_q:
            cand = p["hybrid"][:CANDIDATES]
            t0 = time.perf_counter()
            scores = await reranker.score(p["q"]["question"], [texts[i] for i in cand])
            lat.append((time.perf_counter() - t0) * 1000)
            ranked = [i for _, i in sorted(zip(scores, cand, strict=True), key=lambda x: -x[0])]
            m.append(metrics(ranked, chunks, p["q"]))
        results.append(
            {
                "embedder": best,
                "strategy": "hybrid+rerank",
                "reranker": rr_id,
                **summarize(m),
                "query_ms": base["query_ms"] + statistics.median(lat),
                "rerank_ms": statistics.median(lat),
                "index_s": base["index_s"],
                "load_s": base["load_s"],
                "dim": base["dim"],
                "misses": [p["q"]["id"] for p, x in zip(per_q, m, strict=True) if not x["r5"]],
            }
        )
        r = results[-1]
        print(
            f"  {rr_id:40s} R@1={r['recall@1']:.3f} R@3={r['recall@3']:.3f} R@5={r['recall@5']:.3f} MRR={r['mrr@10']:.3f} rerank={r['rerank_ms']:.0f}ms"
        )

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "retrieval_bakeoff.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = [
        "# Retrieval bake-off",
        "",
        f"{len(chunks)} chunks from {len({c.doc for c in chunks})} documents; {len(questions)} answerable questions. "
        f"Hit = a chunk from the gold document/page in the top {TOP_K}. Latency is median per query on this CPU "
        "(query embedding + rerank of the top 24 hybrid candidates).",
        "",
        "| Embedder | Dim | Strategy | Reranker | Recall@1 | Recall@3 | Recall@5 | MRR@10 | Query latency (ms) | Index time (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r['embedder']}` | {r['dim'] or '—'} | {r['strategy']} | {('`' + r['reranker'] + '`') if r['reranker'] else '—'} "
            f"| {r['recall@1']:.3f} | {r['recall@3']:.3f} | {r['recall@5']:.3f} | {r['mrr@10']:.3f} | {r['query_ms']:.0f} | {r['index_s']:.1f} |"
        )
    (REPORTS / "retrieval_bakeoff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {REPORTS / 'retrieval_bakeoff.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    asyncio.run(main(parser.parse_args().quick))
