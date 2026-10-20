"""Tune hybrid fusion and reranking for the chosen embedder (run after retrieval_bakeoff.py).

Sweeps the keyword weight in weighted RRF and reranker candidate depth / input length, reporting
recall and per-query CPU latency. Uses the chunk cache built by the bake-off.

    python eval/fusion_sweep.py
"""

import asyncio
import json
import pickle
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import retrieval_bakeoff  # noqa: E402
from retrieval_bakeoff import CACHE, REPORTS, _tokens, load_questions, metrics, summarize  # noqa: E402

# The cache was pickled from retrieval_bakeoff run as __main__
sys.modules["__main__"].IndexedChunk = retrieval_bakeoff.IndexedChunk  # type: ignore[attr-defined]

from docmind.retrieval.models import LocalEmbedder, Reranker  # noqa: E402

EMBEDDER = "BAAI/bge-m3"
KEYWORD_WEIGHTS = [0.0, 0.25, 0.5, 1.0]
# CPU budget: the bake-off showed v2-m3 at ~30 s/query on this machine, so sweep the smaller
# reranker with a short candidate list and truncated inputs
RERANK = [
    ("BAAI/bge-reranker-base", 8, 256),
]


def weighted_rrf(dense: list[int], keyword: list[int], kw_weight: float, k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for rank, i in enumerate(dense):
        scores[i] = scores.get(i, 0.0) + 1.0 / (k + rank + 1)
    for rank, i in enumerate(keyword):
        scores[i] = scores.get(i, 0.0) + kw_weight / (k + rank + 1)
    return sorted(scores, key=lambda i: -scores[i])


async def main() -> None:
    from rank_bm25 import BM25Okapi

    chunks = pickle.loads(CACHE.read_bytes())  # noqa: S301 (our own cache)
    questions = load_questions()
    texts = [c.draft.index_text for c in chunks]
    bm25 = BM25Okapi([_tokens(t) for t in texts])
    embedder = LocalEmbedder(EMBEDDER)
    vecs = await embedder.embed_documents(texts)

    per_q = []
    for q in questions:
        qv = await embedder.embed_query(q["question"])
        sims = [sum(a * b for a, b in zip(qv, v, strict=True)) for v in vecs]
        dense = sorted(range(len(chunks)), key=lambda i: -sims[i])[:40]
        kw = bm25.get_scores(_tokens(q["question"]))
        keyword = [i for i in sorted(range(len(chunks)), key=lambda i: -kw[i]) if kw[i] > 0][:40]
        per_q.append({"q": q, "dense": dense, "keyword": keyword})

    results = []
    for w in KEYWORD_WEIGHTS:
        ranked = [weighted_rrf(p["dense"], p["keyword"], w) for p in per_q]
        m = summarize([metrics(r, chunks, p["q"]) for r, p in zip(ranked, per_q, strict=True)])
        results.append({"keyword_weight": w, "reranker": None, **m, "rerank_ms": 0.0})
        print(
            f"kw={w:<4} no-rerank        R@1={m['recall@1']:.3f} R@3={m['recall@3']:.3f} MRR={m['mrr@10']:.3f}"
        )

    for model_id, depth, max_len in RERANK:
        reranker = Reranker(model_id)
        reranker._model.max_length = max_len
        await reranker.score("warm up", ["warm up"])
        for w in (0.25, 0.5):
            mets, lat = [], []
            for p in per_q:
                cand = weighted_rrf(p["dense"], p["keyword"], w)[:depth]
                t0 = time.perf_counter()
                scores = await reranker.score(p["q"]["question"], [texts[i] for i in cand])
                lat.append((time.perf_counter() - t0) * 1000)
                ranked = [i for _, i in sorted(zip(scores, cand, strict=True), key=lambda x: -x[0])]
                mets.append(metrics(ranked, chunks, p["q"]))
            m = summarize(mets)
            results.append(
                {
                    "keyword_weight": w,
                    "reranker": model_id,
                    "depth": depth,
                    "max_len": max_len,
                    **m,
                    "rerank_ms": statistics.median(lat),
                }
            )
            print(
                f"kw={w:<4} {model_id.split('/')[-1]:22s} d={depth} len={max_len} "
                f"R@1={m['recall@1']:.3f} R@3={m['recall@3']:.3f} MRR={m['mrr@10']:.3f} {statistics.median(lat):.0f}ms"
            )

    (REPORTS / "fusion_sweep.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = [
        "# Fusion and reranking sweep (bge-m3)",
        "",
        "| Keyword weight | Reranker | Candidates | Max len | Recall@1 | Recall@3 | MRR@10 | Rerank ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['keyword_weight']} | {r['reranker'] or '—'} | {r.get('depth', '—')} | {r.get('max_len', '—')} "
            f"| {r['recall@1']:.3f} | {r['recall@3']:.3f} | {r['mrr@10']:.3f} | {r['rerank_ms']:.0f} |"
        )
    (REPORTS / "fusion_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
