# Fusion and reranking sweep (bge-m3)

| Keyword weight | Reranker | Candidates | Max len | Recall@1 | Recall@3 | MRR@10 | Rerank ms |
|---|---|---|---|---|---|---|---|
| 0.0 | — | — | — | 0.818 | 0.955 | 0.883 | 0 |
| 0.25 | — | — | — | 0.818 | 0.977 | 0.890 | 0 |
| 0.5 | — | — | — | 0.750 | 0.977 | 0.860 | 0 |
| 1.0 | — | — | — | 0.750 | 0.955 | 0.848 | 0 |
| 0.25 | BAAI/bge-reranker-base | 8 | 256 | 0.773 | 0.955 | 0.864 | 2699 |
| 0.5 | BAAI/bge-reranker-base | 8 | 256 | 0.773 | 0.932 | 0.863 | 2639 |
