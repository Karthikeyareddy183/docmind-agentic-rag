# Retrieval bake-off

58 chunks from 9 documents; 44 answerable questions. Hit = a chunk from the gold document/page in the top 5. Latency is median per query on this CPU (query embedding + rerank of the top 24 hybrid candidates).

| Embedder | Dim | Strategy | Reranker | Recall@1 | Recall@3 | Recall@5 | MRR@10 | Query latency (ms) | Index time (s) |
|---|---|---|---|---|---|---|---|---|---|
| `local:BAAI/bge-small-en-v1.5` | 384 | dense | — | 0.614 | 0.795 | 0.841 | 0.713 | 100 | 4.4 |
| `—` | — | keyword | — | 0.636 | 0.773 | 0.909 | 0.737 | 0 | 0.0 |
| `local:BAAI/bge-small-en-v1.5` | 384 | hybrid | — | 0.636 | 0.886 | 0.932 | 0.767 | 100 | 4.4 |
| `local:BAAI/bge-base-en-v1.5` | 768 | dense | — | 0.682 | 0.909 | 0.932 | 0.792 | 89 | 21.7 |
| `local:BAAI/bge-base-en-v1.5` | 768 | hybrid | — | 0.727 | 0.909 | 0.932 | 0.822 | 89 | 21.7 |
| `local:BAAI/bge-m3` | 1024 | dense | — | 0.818 | 0.955 | 0.977 | 0.883 | 372 | 59.2 |
| `local:BAAI/bge-m3` | 1024 | hybrid | — | 0.750 | 0.955 | 0.955 | 0.848 | 372 | 59.2 |
| `local:Qwen/Qwen3-Embedding-0.6B` | 1024 | dense | — | 0.705 | 0.864 | 0.955 | 0.793 | 1584 | 679.2 |
| `local:Qwen/Qwen3-Embedding-0.6B` | 1024 | hybrid | — | 0.773 | 0.932 | 0.932 | 0.852 | 1584 | 679.2 |
| `litellm:gemini/gemini-embedding-001` | 1024 | dense | — | 0.841 | 0.932 | 0.955 | 0.896 | 427 | 4.0 |
| `litellm:gemini/gemini-embedding-001` | 1024 | hybrid | — | 0.727 | 0.932 | 0.955 | 0.834 | 427 | 4.0 |
| `local:Qwen/Qwen3-Embedding-0.6B` | 1024 | hybrid+rerank | `cross-encoder/ms-marco-MiniLM-L6-v2` | 0.636 | 0.841 | 0.909 | 0.743 | 2648 | 679.2 |
| `local:Qwen/Qwen3-Embedding-0.6B` | 1024 | hybrid+rerank | `BAAI/bge-reranker-base` | 0.795 | 0.955 | 0.977 | 0.884 | 8397 | 679.2 |
| `local:Qwen/Qwen3-Embedding-0.6B` | 1024 | hybrid+rerank | `BAAI/bge-reranker-v2-m3` | 0.750 | 0.955 | 0.977 | 0.857 | 31596 | 679.2 |
