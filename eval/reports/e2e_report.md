# DocMind end-to-end evaluation

Corpus: 9 documents (annual reports with same-template competitors and a prior-year edition as hard negatives, scanned board minutes, a photographed invoice, policies). Questions: 47 (text, table look-ups, table arithmetic, chart-only facts, OCR, multi-document, and unanswerable).

Retrieval: embeddings `local:BAAI/bge-m3`, reranker off (run locally on CPU). Generation and judge models are listed per configuration below.

| Metric | `naive` | `hybrid` | `full` |
|---|---|---|---|
| Answer accuracy (exact facts) | 97.9% | 100.0% | 97.9% |
| Answer accuracy (LLM judge) | 100.0% | 95.7% | 95.7% |
| Retrieval hit rate (gold page cited) | 95.5% | 97.7% | 97.7% |
| Groundedness (claims supported by citations) | 100.0% | 97.9% | 95.7% |
| Correctly declined unanswerable | 100.0% | 100.0% | 100.0% |
| Wrongly declined answerable | 2.3% | 0.0% | 0.0% |
| Latency p50 (s) | 2.6 | 2.8 | 26.3 |
| Latency p95 (s) | 3.5 | 3.6 | 95.2 |
| LLM calls / question | 1.0 | 1.0 | 4.6 |
| Tokens / question | 1,623 | 1,708 | 6,810 |
| Cost / question (USD) | $0.0002 | $0.0002 | $0.0002 |

**Accuracy by question type (exact facts)**

| Type | `naive` | `hybrid` | `full` |
|---|---|---|---|
| calc | 100% | 100% | 75% |
| figure | 100% | 100% | 100% |
| multi | 100% | 100% | 100% |
| ocr | 100% | 100% | 100% |
| table | 100% | 100% | 100% |
| text | 95% | 100% | 100% |

**Confidence calibration** (mean computed confidence)

| Config | Correct answers | Wrong answers |
|---|---|---|
| `naive` | 0.66 | 0.62 |
| `hybrid` | 0.66 | n/a (none) |
| `full` | 0.81 | 0.62 |

Configurations:

- `naive`: {'retrieval_mode': 'dense', 'agent_mode': 'single'} (generation `openrouter/anthropic/claude-haiku-5.5`, judge `openrouter/google/gemini-2.5-flash`)
- `hybrid`: {'retrieval_mode': 'hybrid', 'agent_mode': 'single'} (generation `openrouter/anthropic/claude-haiku-5.5`, judge `openrouter/google/gemini-2.5-flash`)
- `full`: {'retrieval_mode': 'hybrid', 'agent_mode': 'agent'} (generation `groq/openai/gpt-oss-120b`, judge `groq/qwen/qwen3.8-27b`)
