"""End-to-end evaluation of DocMind through its real API, in-process.

1. Migrates a dedicated ``docmind_eval`` schema and runs the actual FastAPI app + ingestion worker.
2. Uploads the corpus once (re-runs reuse it) and waits until every document is indexed.
3. Asks every question in ``eval/datasets/questions.jsonl`` under each configuration:

   naive    dense retrieval only, single pass (classic RAG baseline)
   hybrid   + keyword search fused with weighted RRF
   full     + LangGraph agent: tool planning, exact table maths, fact-check & revise

4. Scores each answer:
   - retrieval hit   a cited source comes from the gold document and page
   - correct         every required fact appears in the answer (deterministic string check)
   - judge_correct   an LLM judge agrees with the reference answer (semantic check)
   - grounded        an LLM judge finds every claim supported by the cited passages
   - abstention      unanswerable questions are declined; answerable ones are not
   plus latency, tokens, cost, and confidence calibration.

    python eval/run_eval.py                          # all configurations
    python eval/run_eval.py --configs full --limit 10

Writes eval/reports/e2e_results.json and eval/reports/e2e_report.md.
"""

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND / "src"))

import os  # noqa: E402

os.chdir(BACKEND)  # settings read backend/.env

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from docmind.agent.prompts import LLM_UNAVAILABLE  # noqa: E402
from docmind.api.app import create_app  # noqa: E402
from docmind.config import Settings, get_settings  # noqa: E402
from docmind.llm.gateway import LLMError, LLMGateway, Usage  # noqa: E402

CORPUS = ROOT / "eval" / "corpus" / "files"
QUESTIONS = ROOT / "eval" / "datasets" / "questions.jsonl"
REPORTS = ROOT / "eval" / "reports"
EVAL_SCHEMA = "docmind_eval"
# Grade with a different model family than the one answering (set via --judge-model)
JUDGE_MODEL = ""
EVAL_USER = {"email": "eval@docmind.dev", "password": "eval-harness-password"}

CONFIGS: dict[str, dict[str, Any]] = {
    # Classic RAG baseline: vector search, one retrieval, one answer
    "naive": {"retrieval_mode": "dense", "agent_mode": "single"},
    # + keyword search, weighted RRF (keyword weight tuned in fusion_sweep.py)
    "hybrid": {"retrieval_mode": "hybrid", "agent_mode": "single"},
    # + LangGraph agent: tool planning, exact table maths, fact-check and revision
    "full": {"retrieval_mode": "hybrid", "agent_mode": "agent"},
}
ABSTAIN = re.compile(
    r"couldn.t find|could not find|not (?:contain|mention|include|state|provide)|no information|does not (?:say|specify)|isn.t (?:mentioned|stated)|not available in",
    re.I,
)


class Judgement(BaseModel):
    correct: bool
    grounded: bool
    reason: str


JUDGE_PROMPT = """You are grading a question-answering system.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Cited passages the system relied on:
{passages}

Decide:
- correct: does the system answer convey the same key facts as the reference (numbers must match; extra correct detail is fine)? If the reference says the information is not in the documents, the system must decline to answer.
- grounded: is every factual claim in the system answer supported by the cited passages? (If the system declined to answer, grounded = true.)
Give a one-sentence reason."""


def eval_settings(**overrides: Any) -> Settings:
    # Pinned: the published results and the eval index were built with local bge-m3
    return get_settings().model_copy(
        update={
            "db_schema": EVAL_SCHEMA,
            "db_pool_size": 3,
            "embedding_model": "local:BAAI/bge-m3",
            **overrides,
        }
    )


def migrate(settings: Settings) -> None:
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.attributes["settings"] = settings
    command.upgrade(cfg, "head")


async def login(client: AsyncClient) -> dict[str, str]:
    await client.post("/api/v1/auth/register", json=EVAL_USER)  # 409 if it exists: fine
    resp = await client.post(
        "/api/v1/auth/login",
        data={"username": EVAL_USER["email"], "password": EVAL_USER["password"]},
    )
    resp.raise_for_status()
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


async def ingest_corpus(timeout_s: int = 3600) -> None:
    settings = eval_settings(run_worker_in_api=True)
    app = create_app(settings, start_background=True)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app), base_url="http://eval", timeout=600
        ) as client,
    ):
        headers = await login(client)
        for path in sorted(CORPUS.iterdir()):
            with path.open("rb") as fh:
                resp = await client.post(
                    "/api/v1/documents", headers=headers, files={"file": (path.name, fh)}
                )
            resp.raise_for_status()
            doc = resp.json()
            if doc["status"] == "failed":
                await client.post(f"/api/v1/documents/{doc['id']}/reprocess", headers=headers)
        started = time.time()
        while True:
            try:
                resp = await client.get("/api/v1/documents?limit=100", headers=headers)
                resp.raise_for_status()
            except Exception as exc:  # transient network/DB hiccup: keep waiting
                print(
                    f"  ingestion: status check failed ({type(exc).__name__}); retrying", flush=True
                )
                await asyncio.sleep(10)
                continue
            docs = resp.json()["items"]
            pending = [d for d in docs if d["status"] in ("uploaded", "processing")]
            failed = [d for d in docs if d["status"] == "failed"]
            print(
                f"  ingestion: {len(docs) - len(pending) - len(failed)}/{len(docs)} ready, {len(pending)} pending, {len(failed)} failed",
                flush=True,
            )
            if not pending:
                if failed:
                    raise RuntimeError(
                        f"Ingestion failed: {[(d['filename'], d['error']) for d in failed]}"
                    )
                return
            if time.time() - started > timeout_s:
                raise TimeoutError("Ingestion did not finish in time")
            await asyncio.sleep(10)


_DASHES = re.compile("[‐-―−]")  # Unicode hyphens/dashes/minus -> "-"


def _normalize(text: str) -> str:
    """Typography-insensitive form: models differ in their spaces (NBSP, narrow NBSP), dashes,
    and markdown emphasis, none of which changes whether a fact is present."""
    text = unicodedata.normalize("NFKC", text).replace("*", "")
    return re.sub(r"\s+", " ", _DASHES.sub("-", text)).lower()


def required_facts_present(answer: str, groups: list[str]) -> bool:
    normalized = _normalize(answer)
    return all(any(_normalize(alt) in normalized for alt in group.split("|")) for group in groups)


def gold_hit(citations: list[dict[str, Any]], q: dict[str, Any]) -> bool:
    docs = set(q["doc"].split("|")) if q["doc"] else set()
    pages = set(q["pages"])
    return any(
        c["filename"] in docs
        and (not pages or any(p in pages for p in range(c["page"], c["page_end"] + 1)))
        for c in citations
    )


async def judge(
    llm: LLMGateway, q: dict[str, Any], result: dict[str, Any], usage: Usage
) -> Judgement:
    passages = "\n\n".join(f"[{c['id']}] {c['snippet']}" for c in result["citations"]) or "(none)"
    try:
        return await llm.structured(
            [
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=q["question"],
                        reference=q["answer"],
                        answer=result["answer"],
                        passages=passages,
                    ),
                }
            ],
            Judgement,
            model=JUDGE_MODEL or None,
            usage=usage,
        )
    except LLMError as exc:
        return Judgement(correct=False, grounded=False, reason=f"judge failed: {exc}")


async def run_config(
    name: str, questions: list[dict[str, Any]], llm: LLMGateway, judge_usage: Usage
) -> list[dict[str, Any]]:
    settings = eval_settings(**CONFIGS[name])
    app = create_app(settings, start_background=False)
    rows: list[dict[str, Any]] = []
    unavailable_streak = 0
    async with (
        app.router.lifespan_context(app),
        # raise_app_exceptions=False: a failing request becomes a 500 row, not a crashed run
        AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://eval",
            timeout=600,
        ) as client,
    ):
        headers = await login(client)
        for i, q in enumerate(questions, 1):
            if unavailable_streak >= 3:
                print(f"  [{name}] stopping: the LLM provider is unavailable (quota?)", flush=True)
                rows.append({"id": "_aborted", "type": "meta", "error": "LLM provider unavailable"})
                break
            t0 = time.perf_counter()
            resp = await client.post(
                "/api/v1/ask", headers=headers, json={"question": q["question"]}
            )
            wall_ms = (time.perf_counter() - t0) * 1000
            if resp.status_code != 200:
                rows.append({"id": q["id"], "type": q["type"], "error": resp.text[:300]})
                print(
                    f"  [{name}] {i:2d}/{len(questions)} {q['id']} ERROR {resp.status_code}",
                    flush=True,
                )
                continue
            result = resp.json()
            if result["answer"] == LLM_UNAVAILABLE:
                unavailable_streak += 1
                rows.append({"id": q["id"], "type": q["type"], "error": "LLM unavailable"})
                print(f"  [{name}] {i:2d}/{len(questions)} {q['id']} LLM UNAVAILABLE", flush=True)
                continue
            unavailable_streak = 0
            unanswerable = q["type"] == "unanswerable"
            # A caveat ("...is not stated directly") next to the right facts is not a refusal
            declined_text = bool(ABSTAIN.search(result["answer"])) and not (
                q["must_include"] and required_facts_present(result["answer"], q["must_include"])
            )
            abstained = (not result["answerable"]) or declined_text
            verdict = await judge(llm, q, result, judge_usage)
            correct = (
                abstained
                if unanswerable
                else (not abstained and required_facts_present(result["answer"], q["must_include"]))
            )
            rows.append(
                {
                    "id": q["id"],
                    "type": q["type"],
                    "question": q["question"],
                    "answer": result["answer"],
                    "correct": correct,
                    "judge_correct": verdict.correct,
                    "grounded": verdict.grounded,
                    "judge_reason": verdict.reason,
                    "retrieval_hit": None if unanswerable else gold_hit(result["citations"], q),
                    "abstained": abstained,
                    "confidence": result["confidence"],
                    "verified": result["verified"],
                    "citations": [f"{c['filename']} p{c['page']}" for c in result["citations"]],
                    "latency_ms": wall_ms,
                    "tokens": result["usage"].get("total_tokens", 0),
                    "cost_usd": result["usage"].get("cost_usd", 0.0),
                    "llm_calls": result["usage"].get("calls", 0),
                    "tools": [s.get("tools") for s in result["trace"] if s.get("name") == "plan"],
                }
            )
            mark = "✓" if correct else "✗"
            print(
                f"  [{name}] {i:2d}/{len(questions)} {q['id']} {mark} judge={'✓' if verdict.correct else '✗'} conf={result['confidence']:.2f} {wall_ms / 1000:.1f}s",
                flush=True,
            )
    return rows


def pct(values: list[bool]) -> float:
    return 100.0 * sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    answerable = [r for r in ok if r["type"] != "unanswerable"]
    unanswerable = [r for r in ok if r["type"] == "unanswerable"]
    lat = sorted(r["latency_ms"] for r in ok) or [0.0]
    by_type: dict[str, float] = {}
    for t in sorted({r["type"] for r in answerable}):
        by_type[t] = pct([r["correct"] for r in answerable if r["type"] == t])
    correct_conf = [r["confidence"] for r in ok if r["correct"] and r["type"] != "unanswerable"]
    wrong_conf = [r["confidence"] for r in ok if not r["correct"] and r["type"] != "unanswerable"]
    return {
        "questions": len(rows),
        "errors": len(rows) - len(ok),
        "accuracy": pct([r["correct"] for r in ok]),
        "judge_accuracy": pct([r["judge_correct"] for r in ok]),
        "answerable_accuracy": pct([r["correct"] for r in answerable]),
        "retrieval_hit_rate": pct([bool(r["retrieval_hit"]) for r in answerable]),
        "groundedness": pct([r["grounded"] for r in ok]),
        "abstention_accuracy": pct([r["abstained"] for r in unanswerable]),
        "false_abstention_rate": pct([r["abstained"] for r in answerable]),
        "accuracy_by_type": by_type,
        "latency_p50_s": statistics.median(lat) / 1000,
        "latency_p95_s": lat[min(len(lat) - 1, int(0.95 * len(lat)))] / 1000,
        "avg_tokens": statistics.mean(r["tokens"] for r in ok) if ok else 0,
        "avg_llm_calls": statistics.mean(r["llm_calls"] for r in ok) if ok else 0,
        "cost_per_query_usd": statistics.mean(r["cost_usd"] for r in ok) if ok else 0,
        "mean_confidence_correct": statistics.mean(correct_conf) if correct_conf else None,
        "mean_confidence_wrong": statistics.mean(wrong_conf) if wrong_conf else None,
    }


def write_report(results: dict[str, dict[str, Any]], settings: Settings) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "e2e_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    names = list(results)
    s = {n: results[n]["summary"] for n in names}
    rows = [
        ("Answer accuracy (exact facts)", "accuracy", "{:.1f}%"),
        ("Answer accuracy (LLM judge)", "judge_accuracy", "{:.1f}%"),
        ("Retrieval hit rate (gold page cited)", "retrieval_hit_rate", "{:.1f}%"),
        ("Groundedness (claims supported by citations)", "groundedness", "{:.1f}%"),
        ("Correctly declined unanswerable", "abstention_accuracy", "{:.1f}%"),
        ("Wrongly declined answerable", "false_abstention_rate", "{:.1f}%"),
        ("Latency p50 (s)", "latency_p50_s", "{:.1f}"),
        ("Latency p95 (s)", "latency_p95_s", "{:.1f}"),
        ("LLM calls / question", "avg_llm_calls", "{:.1f}"),
        ("Tokens / question", "avg_tokens", "{:,.0f}"),
        ("Cost / question (USD)", "cost_per_query_usd", "${:.4f}"),
    ]
    lines = [
        "# DocMind end-to-end evaluation",
        "",
        f"Corpus: {len(list(CORPUS.iterdir()))} documents (annual reports with same-template competitors and a prior-year "
        "edition as hard negatives, scanned board minutes, a photographed invoice, policies). "
        f"Questions: {s[names[0]]['questions']} (text, table look-ups, table arithmetic, chart-only facts, OCR, "
        "multi-document, and unanswerable).",
        "",
        f"Retrieval: embeddings `{settings.embedding_model}`, "
        f"reranker {'`' + settings.reranker_model + '`' if settings.rerank_enabled else 'off'} "
        "(run locally on CPU). Generation and judge models are listed per configuration below.",
        "",
        "| Metric | " + " | ".join(f"`{n}`" for n in names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for label, key, fmt in rows:
        lines.append(f"| {label} | " + " | ".join(fmt.format(s[n][key]) for n in names) + " |")
    lines += [
        "",
        "**Accuracy by question type (exact facts)**",
        "",
        "| Type | " + " | ".join(f"`{n}`" for n in names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for t in sorted({t for n in names for t in s[n]["accuracy_by_type"]}):
        lines.append(
            f"| {t} | "
            + " | ".join(f"{s[n]['accuracy_by_type'].get(t, 0):.0f}%" for n in names)
            + " |"
        )
    lines += [
        "",
        "**Confidence calibration** (mean computed confidence)",
        "",
        "| Config | Correct answers | Wrong answers |",
        "|---|---|---|",
    ]
    for n in names:
        c, w = s[n]["mean_confidence_correct"], s[n]["mean_confidence_wrong"]
        fmt = lambda v: "n/a (none)" if v is None else f"{v:.2f}"  # noqa: E731
        lines.append(f"| `{n}` | {fmt(c)} | {fmt(w)} |")
    lines += ["", "Configurations:", ""]
    for n in names:
        model = results[n].get("llm_model")
        judged = results[n].get("judge_model")
        lines.append(
            f"- `{n}`: {CONFIGS[n]}"
            + (f" (generation `{model}`, judge `{judged}`)" if model else "")
        )
    (REPORTS / "e2e_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def _needs_rerun(row: dict[str, Any]) -> bool:
    """A row is incomplete if the request errored or the judge could not grade it."""
    return "error" in row or str(row.get("judge_reason", "")).startswith("judge failed")


async def main(
    configs: list[str], limit: int | None, skip_ingest: bool, judge_model: str, resume: bool
) -> None:
    global JUDGE_MODEL
    JUDGE_MODEL = judge_model
    settings = eval_settings()
    print(f"Migrating schema {EVAL_SCHEMA}...")
    await asyncio.to_thread(migrate, settings)
    if not skip_ingest:
        print("Ingesting corpus (first run parses + embeds; later runs reuse it)...")
        await ingest_corpus()
    questions = [
        json.loads(line)
        for line in QUESTIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    questions = questions[:limit] if limit else questions
    llm, judge_usage = LLMGateway(settings), Usage()

    results_path = REPORTS / "e2e_results.json"
    previous: dict[str, Any] = {}
    if resume and results_path.exists():
        previous = json.loads(results_path.read_text(encoding="utf-8"))
        previous.pop("_meta", None)
    results: dict[str, dict[str, Any]] = {k: v for k, v in previous.items() if k in CONFIGS}

    for name in configs:
        done = {
            r["id"]: r
            for r in previous.get(name, {}).get("rows", [])
            if r.get("type") != "meta" and not _needs_rerun(r)
        }
        todo = [q for q in questions if q["id"] not in done]
        print()
        print(f"=== {name}: {CONFIGS[name]} ({len(done)} kept, {len(todo)} to run)")
        fresh = {r["id"]: r for r in await run_config(name, todo, llm, judge_usage)} if todo else {}
        rows = [done.get(q["id"]) or fresh.get(q["id"]) for q in questions]
        rows = [r for r in rows if r is not None] + [
            r for r in fresh.values() if r.get("type") == "meta"
        ]
        results[name] = {
            "config": CONFIGS[name],
            # Configs can be run on different days/providers: record what produced each one
            "llm_model": settings.llm_model,
            "judge_model": judge_model or settings.llm_model,
            "summary": summarize(rows),
            "rows": rows,
        }

    ordered = {n: results[n] for n in CONFIGS if n in results}
    write_report(ordered, settings)
    ordered["_meta"] = {  # type: ignore[assignment]
        "judge_usage_this_run": judge_usage.as_dict(),
        "llm_model": settings.llm_model,
        "judge_model": judge_model or settings.llm_model,
    }
    results_path.write_text(json.dumps(ordered, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument(
        "--judge-model",
        default="",
        help="LiteLLM model for grading (default: the app's main model)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed rows from the last run; only run missing/failed questions",
    )
    args = parser.parse_args()
    asyncio.run(main(args.configs, args.limit, args.skip_ingest, args.judge_model, args.resume))
