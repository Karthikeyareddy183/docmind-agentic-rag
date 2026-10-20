"""The answering agent, as a LangGraph state machine.

    condense ─▶ plan ─┬─▶ tools ─▶ plan ...        (native tool calling, bounded rounds)
                      └─▶ generate ─▶ verify ─┬─▶ END
                                              └─▶ generate (one revision if the fact-check fails)

Streams custom events while it runs: ``step`` (what the agent is doing), ``token`` (answer text),
``revision`` (the fact-check replaced the answer). The final state carries the structured result.
"""

import contextlib
import json
import logging
import operator
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from docmind.agent import prompts
from docmind.agent.citations import (
    Citation,
    citation_coverage,
    cited_ids,
    compute_confidence,
    strip_invalid_citations,
)
from docmind.agent.table_calc import CalcPlan, compute
from docmind.config import Settings
from docmind.db.models import Chunk, ChunkType, Document, DocumentStatus
from docmind.llm.gateway import LLMError, LLMGateway, Usage
from docmind.retrieval.models import Embedder, Reranker
from docmind.retrieval.search import HybridSearcher, RetrievedChunk

logger = logging.getLogger(__name__)

SUMMARY_MAX_CHUNKS = 12
TOOL_RESULT_SNIPPET = 400


# ─── Runtime dependencies (not part of the persisted state) ─────────────


@dataclass
class AgentDeps:
    settings: Settings
    factory: async_sessionmaker[AsyncSession]
    llm: LLMGateway
    embedder: Embedder
    reranker: Reranker | None
    owner_id: uuid.UUID
    document_ids: list[uuid.UUID] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


class Verification(BaseModel):
    supported: bool
    unsupported_claims: list[str]


class AgentState(TypedDict, total=False):
    question: str
    history: list[dict[str, str]]
    standalone: str
    catalog: list[dict[str, Any]]
    planner_messages: list[dict[str, Any]]
    tool_rounds: int
    sources: Annotated[list[dict[str, Any]], operator.add]
    computations: Annotated[list[dict[str, Any]], operator.add]
    trace: Annotated[list[dict[str, Any]], operator.add]
    answer: str
    answerable: bool
    verified: bool | None
    problems: list[str]
    revised: bool


def _emit(event: dict[str, Any]) -> None:
    with contextlib.suppress(RuntimeError):  # not running under a streaming call
        get_stream_writer()(event)


def _step(name: str, **detail: Any) -> dict[str, Any]:
    event = {"type": "step", "name": name, **detail}
    _emit(event)
    return {"name": name, "at": round(time.time(), 3), **detail}


def _source_from_chunk(sid: str, c: RetrievedChunk) -> dict[str, Any]:
    return {
        "id": sid,
        "chunk_id": str(c.chunk_id),
        "document_id": str(c.document_id),
        "filename": c.filename,
        "document_title": c.document_title,
        "chunk_type": c.chunk_type,
        "page": c.page_start,
        "page_end": c.page_end,
        "bboxes": c.bboxes,
        "header": c.context_header,
        "content": c.content,
        "data": c.data,
        "score": round(c.score, 4),
    }


def _format_sources(sources: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"[{s['id']}] ({s['header']})\n{s['content']}" for s in sources)


def _format_computations(computations: list[dict[str, Any]]) -> str:
    if not computations:
        return ""
    lines = [
        f"- {c['explanation']}: {c['expression']} (from table {c['table_source']})"
        for c in computations
        if c.get("ok")
    ]
    return "\nComputed results (exact, use as given):\n" + "\n".join(lines) if lines else ""


# ─── Tools (schemas follow the OpenAI function-calling format; LiteLLM adapts them) ──

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Hybrid semantic + keyword search over the user's documents. "
            "Returns the most relevant passages, tables, and figure descriptions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Specific search query"},
                    "document_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: restrict to these document ids",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_table",
            "description": "Answer a numeric question by computing over an extracted table "
            "(sums, averages, differences, ratios, percentages, growth). Exact arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The numeric question"},
                    "document_id": {"type": "string", "description": "Optional document id"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_document",
            "description": "Load the main passages of one document to summarize it.",
            "parameters": {
                "type": "object",
                "properties": {"document_id": {"type": "string"}},
                "required": ["document_id"],
            },
        },
    },
]


def _valid_doc_ids(raw: Any, allowed: list[str]) -> list[uuid.UUID] | None:
    if not raw:
        return None
    ids = [d for d in (raw if isinstance(raw, list) else [raw]) if d in allowed]
    return [uuid.UUID(d) for d in ids] or None


class _Tooling:
    def __init__(self, deps: AgentDeps, state: AgentState) -> None:
        self.deps = deps
        self.state = state
        self.next_index = len(state.get("sources", [])) + 1
        self.known: dict[str, str] = {s["chunk_id"]: s["id"] for s in state.get("sources", [])}
        self.new_sources: list[dict[str, Any]] = []
        self.computations: list[dict[str, Any]] = []
        self.allowed = [d["id"] for d in state.get("catalog", [])]

    def _add(self, chunk: RetrievedChunk) -> str:
        key = str(chunk.chunk_id)
        if key in self.known:
            return self.known[key]
        sid = f"S{self.next_index}"
        self.next_index += 1
        self.known[key] = sid
        self.new_sources.append(_source_from_chunk(sid, chunk))
        return sid

    def _scope(self, requested: list[uuid.UUID] | None) -> list[uuid.UUID] | None:
        return requested or (self.deps.document_ids or None)

    async def search(self, args: dict[str, Any]) -> str:
        query = str(args.get("query") or self.state["standalone"])
        scope = self._scope(_valid_doc_ids(args.get("document_ids"), self.allowed))
        async with self.deps.factory() as session:
            searcher = HybridSearcher(
                session, self.deps.settings, self.deps.embedder, self.deps.reranker
            )
            result = await searcher.search(self.deps.owner_id, query, document_ids=scope)
        _emit(
            {
                "type": "step",
                "name": "search",
                "query": query,
                "hits": len(result.chunks),
                "top_score": round(result.top_score, 3),
                "timings_ms": {k: round(v) for k, v in result.timings_ms.items()},
            }
        )
        if not result.chunks:
            return "No matching passages."
        lines = []
        for c in result.chunks:
            sid = self._add(c)
            snippet = c.content[:TOOL_RESULT_SNIPPET].replace("\n", " ")
            lines.append(
                f"[{sid}] score={c.score:.2f} {c.filename} p{c.page_start} "
                f"({c.chunk_type}): {snippet}"
            )
        return "\n".join(lines)

    async def query_table(self, args: dict[str, Any]) -> str:
        question = str(args.get("question") or self.state["standalone"])
        scope = self._scope(_valid_doc_ids(args.get("document_id"), self.allowed))
        async with self.deps.factory() as session:
            searcher = HybridSearcher(
                session, self.deps.settings, self.deps.embedder, self.deps.reranker
            )
            result = await searcher.search(
                self.deps.owner_id, question, document_ids=scope, limit=12
            )
        tables = [
            c for c in result.chunks if c.chunk_type == ChunkType.TABLE.value and c.data.get("rows")
        ][:3]
        if not tables:
            return "No relevant table found; use search_documents instead."
        ids = [self._add(t) for t in tables]
        listing = "\n\n".join(
            f"[{sid}] {t.context_header}\ncolumns: {t.data['columns']}\nrows:\n"
            + "\n".join(" | ".join(r) for r in t.data["rows"])
            for sid, t in zip(ids, tables, strict=True)
        )
        try:
            plan = await self.deps.llm.structured(
                [
                    {
                        "role": "user",
                        "content": prompts.TABLE_PLAN.format(question=question, tables=listing),
                    }
                ],
                CalcPlan,
                usage=self.deps.usage,
            )
        except LLMError as exc:
            return f"Could not plan a calculation: {exc}"
        table = next(
            (t for sid, t in zip(ids, tables, strict=True) if sid == plan.table_source), tables[0]
        )
        source_id = self.known[str(table.chunk_id)]
        calc = compute(plan, table.data["columns"], table.data["rows"])
        record = {
            "table_source": source_id,
            "explanation": plan.explanation,
            "operation": plan.operation,
            **calc.model_dump(),
        }
        self.computations.append(record)
        _emit(
            {
                "type": "step",
                "name": "calculate",
                "expression": calc.expression or calc.error,
                "ok": calc.ok,
                "table": source_id,
            }
        )
        if not calc.ok:
            return f"Calculation failed ({calc.error}). Table {source_id} is available as a source."
        return f"Computed from {source_id}: {plan.explanation} = {calc.expression}"

    async def summarize(self, args: dict[str, Any]) -> str:
        doc_id = _valid_doc_ids(args.get("document_id"), self.allowed)
        if not doc_id:
            return "Unknown document id."
        async with self.deps.factory() as session:
            document = await session.get(Document, doc_id[0])
            rows = list(
                await session.scalars(
                    select(Chunk)
                    .where(Chunk.document_id == doc_id[0], Chunk.owner_id == self.deps.owner_id)
                    .order_by(Chunk.ordinal)
                    .limit(SUMMARY_MAX_CHUNKS)
                )
            )
        if document is None or not rows:
            return "Document has no indexed content."
        for c in rows:
            self._add(
                RetrievedChunk(
                    chunk_id=c.id,
                    document_id=c.document_id,
                    filename=document.filename,
                    document_title=document.title or document.filename,
                    chunk_type=c.chunk_type.value,
                    content=c.content,
                    context_header=c.context_header,
                    section_path=c.section_path,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    bboxes=c.bboxes,
                    data=c.data,
                    rrf_score=0.03,
                    rerank_score=0.9,
                )
            )
        _emit(
            {
                "type": "step",
                "name": "summarize",
                "document": document.filename,
                "passages": len(rows),
            }
        )
        name = document.title or document.filename
        return f"Loaded {len(rows)} passages of '{name}'. Summary: {document.summary or '-'}"


# ─── Nodes ──────────────────────────────────────────────────────────────


async def condense(state: AgentState, runtime: Runtime[AgentDeps]) -> dict[str, Any]:
    deps = runtime.context
    question = state["question"]
    history = state.get("history") or []
    standalone = question
    if history:
        convo = "\n".join(f"{m['role']}: {m['content'][:600]}" for m in history[-6:])
        try:
            standalone = (
                await deps.llm.text(
                    [
                        {
                            "role": "user",
                            "content": prompts.CONDENSE.format(history=convo, question=question),
                        }
                    ],
                    model=deps.settings.llm_fast_model,
                    usage=deps.usage,
                )
                or question
            )
        except LLMError:
            logger.warning("Condense failed; using the raw question", exc_info=True)

    async with deps.factory() as session:
        stmt = select(Document).where(
            Document.owner_id == deps.owner_id, Document.status == DocumentStatus.READY
        )
        if deps.document_ids:
            stmt = stmt.where(Document.id.in_(deps.document_ids))
        docs = list(await session.scalars(stmt.order_by(Document.created_at)))
    catalog = [
        {
            "id": str(d.id),
            "filename": d.filename,
            "title": d.title or d.filename,
            "pages": d.pages,
            "summary": (d.summary or "")[:300],
        }
        for d in docs
    ]
    trace = [_step("understand", standalone=standalone, documents=len(catalog))]
    return {
        "standalone": standalone,
        "catalog": catalog,
        "tool_rounds": 0,
        "planner_messages": [],
        "trace": trace,
    }


async def plan(state: AgentState, runtime: Runtime[AgentDeps]) -> dict[str, Any]:
    deps = runtime.context
    catalog = (
        "\n".join(
            f"- id={d['id']} | {d['title']} ({d['filename']}, {d['pages']} pages): {d['summary']}"
            for d in state.get("catalog", [])
        )
        or "(none)"
    )
    messages = state.get("planner_messages") or [
        {"role": "system", "content": prompts.PLANNER_SYSTEM.format(catalog=catalog)},
        {"role": "user", "content": state["standalone"]},
    ]
    first_round = state.get("tool_rounds", 0) == 0
    if deps.settings.agent_mode == "single":
        # Baseline: search once with the question itself, then answer
        calls = (
            [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "search_documents",
                        "arguments": json.dumps({"query": state["standalone"]}),
                    },
                }
            ]
            if first_round
            else None
        )
        trace = [_step("plan", tools=["search_documents"] if calls else ["done"])]
        return {
            "planner_messages": [
                *messages,
                {"role": "assistant", "content": None, "tool_calls": calls},
            ],
            "trace": trace,
        }
    try:
        message = await deps.llm.complete(
            messages,
            tools=TOOLS,
            tool_choice="required" if first_round else "auto",
            usage=deps.usage,
        )
    except LLMError:
        logger.warning("Planner failed; falling back to a direct search", exc_info=True)
        message = None

    tool_calls = []
    if message is not None and message.tool_calls:
        tool_calls = [
            {
                "id": tc.id or f"call_{i}",
                "name": tc.function.name,
                "arguments": tc.function.arguments or "{}",
            }
            for i, tc in enumerate(message.tool_calls)
        ]
    elif first_round:  # always look at the documents at least once
        tool_calls = [
            {
                "id": "call_0",
                "name": "search_documents",
                "arguments": json.dumps({"query": state["standalone"]}),
            }
        ]

    assistant = {
        "role": "assistant",
        "content": (message.content if message is not None else None) or None,
        "tool_calls": [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            }
            for tc in tool_calls
        ]
        or None,
    }
    trace = [
        _step("plan", tools=[f"{tc['name']}({tc['arguments']})" for tc in tool_calls] or ["done"])
    ]
    return {"planner_messages": [*messages, assistant], "trace": trace}


def after_plan(state: AgentState, runtime: Runtime[AgentDeps]) -> str:
    last = (state.get("planner_messages") or [{}])[-1]
    if (
        last.get("tool_calls")
        and state.get("tool_rounds", 0) < runtime.context.settings.agent_max_tool_rounds
    ):
        return "tools"
    return "generate"


async def tools(state: AgentState, runtime: Runtime[AgentDeps]) -> dict[str, Any]:
    tooling = _Tooling(runtime.context, state)
    handlers = {
        "search_documents": tooling.search,
        "query_table": tooling.query_table,
        "summarize_document": tooling.summarize,
    }
    messages = list(state["planner_messages"])
    trace = []
    for call in messages[-1]["tool_calls"]:
        name = call["function"]["name"]
        try:
            args = json.loads(call["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        handler = handlers.get(name)
        try:
            result = await handler(args) if handler else f"Unknown tool {name}"
        except Exception as exc:
            logger.exception("Tool failed", extra={"tool": name})
            result = f"Tool error: {type(exc).__name__}"
        messages.append(
            {"role": "tool", "tool_call_id": call["id"], "name": name, "content": result}
        )
        trace.append({"name": "tool", "tool": name, "args": args, "result_preview": result[:300]})
    return {
        "planner_messages": messages,
        "tool_rounds": state.get("tool_rounds", 0) + 1,
        "sources": tooling.new_sources,
        "computations": tooling.computations,
        "trace": trace,
    }


def _relevant_sources(state: AgentState, settings: Settings) -> list[dict[str, Any]]:
    sources = sorted(state.get("sources", []), key=lambda s: -s["score"])
    computed_tables = {c["table_source"] for c in state.get("computations", []) if c.get("ok")}
    keep = [
        s for s in sources if s["score"] >= settings.min_relevance or s["id"] in computed_tables
    ]
    return keep[: max(settings.context_chunks + 2, len(computed_tables))]


async def generate(state: AgentState, runtime: Runtime[AgentDeps]) -> dict[str, Any]:
    deps = runtime.context
    sources = _relevant_sources(state, deps.settings)
    computations = [c for c in state.get("computations", []) if c.get("ok")]
    revising = bool(state.get("problems"))

    if not sources and not computations:
        _emit({"type": "token", "text": prompts.NOT_FOUND})
        return {
            "answer": prompts.NOT_FOUND,
            "answerable": False,
            "verified": None,
            "trace": [_step("generate", outcome="not_found")],
        }

    if revising:
        _emit({"type": "revision", "problems": state.get("problems", [])})
    trace = [_step("revise" if revising else "generate", sources=[s["id"] for s in sources])]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompts.ANSWER_SYSTEM},
        {
            "role": "user",
            "content": prompts.ANSWER_USER.format(
                question=state["standalone"],
                sources=_format_sources(sources),
                computations=_format_computations(computations),
            ),
        },
    ]
    if revising:
        messages += [
            {"role": "assistant", "content": state.get("answer", "")},
            {
                "role": "user",
                "content": prompts.REVISE_USER.format(
                    problems="\n".join(f"- {p}" for p in state["problems"])
                ),
            },
        ]
    parts: list[str] = []
    try:
        async for delta in deps.llm.stream(messages, usage=deps.usage):
            parts.append(delta)
            _emit({"type": "token", "text": delta})
    except LLMError as exc:
        logger.warning("Answer generation failed", extra={"error": str(exc)})
        if revising and state.get("answer"):  # keep the unrevised draft rather than nothing
            return {"revised": True, "trace": [*trace, _step("revise", outcome="failed")]}
        _emit({"type": "revision", "problems": []})
        _emit({"type": "token", "text": prompts.LLM_UNAVAILABLE})
        return {
            "answer": prompts.LLM_UNAVAILABLE,
            "answerable": False,
            "verified": None,
            "trace": [*trace, _step("generate", outcome="llm_unavailable")],
        }
    answer = strip_invalid_citations("".join(parts).strip(), {s["id"] for s in sources})
    return {"answer": answer, "answerable": True, "revised": revising, "trace": trace}


async def verify(state: AgentState, runtime: Runtime[AgentDeps]) -> dict[str, Any]:
    deps = runtime.context
    if not state.get("answerable") or deps.settings.agent_mode == "single":
        return {}
    by_id = {s["id"]: s for s in state.get("sources", [])}
    cited = [by_id[i] for i in cited_ids(state["answer"]) if i in by_id]
    if not cited:
        verdict = Verification(supported=False, unsupported_claims=["The answer cites no sources."])
    else:
        try:
            verdict = await deps.llm.structured(
                [
                    {
                        "role": "user",
                        "content": prompts.VERIFY.format(
                            question=state["standalone"],
                            answer=state["answer"],
                            sources=_format_sources(cited),
                            computations=_format_computations(
                                [c for c in state.get("computations", []) if c.get("ok")]
                            ),
                        ),
                    }
                ],
                Verification,
                model=deps.settings.llm_fast_model,
                usage=deps.usage,
            )
        except LLMError:
            logger.warning("Verification failed to run", exc_info=True)
            return {"verified": None, "trace": [_step("verify", outcome="skipped")]}
    trace = [_step("verify", supported=verdict.supported, problems=verdict.unsupported_claims[:5])]
    return {
        "verified": verdict.supported,
        "problems": [] if verdict.supported else verdict.unsupported_claims,
        "trace": trace,
    }


def after_verify(state: AgentState) -> str:
    if state.get("verified") is False and not state.get("revised"):
        return "generate"
    return END


def build_graph() -> Any:
    graph = StateGraph(AgentState, context_schema=AgentDeps)
    graph.add_node("condense", condense)
    graph.add_node("plan", plan)
    graph.add_node("tools", tools)
    graph.add_node("generate", generate)
    graph.add_node("verify", verify)
    graph.add_edge(START, "condense")
    graph.add_edge("condense", "plan")
    graph.add_conditional_edges("plan", after_plan, {"tools": "tools", "generate": "generate"})
    graph.add_edge("tools", "plan")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", after_verify, {"generate": "generate", END: END})
    return graph.compile()


_GRAPH = None


def get_graph() -> Any:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


# ─── Public entry point ─────────────────────────────────────────────────


class AgentResult(BaseModel):
    answer: str
    answerable: bool
    citations: list[Citation]
    confidence: float
    verified: bool | None
    computations: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    usage: dict[str, Any]
    model: str
    latency_ms: int


def build_result(state: AgentState, deps: AgentDeps, started: float) -> AgentResult:
    by_id = {s["id"]: s for s in state.get("sources", [])}
    citations = [
        Citation(
            id=sid,
            chunk_id=s["chunk_id"],
            document_id=s["document_id"],
            filename=s["filename"],
            document_title=s["document_title"],
            chunk_type=s["chunk_type"],
            page=s["page"],
            page_end=s["page_end"],
            bboxes=s["bboxes"],
            snippet=s["content"][:500],
            score=s["score"],
        )
        for sid in cited_ids(state.get("answer", ""))
        if (s := by_id.get(sid))
    ]
    answerable = bool(state.get("answerable"))
    computations = [c for c in state.get("computations", []) if c.get("ok")]
    top = max((c.score for c in citations), default=0.0)
    confidence = (
        compute_confidence(
            top_relevance=top,
            coverage=citation_coverage(state.get("answer", "")),
            verified=state.get("verified"),
            used_calculation=bool(computations),
        )
        if answerable
        else 0.0
    )
    return AgentResult(
        answer=state.get("answer", ""),
        answerable=answerable,
        citations=citations,
        confidence=confidence,
        verified=state.get("verified"),
        computations=computations,
        trace=state.get("trace", []),
        usage=deps.usage.as_dict(),
        model=deps.settings.llm_model,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


async def run_agent(
    question: str, deps: AgentDeps, history: list[dict[str, str]] | None = None
) -> AsyncIterator[dict[str, Any]]:
    """Yield streaming events; the last one is ``{"type": "final", "result": AgentResult}``."""
    started = time.perf_counter()
    final_state: AgentState = {}
    async for mode, payload in get_graph().astream(
        {"question": question, "history": history or []},
        context=deps,
        stream_mode=["custom", "values"],
    ):
        if mode == "custom":
            yield payload
        else:
            final_state = payload
    yield {"type": "final", "result": build_result(final_state, deps, started)}
