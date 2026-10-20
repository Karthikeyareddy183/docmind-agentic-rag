"""LLM enrichment at ingestion time. Best effort: failures degrade quality, never block indexing."""

import asyncio
import logging

from pydantic import BaseModel, Field

from docmind.config import Settings
from docmind.ingestion.chunker import ChunkDraft, count_tokens
from docmind.ingestion.parser import ParsedDocument
from docmind.llm.gateway import LLMError, LLMGateway, Usage, image_part

logger = logging.getLogger(__name__)

SUMMARY_INPUT_TOKENS = 3000
MAX_CONCURRENT_CALLS = 3


class DocumentOverview(BaseModel):
    title: str = Field(
        description="Specific title that always names the organization and the period or "
        "version when present, e.g. 'Northwind Analytics Annual Report FY2025' — never a "
        "generic title like 'Annual Report'"
    )
    summary: str = Field(description="2-3 sentences: what the document is and its key contents")


FIGURE_PROMPT = """You are indexing a figure from a document so it can be found by search
and used to answer questions.
Document: {title}
Section: {section}
Caption or nearby text: {caption}

Describe the figure in plain text:
1. The type of figure (bar chart, line chart, pie chart, diagram, photo, ...) and its title.
2. Every label and data value you can read, exactly as shown (e.g. "2023: 15.1%").
3. One sentence on the main trend or takeaway.
Do not invent values you cannot read."""

TABLE_PROMPT = """Write one sentence describing what this table contains (subject, rows,
columns, units), for a search index.
Document: {title}
Section: {section}

{table}"""


def _leading_text(parsed: ParsedDocument, limit: int) -> str:
    out: list[str] = []
    used = 0
    for el in parsed.elements:
        text = el.text if el.kind != "figure" else el.caption
        if not text:
            continue
        tokens = count_tokens(text)
        if used + tokens > limit:
            break
        out.append(text)
        used += tokens
    return "\n".join(out)


async def document_overview(
    parsed: ParsedDocument, filename: str, llm: LLMGateway, settings: Settings, usage: Usage
) -> DocumentOverview:
    fallback_title = parsed.title or filename.rsplit(".", 1)[0]
    text = _leading_text(parsed, SUMMARY_INPUT_TOKENS)
    if not text.strip():
        return DocumentOverview(title=fallback_title, summary="")
    try:
        return await llm.structured(
            [
                {
                    "role": "user",
                    "content": f"Filename: {filename}\n\nBeginning of the document:\n{text}\n\n"
                    "Return the document's title and a short summary.",
                }
            ],
            DocumentOverview,
            model=settings.llm_fast_model,
            usage=usage,
        )
    except LLMError:
        logger.warning("Document overview failed", exc_info=True)
        return DocumentOverview(title=fallback_title, summary="")


async def enrich_chunks(
    chunks: list[ChunkDraft], title: str, llm: LLMGateway, settings: Settings, usage: Usage
) -> None:
    """Add figure descriptions and table summaries in place."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

    async def describe(chunk: ChunkDraft) -> None:
        png = chunk.data.pop("_image_png", None)
        if not (settings.describe_figures and png):
            return
        prompt = FIGURE_PROMPT.format(
            title=title,
            section=chunk.section_path or "-",
            caption=chunk.data.get("caption") or "-",
        )
        async with semaphore:
            try:
                description = await llm.text(
                    [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}, image_part(png)],
                        }
                    ],
                    model=settings.vision_model,
                    usage=usage,
                )
            except LLMError:
                logger.warning("Figure description failed", exc_info=True)
                return
        chunk.data["description"] = description
        chunk.content = f"{chunk.content}\n{description}".strip()

    async def summarize(chunk: ChunkDraft) -> None:
        if not settings.summarize_tables:
            return
        prompt = TABLE_PROMPT.format(
            title=title, section=chunk.section_path or "-", table=chunk.content
        )
        async with semaphore:
            try:
                summary = await llm.text(
                    [{"role": "user", "content": prompt}],
                    model=settings.llm_fast_model,
                    usage=usage,
                )
            except LLMError:
                logger.warning("Table summary failed", exc_info=True)
                return
        chunk.data["summary"] = summary
        header, _, body = chunk.content.partition("\n")
        chunk.content = f"{header}\nSummary: {summary}\n{body}"

    tasks = []
    for chunk in chunks:
        if chunk.chunk_type == "figure":
            tasks.append(describe(chunk))
        elif chunk.chunk_type == "table":
            tasks.append(summarize(chunk))
    await asyncio.gather(*tasks)

    for chunk in chunks:  # never persist raw image bytes in JSONB
        chunk.data.pop("_image_png", None)
        chunk.token_count = count_tokens(chunk.content)
