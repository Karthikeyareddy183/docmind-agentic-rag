"""Structure-aware chunking.

- Text is grouped within a section (never across a heading) up to a token budget.
- Every table and every figure is its own chunk, so a table is never split from its header row.
- Each chunk gets a contextual header ("Title › Section — page N") used for indexing only.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from docmind.ingestion.parser import Element, ParsedDocument

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@lru_cache(maxsize=1)
def _encoder() -> Any:
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text, disallowed_special=()))


@dataclass
class ChunkDraft:
    chunk_type: str  # text | table | figure
    content: str
    section_path: str
    page_start: int
    page_end: int
    bboxes: list[dict[str, Any]] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    context_header: str = ""
    token_count: int = 0

    @property
    def index_text(self) -> str:
        """What gets embedded: header gives the chunk its document/section context."""
        return f"{self.context_header}\n{self.content}" if self.context_header else self.content


def _split_long(text: str, max_tokens: int) -> list[str]:
    """Split an oversized paragraph on sentence boundaries."""
    parts: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        candidate = f"{current} {sentence}".strip()
        if current and count_tokens(candidate) > max_tokens:
            parts.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


class _SectionTracker:
    def __init__(self) -> None:
        self.stack: list[tuple[int, str]] = []

    def push(self, level: int, title: str) -> None:
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, title))

    @property
    def path(self) -> str:
        # The document title (level 0) is carried by the context header, not the path
        return " › ".join(t for lvl, t in self.stack if lvl > 0)


# Short text right before a table/figure is usually its title or intro ("Operating margin by year")
_LEAD_IN_MAX_TOKENS = 60


def chunk_document(parsed: ParsedDocument, title: str, max_tokens: int = 450) -> list[ChunkDraft]:
    chunks: list[ChunkDraft] = []
    sections = _SectionTracker()
    buffer: list[Element] = []

    def flush() -> None:
        if not buffer:
            return
        text = "\n".join(e.text for e in buffer)
        chunks.append(
            ChunkDraft(
                chunk_type="text",
                content=text,
                section_path=sections.path,
                page_start=min(e.page for e in buffer),
                page_end=max(e.page for e in buffer),
                bboxes=[e.bbox.as_dict() for e in buffer if e.bbox],
            )
        )
        buffer.clear()

    def buffered_tokens() -> int:
        return count_tokens("\n".join(e.text for e in buffer))

    def take_lead_in(page: int) -> tuple[str, list[dict[str, Any]]]:
        """Absorb a short same-page buffer into the following table/figure; else flush it."""
        if (
            buffer
            and all(e.page == page for e in buffer)
            and buffered_tokens() <= _LEAD_IN_MAX_TOKENS
        ):
            text = " ".join(e.text for e in buffer)
            boxes = [e.bbox.as_dict() for e in buffer if e.bbox]
            buffer.clear()
            return text, boxes
        flush()
        return "", []

    for el in parsed.elements:
        if el.kind == "heading":
            flush()
            sections.push(el.level, el.text)
        elif el.kind == "text":
            pieces = (
                _split_long(el.text, max_tokens)
                if count_tokens(el.text) > max_tokens
                else [el.text]
            )
            for piece in pieces:
                if buffer and buffered_tokens() + count_tokens(piece) > max_tokens:
                    flush()
                buffer.append(Element("text", piece, el.page, el.bbox))
        elif el.kind == "table" and el.table:
            lead_in, lead_boxes = take_lead_in(el.page)
            label = el.caption or sections.path or "Table"
            intro = f"\n{lead_in}" if lead_in else ""
            chunks.append(
                ChunkDraft(
                    chunk_type="table",
                    content=f"Table: {label}{intro}\n{el.table['markdown']}",
                    section_path=sections.path,
                    page_start=el.page,
                    page_end=el.page,
                    bboxes=([el.bbox.as_dict()] if el.bbox else []) + lead_boxes,
                    data={
                        "columns": el.table["columns"],
                        "rows": el.table["rows"],
                        "caption": el.caption,
                    },
                )
            )
        elif el.kind == "figure":
            lead_in, lead_boxes = take_lead_in(el.page)
            caption = " — ".join(t for t in (lead_in, el.caption) if t)
            chunks.append(
                ChunkDraft(
                    chunk_type="figure",
                    content=f"Figure: {caption}".strip(),
                    section_path=sections.path,
                    page_start=el.page,
                    page_end=el.page,
                    bboxes=([el.bbox.as_dict()] if el.bbox else []) + lead_boxes,
                    data={"caption": caption, "_image_png": el.image_png},
                )
            )
    flush()

    for chunk in chunks:
        pages = (
            f"page {chunk.page_start}"
            if chunk.page_start == chunk.page_end
            else f"pages {chunk.page_start}-{chunk.page_end}"
        )
        section = chunk.section_path
        if section.casefold().startswith(title.casefold()):
            section = section[len(title) :].lstrip(" ›")
        where = f"{title} › {section}" if section else title
        chunk.context_header = f"{where} — {pages}"
        chunk.token_count = count_tokens(chunk.content)
    return chunks
