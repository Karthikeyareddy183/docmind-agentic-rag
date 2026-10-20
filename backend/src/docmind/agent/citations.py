"""Citation parsing/validation and the computed (not self-reported) confidence score."""

import re
from typing import Any

from pydantic import BaseModel, Field

# Any bracket group that mentions a source id: "[S1]", "[S1, S3]", and the decorated forms
# models also produce, e.g. "[S1, page 3]" or "[S2, Section 6, p. 3]", and the full-width
# lenticular brackets some models (gpt-oss) use: "【S1】"
_CITE = re.compile(r"[\[【]([^\[\]【】]*?\bS\d+\b[^\[\]【】]*?)[\]】]")
_SOURCE_ID = re.compile(r"\bS\d+\b")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


class Citation(BaseModel):
    id: str
    chunk_id: str
    document_id: str
    filename: str
    document_title: str
    chunk_type: str
    page: int
    page_end: int
    bboxes: list[dict[str, Any]] = Field(default_factory=list)
    snippet: str
    score: float


def cited_ids(answer: str) -> list[str]:
    seen: list[str] = []
    for group in _CITE.findall(answer):
        for sid in _SOURCE_ID.findall(group):
            if sid not in seen:
                seen.append(sid)
    return seen


def strip_invalid_citations(answer: str, valid: set[str]) -> str:
    """Drop citation markers that point at sources the model was never given."""

    def keep(match: re.Match[str]) -> str:
        # Normalise to bare ids ("[S1, page 3]" -> "[S1]"): the UI renders the page itself
        ids = list(dict.fromkeys(s for s in _SOURCE_ID.findall(match.group(1)) if s in valid))
        return f"[{', '.join(ids)}]" if ids else ""

    return _CITE.sub(keep, answer)


def citation_coverage(answer: str) -> float:
    """Fraction of substantive sentences that carry at least one citation."""
    sentences = [s for s in _SENTENCE.split(answer.strip()) if len(s.strip()) > 25]
    if not sentences:
        return 1.0 if cited_ids(answer) else 0.0
    return sum(1 for s in sentences if _CITE.search(s)) / len(sentences)


def compute_confidence(
    *, top_relevance: float, coverage: float, verified: bool | None, used_calculation: bool
) -> float:
    """Blend retrieval strength, citation coverage, and the fact-check verdict.

    The model never rates itself: verbalized LLM confidence is poorly calibrated, and worse
    when retrieved context is noisy.
    """
    verification = 0.5 if verified is None else (1.0 if verified else 0.1)
    retrieval = max(top_relevance, 0.85 if used_calculation else 0.0)
    score = 0.45 * retrieval + 0.20 * coverage + 0.35 * verification
    return round(min(max(score, 0.0), 1.0), 3)
