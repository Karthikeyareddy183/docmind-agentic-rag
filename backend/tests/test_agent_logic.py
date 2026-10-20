"""Pure-logic tests: table arithmetic, citations, confidence, chunking, rank fusion."""

import uuid

import pytest

from docmind.agent.citations import (
    citation_coverage,
    cited_ids,
    compute_confidence,
    strip_invalid_citations,
)
from docmind.agent.table_calc import CalcPlan, CellRef, compute, parse_number
from docmind.ingestion.chunker import chunk_document
from docmind.ingestion.parser import BBox, Element, ParsedDocument
from docmind.retrieval.search import rrf_fuse

COLUMNS = ["Segment", "Q1", "Q2", "Q3", "Q4", "FY2025"]
ROWS = [
    ["Cloud Platform", "52.4", "56.1", "60.3", "64.8", "233.6"],
    ["Data Services", "31.2", "33.0", "34.7", "36.5", "135.4"],
    ["Consulting", "27.0", "28.1", "28.9", "29.6", "113.6"],
    ["Total", "110.6", "117.2", "123.9", "130.9", "482.6"],
]


def plan(op: str, *refs: tuple[str, str]) -> CalcPlan:
    return CalcPlan(
        table_source="S1",
        operation=op,  # type: ignore[arg-type]
        operands=[CellRef(row=r, column=c) for r, c in refs],
        explanation="test",
    )


class TestTableCalc:
    def test_percent_of(self) -> None:
        result = compute(
            plan("percent_of", ("Cloud Platform", "FY2025"), ("Total", "FY2025")), COLUMNS, ROWS
        )
        assert result.ok
        assert result.formatted == "48.4%"
        assert "233.6 / 482.6" in result.expression

    def test_difference_and_percent_change(self) -> None:
        diff = compute(
            plan("difference", ("Consulting", "Q4"), ("Consulting", "Q1")), COLUMNS, ROWS
        )
        assert diff.value == pytest.approx(2.6)
        change = compute(
            plan("percent_change", ("Consulting", "Q1"), ("Consulting", "Q4")), COLUMNS, ROWS
        )
        assert change.formatted == "9.63%"

    def test_ratio_keeps_precision(self) -> None:
        rows = [["Sales & Marketing", "410"], ["Total", "2,340"]]
        result = compute(
            plan("ratio", ("Sales & Marketing", "Employees"), ("Total", "Employees")),
            ["Function", "Employees"],
            rows,
        )
        assert result.formatted == "0.1752"

    def test_mean_over_quarters(self) -> None:
        refs = [("Data Services", q) for q in ("Q1", "Q2", "Q3", "Q4")]
        assert compute(plan("mean", *refs), COLUMNS, ROWS).formatted == "33.85"

    def test_labels_matched_loosely(self) -> None:
        result = compute(plan("value", ("cloud platform", "fy 2025")), COLUMNS, ROWS)
        assert result.ok
        assert result.value == pytest.approx(233.6)

    def test_unknown_row_reports_error(self) -> None:
        result = compute(plan("value", ("Hardware", "Q1")), COLUMNS, ROWS)
        assert not result.ok
        assert "Hardware" in result.error

    def test_binary_op_needs_two_operands(self) -> None:
        assert not compute(plan("ratio", ("Total", "Q1")), COLUMNS, ROWS).ok

    def test_division_by_zero(self) -> None:
        rows = [["A", "5"], ["B", "0"]]
        assert (
            compute(plan("ratio", ("A", "x"), ("B", "x")), ["Name", "x"], rows).error
            == "Division by zero"
        )

    @pytest.mark.parametrize(
        ("raw", "value"),
        [
            ("1,120", 1120.0),
            ("$482.6", 482.6),
            ("₹4.50", 4.5),
            ("19.4%", 19.4),
            ("-3", -3.0),
            ("n/a", None),
        ],
    )
    def test_parse_number(self, raw: str, value: float | None) -> None:
        assert parse_number(raw) == value


class TestCitations:
    def test_cited_ids_in_order_without_duplicates(self) -> None:
        assert cited_ids("A [S2]. B [S1, S2]. C [S3]") == ["S2", "S1", "S3"]

    @pytest.mark.parametrize(
        ("text", "ids"),
        [
            ("Revenue grew [S1, page 3].", ["S1"]),
            ("Leave is 24 days [S2, Section 3. Leave Policy, page 2].", ["S2"]),
            ("Both [S1, p. 3; S4, p. 7] agree.", ["S1", "S4"]),
            ("Cloud was 48.4 %【S1】 and Data【S2, S3】.", ["S1", "S2", "S3"]),
            ("Not a citation [Table 2] or [see S] here.", []),
        ],
    )
    def test_cited_ids_accepts_decorated_forms(self, text: str, ids: list[str]) -> None:
        assert cited_ids(text) == ids

    def test_strip_normalises_decorated_citations(self) -> None:
        text = "Budget is Rs 40,000 [S1, Section 6, page 3]. Margin [S9, page 4]."
        assert strip_invalid_citations(text, {"S1"}) == "Budget is Rs 40,000 [S1]. Margin ."

    def test_strip_invalid(self) -> None:
        text = "Revenue was $482.6M [S1, S9]. Margin rose [S7]."
        assert strip_invalid_citations(text, {"S1"}) == "Revenue was $482.6M [S1]. Margin rose ."

    def test_coverage(self) -> None:
        full = "Revenue was $482.6 million in FY2025 [S1]. The CEO is Meera Iyer according to the letter [S2]."
        half = "Revenue was $482.6 million in FY2025 [S1]. The company also has many offices around the world."
        assert citation_coverage(full) == 1.0
        assert citation_coverage(half) == 0.5

    def test_confidence_orders_sensibly(self) -> None:
        strong = compute_confidence(
            top_relevance=0.95, coverage=1.0, verified=True, used_calculation=False
        )
        unverified = compute_confidence(
            top_relevance=0.95, coverage=1.0, verified=False, used_calculation=False
        )
        weak = compute_confidence(
            top_relevance=0.1, coverage=0.3, verified=None, used_calculation=False
        )
        assert strong > unverified > weak
        assert 0.0 <= weak <= strong <= 1.0


def _el(kind: str, text: str, page: int = 1, level: int = 1, table: dict | None = None) -> Element:
    return Element(kind, text, page, BBox(page, 0.1, 0.1, 0.9, 0.2), level=level, table=table)  # type: ignore[arg-type]


class TestChunker:
    def test_sections_tables_and_headers(self) -> None:
        table = {
            "columns": ["A", "B"],
            "rows": [["x", "1"]],
            "markdown": "| A | B |\n|---|---|\n| x | 1 |",
        }
        parsed = ParsedDocument(
            pages=2,
            elements=[
                _el("heading", "Report", level=0),
                _el("heading", "Leave Policy"),
                _el("text", "Employees receive 24 days of annual leave."),
                _el("heading", "Revenue"),
                _el("text", "Quarterly revenue table:", page=2),
                _el("table", table["markdown"], page=2, table=table),
            ],
        )
        chunks = chunk_document(parsed, "Annual Report", max_tokens=200)
        assert [c.chunk_type for c in chunks] == ["text", "table"]
        leave, tbl = chunks
        assert leave.section_path == "Leave Policy"
        assert leave.context_header == "Annual Report › Leave Policy — page 1"
        # Short lead-in text is folded into the table instead of becoming an orphan chunk
        assert "Quarterly revenue table:" in tbl.content
        assert tbl.data["rows"] == [["x", "1"]]
        assert tbl.page_start == 2

    def test_never_crosses_headings_and_respects_budget(self) -> None:
        sentence = "This sentence about policy details is reasonably long and repeats. "
        parsed = ParsedDocument(
            pages=1,
            elements=[
                _el("heading", "A"),
                _el("text", sentence * 40),
                _el("heading", "B"),
                _el("text", "Short."),
            ],
        )
        chunks = chunk_document(parsed, "Doc", max_tokens=120)
        assert len(chunks) >= 3
        assert all(c.token_count <= 130 for c in chunks)
        assert chunks[-1].section_path == "B" and chunks[-1].content == "Short."


def test_rrf_rewards_agreement() -> None:
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    scores = rrf_fuse([[a, b, c], [b, a]], k=60)
    assert set(scores) == {a, b, c}
    assert scores[a] == pytest.approx(scores[b])  # rank 1+2 in both lists
    assert scores[c] < scores[a]


def test_retry_hint_parsing() -> None:
    from docmind.ingestion.pipeline import _retry_after_seconds

    body = '{"error": {"details": [{"@type": "RetryInfo", "retryDelay": "5191s"}]}}'
    assert _retry_after_seconds(Exception(body)) == 5191.0
    assert _retry_after_seconds(Exception("429 too many requests")) is None


def test_quota_errors_recognised_regardless_of_exception_type() -> None:
    from docmind.ingestion.pipeline import _is_rate_limited

    assert _is_rate_limited(ValueError('{"status": "RESOURCE_EXHAUSTED"}'))
    assert _is_rate_limited(RuntimeError("You exceeded your current quota, please check"))
    assert not _is_rate_limited(ValueError("Invalid input: text too long"))
