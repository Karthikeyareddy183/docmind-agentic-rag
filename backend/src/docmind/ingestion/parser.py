"""Layout-aware parsing with Docling (CPU): text with headings, tables, figures, OCR for scans.

The rest of the pipeline only sees :class:`ParsedDocument`, so the parser can be swapped
(e.g. for a hosted parser) without touching chunking or indexing.
"""

import io
import logging
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

ElementKind = Literal["heading", "text", "table", "figure"]

_SKIP_LABELS = {"page_header", "page_footer", "document_index", "marker", "empty_value"}


@dataclass(frozen=True)
class BBox:
    """Region on a page as fractions of width/height, origin top-left (what the viewer needs)."""

    page: int
    x0: float
    y0: float
    x1: float
    y1: float

    def as_dict(self) -> dict[str, float | int]:
        return {"page": self.page, "x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass
class Element:
    kind: ElementKind
    text: str
    page: int
    bbox: BBox | None = None
    level: int = 1
    caption: str = ""
    table: dict[str, Any] | None = None  # {"columns": [...], "rows": [[...]], "markdown": str}
    image_png: bytes | None = None


@dataclass
class ParsedDocument:
    pages: int
    elements: list[Element] = field(default_factory=list)
    title: str | None = None

    @property
    def text_chars(self) -> int:
        return sum(len(e.text) for e in self.elements)


@lru_cache(maxsize=1)
def _converter() -> Any:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import (
        DocumentConverter,
        ImageFormatOption,
        PdfFormatOption,
    )

    options = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=RapidOcrOptions(),  # pip-installable ONNX OCR; no system Tesseract needed
        do_table_structure=True,
        generate_picture_images=True,
        images_scale=2.0,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=options),
        }
    )


# Docling's converter holds model state; serialise use within a process
_convert_lock = threading.Lock()


def _bbox(item: Any, doc: Any) -> BBox | None:
    prov = item.prov[0] if getattr(item, "prov", None) else None
    if prov is None:
        return None
    page = doc.pages.get(prov.page_no)
    if page is None or not page.size.width or not page.size.height:
        return None
    w, h = page.size.width, page.size.height
    box = prov.bbox.to_top_left_origin(page_height=h)

    def clamp(v: float) -> float:
        return round(min(max(v, 0.0), 1.0), 4)

    return BBox(
        page=prov.page_no,
        x0=clamp(box.l / w),
        y0=clamp(box.t / h),
        x1=clamp(box.r / w),
        y1=clamp(box.b / h),
    )


def _table_payload(item: Any, doc: Any) -> dict[str, Any]:
    df = item.export_to_dataframe(doc=doc)
    columns = [str(c).strip() for c in df.columns]
    rows = [
        [("" if v is None else str(v)).strip() for v in row] for row in df.itertuples(index=False)
    ]
    return {"columns": columns, "rows": rows, "markdown": item.export_to_markdown(doc=doc)}


def parse_file(path: Path, max_pages: int = 200) -> ParsedDocument:
    """Parse a PDF or image file. Blocking and CPU-heavy: call from a worker thread."""
    from docling_core.types.doc import (
        PictureItem,
        SectionHeaderItem,
        TableItem,
        TextItem,
        TitleItem,
    )

    with _convert_lock:
        result = _converter().convert(path, max_num_pages=max_pages)
    doc = result.document

    parsed = ParsedDocument(pages=doc.num_pages() or 1)
    for item, _depth in doc.iterate_items():
        label = str(getattr(item, "label", "") or "")
        if label in _SKIP_LABELS:
            continue
        bbox = _bbox(item, doc)
        page = bbox.page if bbox else 1

        if isinstance(item, TitleItem | SectionHeaderItem):
            text = item.text.strip()
            if not text:
                continue
            level = 0 if isinstance(item, TitleItem) else int(getattr(item, "level", 1) or 1)
            parsed.elements.append(Element("heading", text, page, bbox, level=level))
            if parsed.title is None:
                parsed.title = text
        elif isinstance(item, TableItem):
            try:
                table = _table_payload(item, doc)
            except Exception:
                logger.warning("Table export failed", exc_info=True)
                continue
            if not table["rows"]:
                continue
            caption = item.caption_text(doc) if hasattr(item, "caption_text") else ""
            parsed.elements.append(
                Element("table", table["markdown"], page, bbox, caption=caption, table=table)
            )
        elif isinstance(item, PictureItem):
            image = item.get_image(doc)
            if image is None or image.width < 80 or image.height < 80:
                continue  # icons, rules, logos
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            parsed.elements.append(
                Element(
                    "figure",
                    "",
                    page,
                    bbox,
                    caption=item.caption_text(doc),
                    image_png=buf.getvalue(),
                )
            )
        elif isinstance(item, TextItem):
            text = item.text.strip()
            if text and label != "caption":  # captions are attached to their table/figure
                parsed.elements.append(Element("text", text, page, bbox))

    return parsed


# ─── Fast path: the PDF's own text layer (+ OCR for scanned pages) ─────────


@lru_cache(maxsize=1)
def _ocr_engine() -> Any:
    from rapidocr import RapidOCR

    return RapidOCR()


def _ocr_page(page: Any) -> str:
    """OCR one page rendered at ~200 dpi (used only for pages without a text layer)."""
    import numpy as np

    image = page.render(scale=200 / 72).to_pil().convert("RGB")
    result = _ocr_engine()(np.asarray(image))
    texts = getattr(result, "txts", None) or ()
    return "\n".join(t for t in texts if t)


def parse_pdf_fast(path: Path, max_pages: int = 2000, ocr_min_chars: int = 50) -> ParsedDocument:
    """Text-layer extraction with pdfium: ~0.01 s/page instead of ~3 s/page for layout models.

    Trades table structure, figure crops, and heading detection for speed; used for long
    PDFs, and as the fallback when the layout parser fails.
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        pages = min(len(pdf), max_pages)
        parsed = ParsedDocument(pages=pages)
        ocr_pages = 0
        for index in range(pages):
            page = pdf[index]
            text = page.get_textpage().get_text_range().strip()
            if len(text) < ocr_min_chars:
                try:
                    text = _ocr_page(page).strip() or text
                    ocr_pages += 1
                except Exception:
                    logger.warning("OCR failed", extra={"page": index + 1}, exc_info=True)
            if not text:
                continue
            # Paragraph-sized elements keep chunk boundaries natural
            for block in (b.strip() for b in text.replace("\r", "").split("\n\n")):
                if block:
                    parsed.elements.append(Element("text", " ".join(block.split()), index + 1))
        logger.info("Fast parse complete", extra={"pages": pages, "ocr_pages": ocr_pages})
        return parsed
    finally:
        pdf.close()


def pdf_page_count(path: Path) -> int | None:
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        try:
            return len(pdf)
        finally:
            pdf.close()
    except Exception:
        return None


# ─── Layout path in an isolated process ────────────────────────────────────
# Docling's native PDF parser can crash the interpreter on some files (seen: heap corruption,
# 0xc0000374 on Windows). Running it in a child process turns that into a recoverable error.

_pool_lock = threading.Lock()
_pool: Any = None


def _layout_pool() -> Any:
    global _pool
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with _pool_lock:
        if _pool is None:
            _pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
        return _pool


def _reset_layout_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None


def _parse_in_child(path: str, max_pages: int, env: dict[str, str]) -> ParsedDocument:
    import os

    os.environ.update(env)  # HF_HOME etc. for model downloads in the child
    return parse_file(Path(path), max_pages)


def parse_layout_isolated(path: Path, max_pages: int, timeout_s: float = 3600) -> ParsedDocument:
    import os
    from concurrent.futures.process import BrokenProcessPool

    env = {k: v for k, v in os.environ.items() if k.startswith(("HF_", "TMP", "TEMP", "OMP_"))}
    future = _layout_pool().submit(_parse_in_child, str(path), max_pages, env)
    try:
        parsed: ParsedDocument = future.result(timeout=timeout_s)
        return parsed
    except BrokenProcessPool:
        _reset_layout_pool()  # the child died; next document gets a fresh one
        raise


def parse_document(
    path: Path,
    *,
    content_type: str,
    max_pages: int,
    fast_parse_min_pages: int,
    ocr_min_chars: int,
) -> tuple[ParsedDocument, str]:
    """Pick the parsing strategy. Returns ``(parsed, strategy)``; blocking — call in a thread."""
    if content_type == "application/pdf":
        pages = pdf_page_count(path)
        if pages is not None and pages > fast_parse_min_pages:
            return parse_pdf_fast(path, max_pages, ocr_min_chars), "fast"
    try:
        return parse_layout_isolated(path, max_pages), "layout"
    except Exception:
        if content_type != "application/pdf":
            raise  # images have no text layer to fall back on
        logger.warning("Layout parsing failed; falling back to the text layer", exc_info=True)
        return parse_pdf_fast(path, max_pages, ocr_min_chars), "fast-fallback"
