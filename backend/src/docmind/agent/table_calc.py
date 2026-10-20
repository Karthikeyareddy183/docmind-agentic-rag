"""Deterministic arithmetic over extracted tables.

The LLM only *plans* the calculation (operation + cell references by row/column label); the
numbers are looked up and computed here, so answers like "48.4%" are exact and auditable.
No code or expressions from the model are ever evaluated.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field

Operation = Literal[
    "value", "sum", "mean", "difference", "ratio", "percent_of", "percent_change", "max", "min"
]

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


class CellRef(BaseModel):
    row: str = Field(description="Row label exactly as in the first column, e.g. 'Cloud Platform'")
    column: str = Field(description="Column header exactly as shown, e.g. 'FY2025'")


class CalcPlan(BaseModel):
    table_source: str = Field(description="Source id of the table to use, e.g. 'S3'")
    operation: Operation = Field(
        description="value: one cell. sum/mean/max/min: over all operands. difference: a-b. "
        "ratio: a/b. percent_of: a/b*100. percent_change: (b-a)/a*100."
    )
    operands: list[CellRef] = Field(min_length=1, max_length=12)
    explanation: str = Field(description="One short sentence describing what is computed")


class CalcResult(BaseModel):
    ok: bool
    value: float | None = None
    formatted: str = ""
    expression: str = ""
    cells: list[dict[str, str]] = Field(default_factory=list)
    error: str = ""


def parse_number(text: str) -> float | None:
    cleaned = text.replace(",", "").replace("$", "").replace("₹", "").replace("%", "").strip()
    match = _NUMBER.search(cleaned)
    return float(match.group()) if match else None


def _norm(label: str) -> str:
    """Compare labels ignoring case, spaces, and punctuation ("FY 2025" == "FY2025")."""
    return re.sub(r"[^a-z0-9]+", "", label.lower())


def _find(labels: list[str], wanted: str) -> int | None:
    target = _norm(wanted)
    normalized = [_norm(label) for label in labels]
    if target in normalized:
        return normalized.index(target)
    for i, label in enumerate(normalized):  # tolerant fallback: unique containment match
        if target and (target in label or label in target):
            matches = [
                j for j, other in enumerate(normalized) if target in other or other in target
            ]
            if len(matches) == 1:
                return i
    return None


def lookup(columns: list[str], rows: list[list[str]], ref: CellRef) -> tuple[float, str]:
    col = _find(columns, ref.column)
    if col is None:
        raise ValueError(f"Column {ref.column!r} not found (columns: {columns})")
    row_idx = _find([r[0] if r else "" for r in rows], ref.row)
    if row_idx is None:
        raise ValueError(f"Row {ref.row!r} not found (rows: {[r[0] for r in rows if r]})")
    raw = rows[row_idx][col] if col < len(rows[row_idx]) else ""
    value = parse_number(raw)
    if value is None:
        raise ValueError(f"Cell ({ref.row}, {ref.column}) is not numeric: {raw!r}")
    return value, raw


def _fmt(value: float) -> str:
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def compute(plan: CalcPlan, columns: list[str], rows: list[list[str]]) -> CalcResult:
    try:
        looked = [lookup(columns, rows, ref) for ref in plan.operands]
    except ValueError as exc:
        return CalcResult(ok=False, error=str(exc))
    values = [v for v, _ in looked]
    cells = [
        {"row": ref.row, "column": ref.column, "value": raw}
        for ref, (_, raw) in zip(plan.operands, looked, strict=True)
    ]
    op = plan.operation
    try:
        if op == "value":
            result, expr = values[0], _fmt(values[0])
        elif op == "sum":
            result, expr = sum(values), " + ".join(map(_fmt, values))
        elif op == "mean":
            result = sum(values) / len(values)
            expr = f"({' + '.join(map(_fmt, values))}) / {len(values)}"
        elif op == "max":
            result, expr = max(values), f"max({', '.join(map(_fmt, values))})"
        elif op == "min":
            result, expr = min(values), f"min({', '.join(map(_fmt, values))})"
        elif len(values) != 2:
            return CalcResult(ok=False, error=f"{op} needs exactly 2 operands", cells=cells)
        else:
            a, b = values
            if op == "difference":
                result, expr = a - b, f"{_fmt(a)} - {_fmt(b)}"
            elif op == "ratio":
                result, expr = a / b, f"{_fmt(a)} / {_fmt(b)}"
            elif op == "percent_of":
                result, expr = a / b * 100, f"{_fmt(a)} / {_fmt(b)} × 100"
            else:  # percent_change
                result, expr = (b - a) / a * 100, f"({_fmt(b)} - {_fmt(a)}) / {_fmt(a)} × 100"
    except ZeroDivisionError:
        return CalcResult(ok=False, error="Division by zero", cells=cells)

    suffix = "%" if op in ("percent_of", "percent_change") else ""
    # Ratios are small numbers: keep 4 decimals so 410/2340 reads 0.1752, not 0.18
    digits = 4 if op == "ratio" else 2
    formatted = f"{round(result, digits):,}{suffix}"
    return CalcResult(
        ok=True, value=result, formatted=formatted, expression=f"{expr} = {formatted}", cells=cells
    )
