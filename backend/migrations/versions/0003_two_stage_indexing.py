"""documents.embedded_chunks: track background embedding (two-stage indexing)

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _schema() -> str | None:
    translate = op.get_bind().get_execution_options().get("schema_translate_map") or {}
    return translate.get(None)


def upgrade() -> None:
    schema = _schema()
    op.add_column(
        "documents",
        sa.Column("embedded_chunks", sa.Integer(), nullable=False, server_default="0"),
        schema=schema,
    )
    qualified = f'"{schema}"."documents"' if schema else '"documents"'
    # Documents indexed before this migration were fully embedded in one pass
    op.execute(f"UPDATE {qualified} SET embedded_chunks = chunk_count WHERE status = 'ready'")


def downgrade() -> None:
    op.drop_column("documents", "embedded_chunks", schema=_schema())
