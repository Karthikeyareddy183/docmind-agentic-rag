import re
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from docmind.api.deps import SessionDep, SettingsDep

router = APIRouter(prefix="/health", tags=["health"])

ALEMBIC_INI = Path(__file__).resolve().parents[4] / "alembic.ini"


@lru_cache(maxsize=1)
def expected_revision() -> str | None:
    """The newest migration shipped with this build (None if migrations aren't packaged)."""
    if not ALEMBIC_INI.exists():
        return None
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    return ScriptDirectory.from_config(Config(str(ALEMBIC_INI))).get_current_head()


@router.get("/live")
async def live() -> dict[str, str]:
    """Process is up (no dependency checks) — for container liveness probes."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(session: SessionDep, settings: SettingsDep) -> JSONResponse:
    """Dependencies reachable and the schema migrated — for load balancer readiness checks."""
    try:
        await session.execute(text("SELECT 1"))
    except Exception:
        return JSONResponse(status_code=503, content={"status": "unavailable", "database": "down"})

    expected = expected_revision()
    schema = settings.db_schema
    current = None
    # Server-side config, but validate anyway: only a plain identifier is interpolated
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema):
        try:
            current = await session.scalar(
                text(f'SELECT version_num FROM "{schema}".alembic_version')  # noqa: S608
            )
        except Exception:
            current = None
    if expected and current != expected:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "database": "up",
                "migrations": "pending",
                "current": current,
                "expected": expected,
            },
        )
    return JSONResponse(content={"status": "ok", "database": "up", "migrations": current})
