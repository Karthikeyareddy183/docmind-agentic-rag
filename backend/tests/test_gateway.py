"""LLM gateway error handling (no network: litellm is monkeypatched)."""

from typing import Any

import litellm
import pytest

from docmind.config import Settings
from docmind.llm.gateway import LLMError, LLMGateway, QuotaExhaustedError

QUOTA_BODY = (
    '{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", '
    '"details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}}'
)


def _rate_limit(message: str) -> litellm.exceptions.RateLimitError:
    return litellm.exceptions.RateLimitError(message=message, llm_provider="gemini", model="m")


@pytest.fixture
def gateway(settings: Settings) -> LLMGateway:
    return LLMGateway(settings.model_copy(update={"llm_max_retries": 2}))


async def test_daily_quota_is_not_retried(
    gateway: LLMGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    async def fake(**_: Any) -> Any:
        nonlocal calls
        calls += 1
        raise _rate_limit(QUOTA_BODY)

    monkeypatch.setattr(litellm, "acompletion", fake)
    with pytest.raises(QuotaExhaustedError):
        await gateway.text([{"role": "user", "content": "hi"}])
    assert calls == 1


async def test_transient_rate_limit_retries_then_fails(
    gateway: LLMGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    async def fake(**_: Any) -> Any:
        nonlocal calls
        calls += 1
        raise _rate_limit("429 too many requests per minute")

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(litellm, "acompletion", fake)
    monkeypatch.setattr("docmind.llm.gateway.asyncio.sleep", no_sleep)
    with pytest.raises(LLMError) as info:
        await gateway.text([{"role": "user", "content": "hi"}])
    assert not isinstance(info.value, QuotaExhaustedError)
    assert calls == 3  # 1 + llm_max_retries


async def test_non_retryable_errors_become_llm_error(
    gateway: LLMGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake(**_: Any) -> Any:
        raise ValueError("bad request shape")

    monkeypatch.setattr(litellm, "acompletion", fake)
    with pytest.raises(LLMError):
        await gateway.text([{"role": "user", "content": "hi"}])


async def test_mid_stream_failure_becomes_llm_error(
    gateway: LLMGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Chunk:
        def __init__(self, text: str) -> None:
            self.choices = [type("C", (), {"delta": type("D", (), {"content": text})()})()]

    async def broken_stream() -> Any:
        yield Chunk("Partial ")
        raise _rate_limit(QUOTA_BODY)

    async def fake(**_: Any) -> Any:
        return broken_stream()

    monkeypatch.setattr(litellm, "acompletion", fake)
    received: list[str] = []
    with pytest.raises(QuotaExhaustedError):
        async for delta in gateway.stream([{"role": "user", "content": "hi"}]):
            received.append(delta)
    assert received == ["Partial "]


async def test_in_flight_budget_402_is_retried(
    gateway: LLMGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    class Message:
        content = "pong"

    class Response:
        def __init__(self) -> None:
            self.choices = [type("C", (), {"message": Message()})()]
            self.usage = None

    async def fake(**_: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise litellm.exceptions.APIError(
                status_code=402,
                message='{"error":{"code":402,"metadata":{"reason":"in_flight_budget_exhausted"}}}',
                llm_provider="openrouter",
                model="m",
            )
        return Response()

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(litellm, "acompletion", fake)
    monkeypatch.setattr("docmind.llm.gateway.asyncio.sleep", no_sleep)
    assert await gateway.text([{"role": "user", "content": "hi"}]) == "pong"
    assert calls == 2
