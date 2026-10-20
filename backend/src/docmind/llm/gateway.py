"""Provider-agnostic LLM access via LiteLLM.

One place for: retries with exponential backoff on rate limits / transient errors, vision inputs,
Pydantic-validated structured outputs, streaming, native tool calling, and token/cost accounting.
Swap providers by changing a model string (``gemini/...``, ``openai/...``, ``anthropic/...``).
"""

import asyncio
import base64
import json
import logging
import random
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

import litellm
from pydantic import BaseModel, ValidationError

from docmind.config import Settings

logger = logging.getLogger(__name__)

litellm.drop_params = True  # ignore params a provider does not support instead of failing
litellm.suppress_debug_info = True

T = TypeVar("T", bound=BaseModel)

_RETRYABLE = (
    litellm.exceptions.RateLimitError,
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.InternalServerError,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.Timeout,
    litellm.exceptions.BadGatewayError,
)


class LLMError(RuntimeError):
    """The model could not produce a usable response."""


class QuotaExhaustedError(LLMError):
    """The provider's quota is used up (e.g. a free-tier daily limit); retrying won't help."""


def _is_transient_budget(exc: Exception) -> bool:
    """OpenRouter 402 when *in-flight* requests have reserved the available credit; it clears
    as soon as earlier requests finish, so it is worth retrying (unlike an empty balance)."""
    return "in_flight_budget_exhausted" in str(exc)


def _is_quota_exhausted(exc: Exception) -> bool:
    text = str(exc)
    return (
        "RESOURCE_EXHAUSTED" in text and ("PerDay" in text or "insufficient_quota" in text)
    ) or ("insufficient_quota" in text)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    by_model: dict[str, int] = field(default_factory=dict)

    def add(self, model: str, response: Any) -> None:
        usage = getattr(response, "usage", None)
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1
        self.by_model[model] = self.by_model.get(model, 0) + prompt + completion
        cost = 0.0
        try:
            cost = float(litellm.completion_cost(completion_response=response) or 0.0)
        except Exception:  # unknown pricing for a model is not an error
            logger.debug("No LiteLLM pricing for %s", model)
        if not cost:
            price = _openrouter_price(model)
            if price:
                cost = prompt * price[0] + completion * price[1]
        self.cost_usd += cost

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "calls": self.calls,
            "by_model": self.by_model,
        }


_OPENROUTER_PRICES: dict[str, tuple[float, float]] | None = None


def _openrouter_price(model: str) -> tuple[float, float] | None:
    """Per-token (prompt, completion) USD price for ``openrouter/...`` models.

    LiteLLM's built-in price map lags behind new model releases; OpenRouter publishes current
    prices, fetched once per process.
    """
    global _OPENROUTER_PRICES
    if not model.startswith("openrouter/"):
        return None
    if _OPENROUTER_PRICES is None:
        _OPENROUTER_PRICES = {}
        try:
            import httpx

            data = httpx.get("https://openrouter.ai/api/v1/models", timeout=15).json()["data"]
            _OPENROUTER_PRICES = {
                m["id"]: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"]))
                for m in data
                if m.get("pricing")
            }
        except Exception:
            logger.warning("Could not load OpenRouter prices; cost will read as 0")
    return _OPENROUTER_PRICES.get(model.removeprefix("openrouter/"))


def image_part(png_bytes: bytes, mime: str = "image/png") -> dict[str, Any]:
    """An OpenAI-style image content part (LiteLLM translates it for each provider)."""
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
    return text.strip()


class LLMGateway:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def _call(self, **kwargs: Any) -> Any:
        """``litellm.acompletion`` with jittered exponential backoff on retryable errors."""
        # Gemini 3+ deprecates sampling params; only send what the caller set explicitly
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs.setdefault("max_tokens", self.settings.llm_max_output_tokens)
        attempts = self.settings.llm_max_retries + 1
        for attempt in range(attempts):
            try:
                return await litellm.acompletion(
                    timeout=self.settings.llm_timeout_seconds, **kwargs
                )
            except Exception as exc:
                transient = isinstance(exc, _RETRYABLE) or _is_transient_budget(exc)
                if not transient:  # auth, bad request, unknown model, ...: not retryable
                    raise LLMError(
                        f"{kwargs.get('model')}: {type(exc).__name__}: {str(exc)[:300]}"
                    ) from exc
                if _is_quota_exhausted(exc):
                    raise QuotaExhaustedError(
                        f"{kwargs.get('model')}: provider quota exhausted"
                    ) from exc
                if attempt == attempts - 1:
                    raise LLMError(f"{kwargs.get('model')}: {type(exc).__name__}") from exc
                delay = min(30.0, 2.0 * 2**attempt) + random.uniform(0, 1)  # noqa: S311
                logger.warning(
                    "LLM call retry",
                    extra={"model": kwargs.get("model"), "attempt": attempt + 1, "delay_s": delay},
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def complete(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str | None = None,
        usage: Usage | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> Any:
        """Return the raw message (``.content`` and ``.tool_calls``).

        ``tool_choice``: "auto" | "required" | "none" (only sent together with ``tools``).
        """
        model = model or self.settings.llm_model
        response = await self._call(
            model=model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice if tools else None,
        )
        if usage is not None:
            usage.add(model, response)
        return response.choices[0].message

    async def text(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> str:
        message = await self.complete(messages, **kwargs)
        return (message.content or "").strip()

    async def structured(
        self,
        messages: Sequence[dict[str, Any]],
        schema: type[T],
        *,
        model: str | None = None,
        usage: Usage | None = None,
        temperature: float | None = None,
        max_tokens: int = 1024,
    ) -> T:
        """JSON output validated against ``schema``; one repair attempt on invalid output.

        Structured replies are short, so a small ``max_tokens`` keeps each request's reserved
        cost (and worst-case latency) low.
        """
        model = model or self.settings.llm_model
        convo = list(messages)
        for attempt in range(2):
            response = await self._call(
                model=model,
                messages=convo,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
                },
            )
            if usage is not None:
                usage.add(model, response)
            raw = response.choices[0].message.content or ""
            try:
                return schema.model_validate(json.loads(_strip_json_fence(raw)))
            except (json.JSONDecodeError, ValidationError) as exc:
                if attempt == 1:
                    raise LLMError(f"Invalid structured output from {model}: {exc}") from exc
                convo = [
                    *convo,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": f"That was not valid JSON for the schema ({exc}). "
                        "Reply with only the corrected JSON object.",
                    },
                ]
        raise AssertionError("unreachable")

    async def stream(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str | None = None,
        usage: Usage | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Yield text deltas. Usage is recorded from the final chunk when the provider sends it."""
        model = model or self.settings.llm_model
        response = await self._call(
            model=model,
            messages=list(messages),
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks: list[Any] = []
        try:
            async for chunk in response:
                chunks.append(chunk)
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception as exc:  # provider failed mid-stream (rate limit, disconnect, ...)
            if _is_quota_exhausted(exc):
                raise QuotaExhaustedError(f"{model}: provider quota exhausted") from exc
            raise LLMError(f"{model}: stream failed: {type(exc).__name__}") from exc
        if usage is not None and chunks:
            try:
                usage.add(model, litellm.stream_chunk_builder(chunks, messages=list(messages)))
            except Exception:
                logger.debug("Could not compute streaming usage", exc_info=True)
