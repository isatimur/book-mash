"""Retry-with-backoff for judge model calls.

Rate limits (HTTP 429) and transient overload (Anthropic's 529, plus 5xx) were
silently turning into permanent coverage gaps: each judge's `judge()` catches
the exception and returns a `None`/ERROR score, so a throttled paragraph never
gets re-tried. The two judges with the largest prompts (humanness bundles the
surrounding paragraphs; claim_defensibility bundles the relevant ledger) blow
the tokens-per-minute budget first, which is why prior runs left exactly those
two dimensions partial while small-prompt usefulness stayed fully covered.

This wraps the model call so transient throttling recovers instead of gapping.
Backoff honors a `Retry-After` header when the API sends one, otherwise uses
exponential backoff with full jitter. `sleep`/`rng` are injectable so the
behavior is unit-testable without real waits.
"""

import asyncio
import random
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")

# 429 = rate limit, 529 = Anthropic "overloaded", 500/502/503 = transient server.
_RETRYABLE_STATUS = {429, 500, 502, 503, 529}

# Class-name fragments for SDK errors that don't expose a status code.
_RETRYABLE_NAME_FRAGMENTS = (
    "ratelimit",
    "overload",
    "apiconnection",
    "apitimeout",
    "internalserver",
    "serviceunavailable",
)


def _status_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value
    return None


def _is_retryable(exc: BaseException) -> bool:
    code = _status_code(exc)
    if code is not None:
        return code in _RETRYABLE_STATUS
    name = type(exc).__name__.lower()
    return any(fragment in name for fragment in _RETRYABLE_NAME_FRAGMENTS)


def _retry_after(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None  # HTTP-date form is rare here; fall back to computed backoff


async def run_with_backoff(
    factory: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 6,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: Callable[[], float] = random.random,
) -> T:
    """Call ``factory()`` with retry on transient throttling/overload.

    Non-retryable errors (4xx other than 429, schema/validation bugs) propagate
    immediately. Retryable ones back off until ``max_attempts`` is reached, then
    the last exception is re-raised so the caller's existing error handling still
    records a clean ERROR score.
    """
    attempt = 0
    while True:
        try:
            return await factory()
        except Exception as exc:
            attempt += 1
            if attempt >= max_attempts or not _is_retryable(exc):
                raise
            delay = _retry_after(exc)
            if delay is None:
                capped = min(max_delay, base_delay * (2 ** (attempt - 1)))
                # Full jitter keeps concurrent workers from retrying in lockstep.
                delay = capped + rng() * capped
            await sleep(delay)
