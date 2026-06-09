import pytest

from book_mash.judges._retry import (
    run_with_backoff,
    _is_retryable,
    _retry_after,
)


class _FakeHeaders(dict):
    """Case-insensitive-ish header bag; anthropic uses httpx.Headers which is."""


class _FakeResponse:
    def __init__(self, status_code=None, headers=None):
        self.status_code = status_code
        self.headers = _FakeHeaders(headers or {})


class _StatusError(Exception):
    """Mimics anthropic.APIStatusError shape (status_code + response)."""

    def __init__(self, status_code, headers=None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.response = _FakeResponse(status_code, headers)


class RateLimitError(Exception):
    """Mimics an SDK error detectable purely by class name (no status_code)."""


def _recorder():
    calls = []

    async def sleep(delay):
        calls.append(delay)

    return calls, sleep


# ---- detection ----

def test_is_retryable_by_status_code():
    assert _is_retryable(_StatusError(429))
    assert _is_retryable(_StatusError(529))  # Anthropic "overloaded"
    assert _is_retryable(_StatusError(503))


def test_is_retryable_by_class_name():
    assert _is_retryable(RateLimitError("slow down"))


def test_not_retryable_for_client_errors_and_bugs():
    assert not _is_retryable(_StatusError(400))
    assert not _is_retryable(_StatusError(404))
    assert not _is_retryable(ValueError("schema mismatch"))


def test_retry_after_parsed_from_header():
    assert _retry_after(_StatusError(429, {"retry-after": "12"})) == 12.0
    assert _retry_after(_StatusError(429, {})) is None
    assert _retry_after(ValueError("x")) is None


# ---- behavior ----

async def test_succeeds_first_try_no_sleep():
    calls, sleep = _recorder()

    async def factory():
        return "ok"

    out = await run_with_backoff(factory, sleep=sleep, rng=lambda: 0.0)
    assert out == "ok"
    assert calls == []


async def test_retries_then_succeeds():
    calls, sleep = _recorder()
    attempts = {"n": 0}

    async def factory():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _StatusError(429)
        return "recovered"

    out = await run_with_backoff(
        factory, base_delay=2.0, sleep=sleep, rng=lambda: 0.0
    )
    assert out == "recovered"
    assert attempts["n"] == 3
    assert len(calls) == 2  # slept before attempts 2 and 3


async def test_non_retryable_raises_immediately():
    calls, sleep = _recorder()

    async def factory():
        raise ValueError("not a rate limit")

    with pytest.raises(ValueError):
        await run_with_backoff(factory, sleep=sleep, rng=lambda: 0.0)
    assert calls == []  # never retried


async def test_exhausts_attempts_and_reraises():
    calls, sleep = _recorder()

    async def factory():
        raise _StatusError(429)

    with pytest.raises(_StatusError):
        await run_with_backoff(
            factory, max_attempts=4, sleep=sleep, rng=lambda: 0.0
        )
    assert len(calls) == 3  # 4 attempts -> 3 sleeps between them


async def test_honors_retry_after_header():
    calls, sleep = _recorder()
    attempts = {"n": 0}

    async def factory():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _StatusError(429, {"retry-after": "7"})
        return "ok"

    out = await run_with_backoff(factory, sleep=sleep, rng=lambda: 0.0)
    assert out == "ok"
    assert calls == [7.0]  # used the header, not computed backoff


async def test_backoff_is_exponential_and_capped():
    calls, sleep = _recorder()

    async def factory():
        raise _StatusError(429)

    with pytest.raises(_StatusError):
        await run_with_backoff(
            factory,
            max_attempts=6,
            base_delay=2.0,
            max_delay=10.0,
            sleep=sleep,
            rng=lambda: 0.0,  # zero jitter -> deterministic
        )
    # 2, 4, 8, then capped at 10, 10
    assert calls == [2.0, 4.0, 8.0, 10.0, 10.0]
