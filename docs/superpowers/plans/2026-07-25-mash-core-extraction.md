# mash-core Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract book-mash's judge-core cluster (`JudgeDim`, `JudgeScore`-family types, provider-agnostic model factory, model settings, pricing table, retry-with-backoff) into a new standalone repo, `mash-core`, and rewire book-mash to depend on it.

**Architecture:** New sibling Poetry project `mash-core/` with package `mash_core`, containing verbatim (renamed) copies of the six files below plus a re-exporting `__init__.py`. book-mash adds it as a local Poetry path dependency and every internal import that currently points at `book_mash.judges.{base,models,_model_factory,_model_settings,_pricing,_retry}` is rewritten to import from `mash_core` instead. Those six source files (and the two test files that test them in isolation) are then deleted from book-mash.

**Tech Stack:** Python 3.13, Poetry (dual PEP 621 / `[tool.poetry]` metadata, matching book-mash's existing convention), pydantic 2.9, pydantic-ai 0.0.40, pytest + pytest-asyncio, mypy, ruff.

## Global Constraints

- Python `>=3.13,<3.14` (matches book-mash exactly).
- `pydantic>=2.9,<3.0`, `pydantic-ai>=0.0.40,<0.0.41`, `anthropic>=0.49,<0.50`, `openai>=1.50,<2.0` — same pins as book-mash, since `mash_core.model_factory` constructs `AnthropicModel`/`OpenAIModel` from pydantic-ai and needs the matching SDK versions.
- MIT license, `Copyright (c) 2026 Timur Isachenko` — copy book-mash's `LICENSE` verbatim.
- No new abstractions: every moved file's logic is copied verbatim; only import paths and module names change (dropping the internal `_` prefix now that these are the package's own public modules, e.g. `_model_factory.py` → `model_factory.py`).
- Package must ship a `py.typed` marker (PEP 561) so book-mash's `mypy book_mash` run doesn't flag missing stubs for `mash_core`.
- After every task that touches book-mash, run `poetry run pytest`, `poetry run mypy book_mash`, `poetry run ruff check book_mash tests` and confirm they pass before moving on — book-mash's CLAUDE.md commands are the source of truth for this.
- Commit after each task (per-task commits, not one giant commit at the end).

---

### Task 1: Scaffold `mash-core` and land `models.py` + `base.py`

**Files:**
- Create: `/Users/timur_isachenko/Dev/LifeOS/mash-core/pyproject.toml`
- Create: `/Users/timur_isachenko/Dev/LifeOS/mash-core/LICENSE`
- Create: `/Users/timur_isachenko/Dev/LifeOS/mash-core/README.md`
- Create: `/Users/timur_isachenko/Dev/LifeOS/mash-core/mash_core/__init__.py` (empty for now — populated in Task 3)
- Create: `/Users/timur_isachenko/Dev/LifeOS/mash-core/mash_core/py.typed` (empty marker file)
- Create: `/Users/timur_isachenko/Dev/LifeOS/mash-core/mash_core/models.py`
- Create: `/Users/timur_isachenko/Dev/LifeOS/mash-core/mash_core/base.py`
- Create: `/Users/timur_isachenko/Dev/LifeOS/mash-core/tests/test_models.py`
- Create: `/Users/timur_isachenko/Dev/LifeOS/mash-core/tests/test_base.py`

**Interfaces:**
- Produces: `mash_core.models.JudgeLabel` (str Enum: STRONG/MODERATE/WEAK/FAIL/ERROR), `mash_core.models.UnitType` (`Literal["paragraph","section","chapter"]`), `mash_core.models.JudgeScore` (pydantic `BaseModel`: `dim_name: str`, `unit_id: str`, `score_0_100: float | None`, `label: JudgeLabel`, `reasoning: str`, `evidence_refs: list[str]`, `model: str`, `cost_usd: float`, `derived: bool = False`), `mash_core.models.JudgeInput` (`unit_id: str`, `unit_type: UnitType`, `unit_text: str`, `dim_name: str`, `context: dict`), `mash_core.models.JudgeResult` (`unit_id: str`, `scores: list[JudgeScore]`).
- Produces: `mash_core.base.JudgeDim` — ABC with `ClassVar`s `name: str`, `unit_type: str`, `model_id: str`; abstract `async def judge(self, input: JudgeInput) -> JudgeScore`; concrete `context_cache_key(self, input: JudgeInput) -> str` (returns `""`); concrete `label_for_score(self, score: float) -> str` (`>=80` "strong", `>=50` "moderate", `>=20` "weak", else "fail").

- [ ] **Step 1: Scaffold the repo skeleton**

```bash
mkdir -p /Users/timur_isachenko/Dev/LifeOS/mash-core/mash_core
mkdir -p /Users/timur_isachenko/Dev/LifeOS/mash-core/tests
cd /Users/timur_isachenko/Dev/LifeOS/mash-core
git init
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
## Dual-compat: PEP 621 `[project]` is read by pip and modern build tools.
## Poetry 1.8.x reads `[tool.poetry]` — keep both in sync on version bumps.

[project]
name = "mash-core"
version = "0.1.0"
description = "Provider-agnostic LLM-judge core: JudgeDim/JudgeScore contract, cost-aware model routing, retry-with-backoff. Extracted from book-mash."
readme = "README.md"
requires-python = ">=3.13,<3.14"
license = "MIT"
license-files = ["LICENSE"]
authors = [{name = "Timur Isachenko"}]
keywords = ["llm-judge", "evaluation", "mash", "pydantic-ai"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.13",
]
dependencies = [
    "pydantic>=2.9,<3.0",
    "pydantic-ai>=0.0.40,<0.0.41",
    "anthropic>=0.49,<0.50",
    "openai>=1.50,<2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3,<9.0",
    "pytest-asyncio>=0.24,<0.25",
    "pytest-mock>=3.14,<4.0",
    "mypy>=1.13,<2.0",
    "ruff>=0.7,<1.0",
]

# Legacy Poetry 1.x metadata. Mirrors `[project]` above — keep them in sync.
[tool.poetry]
name = "mash-core"
version = "0.1.0"
description = "Provider-agnostic LLM-judge core: JudgeDim/JudgeScore contract, cost-aware model routing, retry-with-backoff. Extracted from book-mash."
authors = ["Timur Isachenko"]
readme = "README.md"
license = "MIT"
packages = [{include = "mash_core"}]

[tool.poetry.dependencies]
python = "^3.13,<3.14"
pydantic = "^2.9"
pydantic-ai = "^0.0.40"
anthropic = "^0.49"
openai = "^1.50"

[tool.poetry.group.dev.dependencies]
pytest = "^8.3"
pytest-asyncio = "^0.24"
pytest-mock = "^3.14"
mypy = "^1.13"
ruff = "^0.7"

[tool.pytest.ini_options]
asyncio_mode = "auto"

[tool.ruff]
target-version = "py313"
line-length = 100

[build-system]
requires = ["poetry-core>=1.8.0"]
build-backend = "poetry.core.masonry.api"
```

- [ ] **Step 3: Write `LICENSE`** (verbatim copy of book-mash's)

```
MIT License

Copyright (c) 2026 Timur Isachenko

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 4: Write `README.md`**

```markdown
# mash-core

Provider-agnostic LLM-judge core: the `JudgeDim`/`JudgeScore` contract, cost-aware
model routing (Anthropic / OpenRouter / OpenAI-compatible via pydantic-ai), a
per-model pricing table, and retry-with-backoff tuned for judge calls.

Extracted from [`book-mash`](https://github.com/isatimur/book-mash)'s
`book_mash/judges/{base,models,_model_factory,_model_settings,_pricing,_retry}.py`,
where this cluster was already generic and decoupled from book-mash's own
manuscript-specific rollup/broadcast logic. book-mash depends on this package;
this package has no book-mash dependency in the other direction.

## Install

```bash
poetry install
```

Path-dependency install from a sibling consumer project (e.g. book-mash):

```toml
[tool.poetry.dependencies]
mash-core = {path = "../mash-core", develop = true}
```

## What's here

- `mash_core.models` — `JudgeLabel`, `UnitType`, `JudgeScore`, `JudgeInput`, `JudgeResult`.
- `mash_core.base` — the `JudgeDim` ABC every judge dimension implements.
- `mash_core.model_factory` — `build_judge_model()`, provider-configurable via
  `BOOK_MASH_JUDGE_PROVIDER` / `BOOK_MASH_JUDGE_MODEL` / `BOOK_MASH_JUDGE_BASE_URL` /
  `BOOK_MASH_JUDGE_API_KEY_ENV` env vars.
- `mash_core.model_settings` — `JUDGE_MODEL_SETTINGS` (temperature=0, timeout, max_tokens cap).
- `mash_core.pricing` — `estimate_cost(model_id, input_tokens, output_tokens)`.
- `mash_core.retry` — `run_with_backoff(factory, ...)`.

License: MIT.
```

- [ ] **Step 5: Create the `py.typed` marker**

```bash
touch /Users/timur_isachenko/Dev/LifeOS/mash-core/mash_core/py.typed
```

- [ ] **Step 6: Write `mash_core/models.py`** (verbatim from `book-mash/book_mash/judges/models.py`)

```python
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class JudgeLabel(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    FAIL = "fail"
    ERROR = "error"


UnitType = Literal["paragraph", "section", "chapter"]


class JudgeScore(BaseModel):
    dim_name: str
    unit_id: str
    score_0_100: float | None  # None when label == ERROR
    label: JudgeLabel
    reasoning: str
    evidence_refs: list[str] = Field(default_factory=list)
    model: str
    cost_usd: float
    derived: bool = False  # True when broadcast from a higher-level native score


class JudgeInput(BaseModel):
    unit_id: str
    unit_type: UnitType
    unit_text: str
    dim_name: str
    context: dict = Field(default_factory=dict)


class JudgeResult(BaseModel):
    unit_id: str
    scores: list[JudgeScore]
```

- [ ] **Step 7: Write `mash_core/base.py`** (same as book-mash's `base.py`, import path updated)

```python
from abc import ABC, abstractmethod
from typing import ClassVar

from mash_core.models import JudgeInput, JudgeScore


class JudgeDim(ABC):
    name: ClassVar[str]
    unit_type: ClassVar[str]  # "paragraph" | "section" | "chapter"
    model_id: ClassVar[str]

    @abstractmethod
    async def judge(self, input: JudgeInput) -> JudgeScore:
        ...

    def context_cache_key(self, input: JudgeInput) -> str:
        # Override to add mutable context (e.g. claims ledger) to the cache key.
        return ""

    def label_for_score(self, score: float) -> str:
        if score >= 80:
            return "strong"
        if score >= 50:
            return "moderate"
        if score >= 20:
            return "weak"
        return "fail"
```

- [ ] **Step 8: Write `tests/test_models.py`** (the two JudgeScore tests, moved out of book-mash's `test_models.py`)

```python
from mash_core import JudgeLabel, JudgeScore


def test_judge_score_label_enum():
    s = JudgeScore(
        dim_name="humanness",
        unit_id="paragraph:x",
        score_0_100=72.0,
        label=JudgeLabel.MODERATE,
        reasoning="Has a point of view but generic intro.",
        evidence_refs=["phrase:'in today's evolving landscape'"],
        model="claude-sonnet-4-6",
        cost_usd=0.011,
        derived=False,
    )
    assert s.label == JudgeLabel.MODERATE
    assert s.score_0_100 == 72.0


def test_judge_score_error_allows_null_score():
    s = JudgeScore(
        dim_name="humanness",
        unit_id="paragraph:x",
        score_0_100=None,
        label=JudgeLabel.ERROR,
        reasoning="API timeout after 3 retries",
        evidence_refs=[],
        model="claude-sonnet-4-6",
        cost_usd=0.0,
        derived=False,
    )
    assert s.score_0_100 is None
    assert s.label == JudgeLabel.ERROR
```

Note: this imports `from mash_core import ...` (the package's public surface), which won't resolve until Task 3 populates `__init__.py`. That's expected — Step 9 below runs it and confirms the *specific* failure.

- [ ] **Step 9: Write `tests/test_base.py`** (the JudgeDim test, moved out of book-mash's `test_judge_base.py`)

```python
from mash_core import JudgeDim, JudgeInput, JudgeLabel, JudgeScore


class FakeJudge(JudgeDim):
    name = "fake"
    unit_type = "paragraph"
    model_id = "test-model"

    async def judge(self, input: JudgeInput) -> JudgeScore:
        return JudgeScore(
            dim_name=self.name,
            unit_id=input.unit_id,
            score_0_100=75.0,
            label=JudgeLabel.MODERATE,
            reasoning="fake reasoning",
            evidence_refs=[],
            model=self.model_id,
            cost_usd=0.0,
            derived=False,
        )


async def test_judge_returns_score():
    judge = FakeJudge()
    input = JudgeInput(
        unit_id="paragraph:x",
        unit_type="paragraph",
        unit_text="hello",
        dim_name="fake",
    )
    score = await judge.judge(input)
    assert score.score_0_100 == 75.0
    assert score.label == JudgeLabel.MODERATE
```

- [ ] **Step 10: Leave `mash_core/__init__.py` empty for now, install deps, and confirm the expected failure**

```bash
cd /Users/timur_isachenko/Dev/LifeOS/mash-core
poetry install
poetry run pytest -v
```

Expected: collection or test FAILURE — `ImportError: cannot import name 'JudgeLabel' from 'mash_core'` (or `JudgeDim`) — because `__init__.py` is still empty. This confirms the tests are wired to the real package path before Task 3 makes them pass.

- [ ] **Step 11: Commit**

```bash
git add pyproject.toml LICENSE README.md mash_core/ tests/
git commit -m "Scaffold mash-core: models.py and base.py"
```

---

### Task 2: Land `retry.py`, `pricing.py`, `model_settings.py`, `model_factory.py`

**Files:**
- Create: `/Users/timur_isachenko/Dev/LifeOS/mash-core/mash_core/retry.py`
- Create: `/Users/timur_isachenko/Dev/LifeOS/mash-core/mash_core/pricing.py`
- Create: `/Users/timur_isachenko/Dev/LifeOS/mash-core/mash_core/model_settings.py`
- Create: `/Users/timur_isachenko/Dev/LifeOS/mash-core/mash_core/model_factory.py`
- Create: `/Users/timur_isachenko/Dev/LifeOS/mash-core/tests/test_retry.py`
- Create: `/Users/timur_isachenko/Dev/LifeOS/mash-core/tests/test_pricing.py`
- Create: `/Users/timur_isachenko/Dev/LifeOS/mash-core/tests/test_model_settings.py`
- Create: `/Users/timur_isachenko/Dev/LifeOS/mash-core/tests/test_model_factory.py`

**Interfaces:**
- Consumes: nothing from Task 1's public API (these four modules are self-contained; `model_factory.py` doesn't import `models.py` or `base.py`).
- Produces: `mash_core.retry.run_with_backoff(factory, *, max_attempts=6, base_delay=2.0, max_delay=60.0, sleep=asyncio.sleep, rng=random.random) -> T` (plus private `_is_retryable`, `_retry_after`, `_status_code` for direct testing). `mash_core.pricing.estimate_cost(model_id: str, input_tokens: int | None, output_tokens: int | None) -> float`. `mash_core.model_settings.JUDGE_MODEL_SETTINGS` (dict-like `ModelSettings`) and `JUDGE_REQUEST_TIMEOUT_S: float = 120.0`. `mash_core.model_factory.build_judge_model() -> tuple[Model, str]`, `DEFAULT_JUDGE_MODEL_ID = "claude-sonnet-4-6"`, `DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"`, `DEFAULT_JUDGE_API_KEY_ENV = "OPENROUTER_API_KEY"`.

- [ ] **Step 1: Write `mash_core/retry.py`** (verbatim from book-mash's `_retry.py`, file renamed)

```python
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
```

- [ ] **Step 2: Write `tests/test_retry.py`** (moved verbatim from book-mash's `tests/test_retry.py`, import path updated)

```python
import pytest

from mash_core.retry import (
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
```

- [ ] **Step 3: Write `mash_core/pricing.py`** (verbatim from book-mash's `_pricing.py`, file renamed)

```python
# USD per token, per model. Revise when Anthropic pricing changes.
_PRICE_PER_TOKEN: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
    "claude-haiku-4-5": {"input": 1.0 / 1_000_000, "output": 5.0 / 1_000_000},
    "openai-compatible:gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "openai-compatible:gpt-4.1-mini": {"input": 0.40 / 1_000_000, "output": 1.60 / 1_000_000},
    "openai-compatible:gpt-5-chat-latest": {"input": 1.25 / 1_000_000, "output": 10.0 / 1_000_000},
    "openrouter:moonshotai/kimi-k2-0905": {"input": 0.60 / 1_000_000, "output": 2.50 / 1_000_000},
    "openrouter:qwen/qwen3-235b-a22b-2507": {"input": 0.09 / 1_000_000, "output": 0.55 / 1_000_000},
    "openrouter:z-ai/glm-4.7": {"input": 0.40 / 1_000_000, "output": 1.75 / 1_000_000},
}


def estimate_cost(model_id: str, input_tokens: int | None, output_tokens: int | None) -> float:
    rates = _PRICE_PER_TOKEN.get(model_id)
    if rates is None:
        return 0.0
    return (input_tokens or 0) * rates["input"] + (output_tokens or 0) * rates["output"]
```

- [ ] **Step 4: Write `tests/test_pricing.py`** (new — book-mash had no dedicated test for this module; only covered indirectly via judge tests)

```python
from mash_core import estimate_cost


def test_known_model_computes_input_and_output_cost():
    # claude-sonnet-4-6: $3/1M input, $15/1M output.
    cost = estimate_cost("claude-sonnet-4-6", 1_000_000, 1_000_000)
    assert cost == 3.0 + 15.0


def test_unknown_model_returns_zero():
    assert estimate_cost("nonexistent-model", 1000, 1000) == 0.0


def test_none_token_counts_treated_as_zero():
    assert estimate_cost("claude-sonnet-4-6", None, None) == 0.0
```

- [ ] **Step 5: Write `mash_core/model_settings.py`** (verbatim from book-mash's `_model_settings.py`, file renamed)

```python
"""Shared model settings for every judge Agent.

Two reliability properties live here so every judge dimension gets them
identically — there is one definition, not N copies that can drift apart.

1. ``temperature=0.0`` — makes cold re-runs near-reproducible. (No provider
   guarantees *bit*-exact determinism even at 0, but it removes the dominant
   sampling-variance source.)

2. ``timeout`` — a per-request httpx timeout, so one stalled connection can't
   block an entire run. A timed-out request raises an APITimeout-class error,
   which ``run_with_backoff`` treats as retryable (bounded ``max_attempts``);
   if it keeps timing out the exception re-raises and the caller's existing
   handler records a clean ERROR-labeled score instead of hanging forever.

pydantic-ai 0.0.40: ``ModelSettings`` is a TypedDict; ``AnthropicModel`` reads
``temperature`` and ``timeout`` from it and passes them straight into the
Anthropic SDK ``messages.create`` call. Pass it to ``Agent(model_settings=...)``.
"""

from pydantic_ai.settings import ModelSettings

# Per-request HTTP timeout in seconds. Bounds a single stalled call; combined
# with run_with_backoff's bounded retries this caps total time per unit instead
# of allowing an unbounded hang.
JUDGE_REQUEST_TIMEOUT_S = 120.0

# Single source of truth for every judge Agent's sampling + timeout behavior.
JUDGE_MODEL_SETTINGS = ModelSettings(
    temperature=0.0,
    timeout=JUDGE_REQUEST_TIMEOUT_S,
    # Judges return a short structured verdict (~300 tokens). Without an explicit
    # cap, OpenAI-compatible gateways (OpenRouter) pre-authorize the model's full
    # output window (16k+) per request against remaining credits, which 402s
    # whole runs at high concurrency long before any real spend.
    max_tokens=2048,
)
```

- [ ] **Step 6: Write `tests/test_model_settings.py`** (new — book-mash had no dedicated test for this module)

```python
from mash_core import JUDGE_MODEL_SETTINGS, JUDGE_REQUEST_TIMEOUT_S


def test_judge_model_settings_are_deterministic_and_bounded():
    assert JUDGE_MODEL_SETTINGS["temperature"] == 0.0
    assert JUDGE_MODEL_SETTINGS["timeout"] == JUDGE_REQUEST_TIMEOUT_S
    assert JUDGE_MODEL_SETTINGS["max_tokens"] == 2048


def test_judge_request_timeout_is_120_seconds():
    assert JUDGE_REQUEST_TIMEOUT_S == 120.0
```

- [ ] **Step 7: Write `mash_core/model_factory.py`** (verbatim from book-mash's `_model_factory.py`, file renamed, docstring unchanged since it's still accurate)

```python
"""Configurable judge-model factory — one place all judges get their model.

Historically every judge hardwired ``AnthropicModel("claude-sonnet-4-6", ...)``.
This factory makes the judge model configurable from the environment so judges can
run on a cheaper, non-Anthropic model via OpenRouter (OpenAI-compatible API).

Two reasons to want that:

1. **Cost / access.** OpenRouter is pay-as-you-go and ~11-14x cheaper than
   Anthropic Sonnet for comparable open models (e.g. ``deepseek/deepseek-chat``).
2. **Cross-family objectivity.** The manuscript under test is partly Claude-written.
   Judging Claude prose with a Claude judge is self-preference bias; a non-Claude
   judge is a more objective grader.

Default behavior is UNCHANGED: with no env vars set, the factory returns the same
Anthropic Sonnet model and the same model-id string ``claude-sonnet-4-6`` as before,
so existing runs and cached scores are byte-for-byte compatible.

Environment / configuration
---------------------------
- ``BOOK_MASH_JUDGE_PROVIDER`` — ``anthropic`` (default) | ``openrouter`` |
  ``openai-compatible``.
- ``BOOK_MASH_JUDGE_MODEL`` — model id. Default: ``claude-sonnet-4-6`` (the prior
  hardwired id). For OpenRouter this is the OpenRouter model slug, e.g.
  ``deepseek/deepseek-chat`` or ``meta-llama/llama-3.3-70b-instruct``.
- ``BOOK_MASH_JUDGE_BASE_URL`` — base URL for the OpenAI-compatible endpoint.
  Defaults to OpenRouter's ``https://openrouter.ai/api/v1`` for both the
  ``openrouter`` and ``openai-compatible`` providers (override for any other
  OpenAI-compatible gateway).
- ``BOOK_MASH_JUDGE_API_KEY_ENV`` — name of the env var holding the API key.
  Defaults to ``OPENROUTER_API_KEY``. The key value is read from whatever env var
  this names, so a different gateway can use its own key var without code changes.
- Anthropic provider continues to read ``ANTHROPIC_API_KEY`` directly.

Cache-key contract
------------------
The factory returns BOTH the model object AND a stable model-id STRING. Consumers
should thread that string into their own cache key and into their per-result score's
``model`` field. Non-Anthropic providers get a provider-prefixed id (e.g.
``openrouter:deepseek/deepseek-chat``) so a provider switch ALWAYS busts the cache —
an Anthropic-keyed cached score is never reused for an OpenRouter run, and vice-versa,
even in the unlikely case two providers expose the same bare model slug.

pydantic-ai 0.0.40 API (verified against the installed version)
--------------------------------------------------------------
``poetry run python -c "from pydantic_ai.models.openai import OpenAIModel; import
inspect; print(inspect.signature(OpenAIModel.__init__))"`` reports::

    OpenAIModel.__init__(self, model_name, *, provider='openai' | Provider | None,
                         base_url=None, api_key=None, openai_client=None, ...)

So an OpenAI-compatible endpoint is constructed directly via
``OpenAIModel(model_name, base_url=..., api_key=...)`` (no separate provider object
required). pydantic-ai requests structured output for ``result_type`` via the
OpenAI tool/JSON path, so a judge's pydantic ``result_type`` works unchanged.
"""

from __future__ import annotations

import os

from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel

# The previously hardwired Anthropic model id. Kept as the default so that with no
# env configuration the factory reproduces the original behavior exactly (same
# model object and same model-id string flowing into the cache key).
DEFAULT_JUDGE_MODEL_ID = "claude-sonnet-4-6"

# Default OpenAI-compatible gateway. OpenRouter speaks the OpenAI API.
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default env var name for the OpenAI-compatible API key.
DEFAULT_JUDGE_API_KEY_ENV = "OPENROUTER_API_KEY"


def build_judge_model() -> tuple[Model, str]:
    """Construct the judge model from the environment.

    Returns a ``(model, model_id)`` tuple. ``model_id`` is the stable cache-key
    string (provider-prefixed for non-Anthropic providers) — see the module
    docstring's cache-key contract.
    """
    provider = os.environ.get("BOOK_MASH_JUDGE_PROVIDER", "anthropic").strip().lower()
    model_name = os.environ.get("BOOK_MASH_JUDGE_MODEL", DEFAULT_JUDGE_MODEL_ID).strip()
    model: Model

    if provider == "anthropic":
        # AnthropicModel takes api_key= directly in pydantic-ai 0.0.40 (no provider
        # object). Same construction as the prior hardwired judges.
        model = AnthropicModel(model_name, api_key=os.environ["ANTHROPIC_API_KEY"])
        # Bare id for back-compat: an unset env reproduces "claude-sonnet-4-6" and
        # therefore the exact prior cache keys.
        return model, model_name

    if provider in ("openrouter", "openai-compatible"):
        base_url = os.environ.get("BOOK_MASH_JUDGE_BASE_URL", DEFAULT_OPENROUTER_BASE_URL).strip()
        api_key_env = os.environ.get("BOOK_MASH_JUDGE_API_KEY_ENV", DEFAULT_JUDGE_API_KEY_ENV).strip()
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"BOOK_MASH_JUDGE_PROVIDER={provider} requires an API key in env "
                f"var {api_key_env!r} (set BOOK_MASH_JUDGE_API_KEY_ENV to use a "
                f"different var)."
            )
        # OpenAI-compatible construction; pydantic-ai requests structured output for
        # result_type via the OpenAI tool/JSON path.
        model = OpenAIModel(model_name, base_url=base_url, api_key=api_key)
        # Provider-prefix the cache-key id so an OpenRouter run never collides with
        # an Anthropic-keyed cached score (and vice-versa), even if two gateways
        # expose the same bare slug.
        return model, f"openrouter:{model_name}" if provider == "openrouter" else f"openai-compatible:{model_name}"

    raise RuntimeError(
        f"Unknown BOOK_MASH_JUDGE_PROVIDER={provider!r}; "
        f"expected 'anthropic', 'openrouter', or 'openai-compatible'."
    )
```

- [ ] **Step 8: Write `tests/test_model_factory.py`** (moved verbatim from book-mash's `tests/test_model_factory.py`, import path updated)

```python
"""Tests for the configurable judge-model factory (mash_core/model_factory.py).

No network calls: pydantic-ai constructs the model object (and its SDK client)
lazily without issuing a request, so we can assert on the model type / model_name /
base_url with only a dummy API key in the environment.

The cache-key contract is the load-bearing property under test: the model-id STRING
returned by the factory is what a consumer threads into its own cache key and score
record, so a provider switch must produce a DIFFERENT string (otherwise an
Anthropic-keyed cached score would be wrongly reused for an OpenRouter run, and
vice-versa).
"""

import pytest

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel

from mash_core import (
    DEFAULT_JUDGE_MODEL_ID,
    DEFAULT_OPENROUTER_BASE_URL,
    build_judge_model,
)


# All factory env vars, cleared before each test so a test sets exactly what it means.
_FACTORY_ENV = (
    "BOOK_MASH_JUDGE_PROVIDER",
    "BOOK_MASH_JUDGE_MODEL",
    "BOOK_MASH_JUDGE_BASE_URL",
    "BOOK_MASH_JUDGE_API_KEY_ENV",
)


@pytest.fixture(autouse=True)
def _clean_factory_env(monkeypatch):
    for var in _FACTORY_ENV:
        monkeypatch.delenv(var, raising=False)
    # A dummy key is enough — no request is made during construction.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-anthropic-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def test_unset_env_returns_anthropic_default_and_legacy_model_id():
    """(a) Back-compat: no factory env -> Anthropic default + the prior model-id
    string, so existing cache keys and score records are unchanged."""
    model, model_id = build_judge_model()

    assert isinstance(model, AnthropicModel)
    assert model.model_name == "claude-sonnet-4-6"
    # The model-id string is exactly the legacy hardwired value -> identical cache keys.
    assert model_id == DEFAULT_JUDGE_MODEL_ID == "claude-sonnet-4-6"


def test_openrouter_returns_openai_model_pointed_at_openrouter(monkeypatch):
    """(b) openrouter provider -> OpenAIModel at the OpenRouter base_url, model-id
    string is the OpenRouter slug (prefixed) so the cache busts vs anthropic."""
    monkeypatch.setenv("BOOK_MASH_JUDGE_PROVIDER", "openrouter")
    monkeypatch.setenv("BOOK_MASH_JUDGE_MODEL", "deepseek/deepseek-chat")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter-key")

    model, model_id = build_judge_model()

    assert isinstance(model, OpenAIModel)
    assert model.model_name == "deepseek/deepseek-chat"
    # OpenAIModel exposes the underlying AsyncOpenAI client; base_url proves the
    # request would go to OpenRouter, not Anthropic. (No request is made here.)
    assert str(model.client.base_url).rstrip("/") == DEFAULT_OPENROUTER_BASE_URL.rstrip("/")
    # The slug appears in the cache-key string (provider-prefixed).
    assert "deepseek/deepseek-chat" in model_id


def test_provider_switch_changes_model_id_so_cache_keys_differ(monkeypatch):
    """(c) The model-id string differs between providers, so a consumer's cache keys
    differ and a provider switch cannot reuse the other provider's cached scores."""
    anthropic_model, anthropic_id = build_judge_model()

    monkeypatch.setenv("BOOK_MASH_JUDGE_PROVIDER", "openrouter")
    monkeypatch.setenv("BOOK_MASH_JUDGE_MODEL", "deepseek/deepseek-chat")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter-key")
    _, openrouter_id = build_judge_model()

    assert anthropic_id != openrouter_id
    # Even if a future config named the same bare slug under anthropic, the provider
    # prefix on the OpenRouter id keeps the two cache namespaces disjoint.
    assert openrouter_id.startswith("openrouter:")


def test_openai_compatible_uses_custom_base_url_and_key_env(monkeypatch):
    """openai-compatible provider honors a custom base_url and a configurable key env
    var, and prefixes the model-id distinctly from openrouter."""
    monkeypatch.setenv("BOOK_MASH_JUDGE_PROVIDER", "openai-compatible")
    monkeypatch.setenv("BOOK_MASH_JUDGE_MODEL", "meta-llama/llama-3.3-70b-instruct")
    monkeypatch.setenv("BOOK_MASH_JUDGE_BASE_URL", "https://gateway.example.com/v1")
    monkeypatch.setenv("BOOK_MASH_JUDGE_API_KEY_ENV", "MY_GATEWAY_KEY")
    monkeypatch.setenv("MY_GATEWAY_KEY", "dummy-gateway-key")

    model, model_id = build_judge_model()

    assert isinstance(model, OpenAIModel)
    assert str(model.client.base_url).rstrip("/") == "https://gateway.example.com/v1"
    assert model_id == "openai-compatible:meta-llama/llama-3.3-70b-instruct"


def test_openrouter_without_api_key_raises(monkeypatch):
    """A missing OpenRouter key is a clear configuration error, not a silent
    fallthrough to a keyless (and failing) network call."""
    monkeypatch.setenv("BOOK_MASH_JUDGE_PROVIDER", "openrouter")
    monkeypatch.setenv("BOOK_MASH_JUDGE_MODEL", "deepseek/deepseek-chat")
    # OPENROUTER_API_KEY intentionally not set.
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        build_judge_model()


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("BOOK_MASH_JUDGE_PROVIDER", "gemini-direct")
    with pytest.raises(RuntimeError, match="Unknown BOOK_MASH_JUDGE_PROVIDER"):
        build_judge_model()
```

- [ ] **Step 9: Confirm the still-empty `__init__.py` fails the same documented way**

```bash
cd /Users/timur_isachenko/Dev/LifeOS/mash-core
poetry run pytest -v
```

Expected: FAIL — `ImportError` on `mash_core.estimate_cost` / `mash_core.JUDGE_MODEL_SETTINGS` / `mash_core.build_judge_model` (public names not yet re-exported). `test_retry.py` (imports from `mash_core.retry` submodule directly) should already PASS.

- [ ] **Step 10: Commit**

```bash
git add mash_core/ tests/
git commit -m "Add retry, pricing, model_settings, model_factory modules"
```

---

### Task 3: Public API surface, full mash-core suite green

**Files:**
- Modify: `/Users/timur_isachenko/Dev/LifeOS/mash-core/mash_core/__init__.py`

**Interfaces:**
- Consumes: every symbol produced in Tasks 1–2.
- Produces: the package's public import surface — `from mash_core import (JudgeLabel, UnitType, JudgeScore, JudgeInput, JudgeResult, JudgeDim, build_judge_model, DEFAULT_JUDGE_MODEL_ID, DEFAULT_OPENROUTER_BASE_URL, DEFAULT_JUDGE_API_KEY_ENV, JUDGE_MODEL_SETTINGS, JUDGE_REQUEST_TIMEOUT_S, estimate_cost, run_with_backoff)`.

- [ ] **Step 1: Write `mash_core/__init__.py`**

```python
from mash_core.models import JudgeInput, JudgeLabel, JudgeResult, JudgeScore, UnitType
from mash_core.base import JudgeDim
from mash_core.model_factory import (
    DEFAULT_JUDGE_API_KEY_ENV,
    DEFAULT_JUDGE_MODEL_ID,
    DEFAULT_OPENROUTER_BASE_URL,
    build_judge_model,
)
from mash_core.model_settings import JUDGE_MODEL_SETTINGS, JUDGE_REQUEST_TIMEOUT_S
from mash_core.pricing import estimate_cost
from mash_core.retry import run_with_backoff

__all__ = [
    "JudgeInput",
    "JudgeLabel",
    "JudgeResult",
    "JudgeScore",
    "UnitType",
    "JudgeDim",
    "build_judge_model",
    "DEFAULT_JUDGE_API_KEY_ENV",
    "DEFAULT_JUDGE_MODEL_ID",
    "DEFAULT_OPENROUTER_BASE_URL",
    "JUDGE_MODEL_SETTINGS",
    "JUDGE_REQUEST_TIMEOUT_S",
    "estimate_cost",
    "run_with_backoff",
]
```

- [ ] **Step 2: Run the full mash-core test suite**

```bash
cd /Users/timur_isachenko/Dev/LifeOS/mash-core
poetry run pytest -v
```

Expected: all tests PASS (Tasks 1–2's tests, previously failing on the empty `__init__.py`, now succeed).

- [ ] **Step 3: Type-check and lint**

```bash
poetry run mypy mash_core
poetry run ruff check mash_core tests
```

Expected: both clean. If `ruff` flags unused imports or import order in `__init__.py`, fix with `poetry run ruff check --fix mash_core`.

- [ ] **Step 4: Commit**

```bash
git add mash_core/__init__.py
git commit -m "Wire mash_core public API; full test suite green"
```

---

### Task 4: Wire book-mash to depend on mash-core

**Files:**
- Modify: `/Users/timur_isachenko/Dev/LifeOS/book-mash/pyproject.toml`

**Interfaces:**
- Consumes: the `mash-core` package published locally at `../mash-core` (path dependency, not yet on PyPI).
- Produces: `mash-core` importable from book-mash's Poetry-managed virtualenv.

- [ ] **Step 1: Add `mash-core` to `[project] dependencies`**

In `/Users/timur_isachenko/Dev/LifeOS/book-mash/pyproject.toml`, under `[project]`, change:

```toml
dependencies = [
    "pydantic>=2.9,<3.0",
    "pydantic-ai>=0.0.40,<0.0.41",
    "anthropic>=0.49,<0.50",
    "openai>=1.50,<2.0",
    "typer>=0.15,<0.16",
    "rich>=13.9,<14.0",
    "tomli>=2.0,<3.0",
]
```

to:

```toml
dependencies = [
    "mash-core @ file:///Users/timur_isachenko/Dev/LifeOS/mash-core",
    "pydantic>=2.9,<3.0",
    "pydantic-ai>=0.0.40,<0.0.41",
    "anthropic>=0.49,<0.50",
    "openai>=1.50,<2.0",
    "typer>=0.15,<0.16",
    "rich>=13.9,<14.0",
    "tomli>=2.0,<3.0",
]
```

- [ ] **Step 2: Add the Poetry path dependency**

Under `[tool.poetry.dependencies]`, change:

```toml
[tool.poetry.dependencies]
python = "^3.13,<3.14"
pydantic = "^2.9"
pydantic-ai = "^0.0.40"
anthropic = "^0.49"
openai = "^1.50"
typer = "^0.15"
rich = "^13.9"
tomli = "^2.0"
```

to:

```toml
[tool.poetry.dependencies]
python = "^3.13,<3.14"
mash-core = {path = "../mash-core", develop = true}
pydantic = "^2.9"
pydantic-ai = "^0.0.40"
anthropic = "^0.49"
openai = "^1.50"
typer = "^0.15"
rich = "^13.9"
tomli = "^2.0"
```

- [ ] **Step 3: Relock and install**

```bash
cd /Users/timur_isachenko/Dev/LifeOS/book-mash
poetry lock
poetry install
```

Expected: lock succeeds, `mash-core` installed in develop/editable mode from the sibling path.

- [ ] **Step 4: Confirm it imports**

```bash
poetry run python -c "import mash_core; print(mash_core.JudgeDim, mash_core.estimate_cost('claude-sonnet-4-6', 100, 100))"
```

Expected: prints the `JudgeDim` class and a nonzero float — no `ImportError`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml poetry.lock
git commit -m "deps: add mash-core as a local path dependency"
```

---

### Task 5: Rewrite book-mash's source imports to use mash_core

**Files:**
- Modify: `book_mash/judges/humanness.py`
- Modify: `book_mash/judges/usefulness.py`
- Modify: `book_mash/judges/voice.py`
- Modify: `book_mash/judges/evidence_density.py`
- Modify: `book_mash/judges/claim_defensibility.py`
- Modify: `book_mash/judges/redundancy.py`
- Modify: `book_mash/judges/registry.py`
- Modify: `book_mash/cache.py`
- Modify: `book_mash/output/annotations.py`
- Modify: `book_mash/output/report.py`
- Modify: `book_mash/rollups.py`
- Modify: `book_mash/runners/models.py`
- Modify: `book_mash/runners/measurement.py`
- Modify: `book_mash/runners/planner.py`
- Modify: `tests/test_cache.py`
- Modify: `tests/test_judges_smoke.py`
- Modify: `tests/test_output_annotations.py`
- Modify: `tests/test_output_report.py`
- Modify: `tests/test_rollups.py`
- Modify: `tests/test_runner_measurement.py`
- Modify: `tests/test_models.py`
- Modify: `tests/test_judge_base.py`

**Interfaces:**
- Consumes: `mash_core.{JudgeDim, JudgeInput, JudgeLabel, JudgeScore, JudgeResult, UnitType, build_judge_model, DEFAULT_JUDGE_MODEL_ID, JUDGE_MODEL_SETTINGS, estimate_cost, run_with_backoff}`.
- Produces: no new symbols — this task only repoints imports. `book_mash.judges.{base,models,_model_factory,_model_settings,_pricing,_retry}` are NOT deleted yet (Task 6 does that).

**Revision note (post-hoc, added after Task 5's first dispatch went BLOCKED):** the plan originally split source-import rewiring (this task) from test-file import rewiring (Task 6), on the assumption that leaving the old modules in place made both halves independently green. That assumption was wrong: `mash_core.JudgeScore` is a separate Pydantic class from `book_mash.judges.models.JudgeScore` (mash-core has its own copy, not a re-export), so once this task retypes `Run.scores: list[JudgeScore]` to the new class, any test still constructing `Run`/`JudgeScore` from the *old* module fails Pydantic's strict validation. Confirmed empirically (87 passed/3 deselected baseline → 80 passed/7 failed/3 deselected after only the 14 source-file edits, isolated to `test_output_report.py` and `test_runner_measurement.py`). Fix: the 8 test-file import-only edits (formerly Task 6 Steps 1-8, now this task's Steps 15-22 below) are pulled into this task, so the commit that retypes `Run.scores` also updates every test file that constructs against it — the plan's "suite must pass after every task" constraint holds again. Task 6 now only does deletions.

- [ ] **Step 1: `book_mash/judges/humanness.py`** — replace these 7 lines:

```python
from book_mash.judges.base import JudgeDim
from book_mash.judges.models import JudgeInput, JudgeLabel, JudgeScore
from book_mash.judges.registry import register_dim
from book_mash.judges._pricing import estimate_cost
from book_mash.judges._model_factory import DEFAULT_JUDGE_MODEL_ID, build_judge_model
from book_mash.judges._model_settings import JUDGE_MODEL_SETTINGS
from book_mash.judges._retry import run_with_backoff
```

with:

```python
from book_mash.judges.registry import register_dim
from mash_core import (
    DEFAULT_JUDGE_MODEL_ID,
    JUDGE_MODEL_SETTINGS,
    JudgeDim,
    JudgeInput,
    JudgeLabel,
    JudgeScore,
    build_judge_model,
    estimate_cost,
    run_with_backoff,
)
```

- [ ] **Step 2: `book_mash/judges/usefulness.py`** — replace these 7 lines:

```python
from book_mash.judges._model_factory import DEFAULT_JUDGE_MODEL_ID, build_judge_model
from book_mash.judges._model_settings import JUDGE_MODEL_SETTINGS
from book_mash.judges._pricing import estimate_cost
from book_mash.judges.base import JudgeDim
from book_mash.judges.models import JudgeInput, JudgeLabel, JudgeScore
from book_mash.judges.registry import register_dim
from book_mash.judges._retry import run_with_backoff
```

with:

```python
from book_mash.judges.registry import register_dim
from mash_core import (
    DEFAULT_JUDGE_MODEL_ID,
    JUDGE_MODEL_SETTINGS,
    JudgeDim,
    JudgeInput,
    JudgeLabel,
    JudgeScore,
    build_judge_model,
    estimate_cost,
    run_with_backoff,
)
```

- [ ] **Step 3: `book_mash/judges/voice.py`** — replace these 7 lines:

```python
from book_mash.judges._model_factory import DEFAULT_JUDGE_MODEL_ID, build_judge_model
from book_mash.judges._model_settings import JUDGE_MODEL_SETTINGS
from book_mash.judges._pricing import estimate_cost
from book_mash.judges.base import JudgeDim
from book_mash.judges.models import JudgeInput, JudgeLabel, JudgeScore
from book_mash.judges.registry import register_dim
from book_mash.judges._retry import run_with_backoff
```

with:

```python
from book_mash.judges.registry import register_dim
from mash_core import (
    DEFAULT_JUDGE_MODEL_ID,
    JUDGE_MODEL_SETTINGS,
    JudgeDim,
    JudgeInput,
    JudgeLabel,
    JudgeScore,
    build_judge_model,
    estimate_cost,
    run_with_backoff,
)
```

- [ ] **Step 4: `book_mash/judges/evidence_density.py`** — replace these 7 lines:

```python
from book_mash.judges._model_factory import DEFAULT_JUDGE_MODEL_ID, build_judge_model
from book_mash.judges._model_settings import JUDGE_MODEL_SETTINGS
from book_mash.judges._pricing import estimate_cost
from book_mash.judges.base import JudgeDim
from book_mash.judges.models import JudgeInput, JudgeLabel, JudgeScore
from book_mash.judges.registry import register_dim
from book_mash.judges._retry import run_with_backoff
```

with:

```python
from book_mash.judges.registry import register_dim
from mash_core import (
    DEFAULT_JUDGE_MODEL_ID,
    JUDGE_MODEL_SETTINGS,
    JudgeDim,
    JudgeInput,
    JudgeLabel,
    JudgeScore,
    build_judge_model,
    estimate_cost,
    run_with_backoff,
)
```

- [ ] **Step 5: `book_mash/judges/claim_defensibility.py`** — replace these 7 lines:

```python
from book_mash.judges._model_factory import DEFAULT_JUDGE_MODEL_ID, build_judge_model
from book_mash.judges._model_settings import JUDGE_MODEL_SETTINGS
from book_mash.judges._pricing import estimate_cost
from book_mash.judges.base import JudgeDim
from book_mash.judges.models import JudgeInput, JudgeLabel, JudgeScore
from book_mash.judges.registry import register_dim
from book_mash.judges._retry import run_with_backoff
```

with:

```python
from book_mash.judges.registry import register_dim
from mash_core import (
    DEFAULT_JUDGE_MODEL_ID,
    JUDGE_MODEL_SETTINGS,
    JudgeDim,
    JudgeInput,
    JudgeLabel,
    JudgeScore,
    build_judge_model,
    estimate_cost,
    run_with_backoff,
)
```

- [ ] **Step 6: `book_mash/judges/redundancy.py`** — replace the same 7 lines (leave `from book_mash.judges.embeddings import EmbeddingClient, cosine_similarity` untouched):

Before:
```python
from book_mash.judges._model_factory import DEFAULT_JUDGE_MODEL_ID, build_judge_model
from book_mash.judges._model_settings import JUDGE_MODEL_SETTINGS
from book_mash.judges._pricing import estimate_cost
from book_mash.judges.base import JudgeDim
from book_mash.judges.embeddings import EmbeddingClient, cosine_similarity
from book_mash.judges.models import JudgeInput, JudgeLabel, JudgeScore
from book_mash.judges.registry import register_dim
from book_mash.judges._retry import run_with_backoff
```

After:
```python
from book_mash.judges.embeddings import EmbeddingClient, cosine_similarity
from book_mash.judges.registry import register_dim
from mash_core import (
    DEFAULT_JUDGE_MODEL_ID,
    JUDGE_MODEL_SETTINGS,
    JudgeDim,
    JudgeInput,
    JudgeLabel,
    JudgeScore,
    build_judge_model,
    estimate_cost,
    run_with_backoff,
)
```

- [ ] **Step 7: `book_mash/judges/registry.py`** — replace:

```python
from book_mash.judges.base import JudgeDim
```

with:

```python
from mash_core import JudgeDim
```

- [ ] **Step 8: `book_mash/cache.py`** — replace:

```python
from book_mash.judges.models import JudgeScore
```

with:

```python
from mash_core import JudgeScore
```

- [ ] **Step 9: `book_mash/output/annotations.py`** — replace:

```python
from book_mash.judges.models import JudgeLabel, JudgeScore
```

with:

```python
from mash_core import JudgeLabel, JudgeScore
```

- [ ] **Step 10: `book_mash/output/report.py`** — replace:

```python
from book_mash.judges.models import JudgeLabel, JudgeScore
```

with:

```python
from mash_core import JudgeLabel, JudgeScore
```

- [ ] **Step 11: `book_mash/rollups.py`** — replace:

```python
from book_mash.judges.models import JudgeLabel, JudgeScore
```

with:

```python
from mash_core import JudgeLabel, JudgeScore
```

- [ ] **Step 12: `book_mash/runners/models.py`** — replace:

```python
from book_mash.judges.models import JudgeScore
```

with:

```python
from mash_core import JudgeScore
```

- [ ] **Step 13: `book_mash/runners/measurement.py`** — replace these two lines (keep all other `book_mash.judges.*` import lines in this file untouched — `claim_defensibility`, `embeddings`, `evidence_density`, `humanness`, `redundancy`, `registry`, `usefulness`, `voice` all stay in book-mash):

```python
from book_mash.judges.base import JudgeDim
from book_mash.judges.models import JudgeInput, JudgeLabel, JudgeScore
```

with a single new line, added anywhere alongside the other `book_mash.judges.*` imports (alphabetical position after the `book_mash.judges.*` block, before `book_mash.output.*` if present, is fine — this file's existing import order isn't alphabetized project-wide, so just keep it readable):

```python
from mash_core import JudgeDim, JudgeInput, JudgeLabel, JudgeScore
```

- [ ] **Step 14: `book_mash/runners/planner.py`** — replace the import line:

```python
from book_mash.judges import _pricing
```

with:

```python
from mash_core import estimate_cost
```

and update the one call site (currently `return _pricing.estimate_cost(model_id, in_tokens, out_tokens)`) to:

```python
return estimate_cost(model_id, in_tokens, out_tokens)
```

- [ ] **Step 15: `tests/test_cache.py`** — replace:

```python
from book_mash.judges.models import JudgeLabel, JudgeScore
```

with:

```python
from mash_core import JudgeLabel, JudgeScore
```

- [ ] **Step 16: `tests/test_judges_smoke.py`** — replace:

```python
from book_mash.judges.models import JudgeInput, JudgeLabel
```

with:

```python
from mash_core import JudgeInput, JudgeLabel
```

- [ ] **Step 17: `tests/test_output_annotations.py`** — replace:

```python
from book_mash.judges.models import JudgeLabel, JudgeScore
```

with:

```python
from mash_core import JudgeLabel, JudgeScore
```

- [ ] **Step 18: `tests/test_output_report.py`** — replace:

```python
from book_mash.judges.models import JudgeLabel, JudgeScore
```

with:

```python
from mash_core import JudgeLabel, JudgeScore
```

- [ ] **Step 19: `tests/test_rollups.py`** — replace:

```python
from book_mash.judges.models import JudgeLabel, JudgeScore
```

with:

```python
from mash_core import JudgeLabel, JudgeScore
```

- [ ] **Step 20: `tests/test_runner_measurement.py`** — replace:

```python
from book_mash.judges.models import JudgeLabel, JudgeScore
```

with:

```python
from mash_core import JudgeLabel, JudgeScore
```

- [ ] **Step 21: `tests/test_models.py`** — remove the now-unused import and the two tests that moved to mash-core in Task 1 Step 8. Before:

```python
from book_mash.corpus.models import Paragraph, Section, Chapter, Corpus
from book_mash.judges.models import JudgeScore, JudgeLabel, JudgeResult
from book_mash.runners.models import Run, RunStatus
```

After:

```python
from book_mash.corpus.models import Paragraph, Section, Chapter, Corpus
from book_mash.runners.models import Run, RunStatus
```

Then delete these two functions entirely from the file (they now live in `mash-core/tests/test_models.py`):

```python
def test_judge_score_label_enum():
    s = JudgeScore(
        dim_name="humanness",
        unit_id="paragraph:x",
        score_0_100=72.0,
        label=JudgeLabel.MODERATE,
        reasoning="Has a point of view but generic intro.",
        evidence_refs=["phrase:'in today's evolving landscape'"],
        model="claude-sonnet-4-6",
        cost_usd=0.011,
        derived=False,
    )
    assert s.label == JudgeLabel.MODERATE
    assert s.score_0_100 == 72.0


def test_judge_score_error_allows_null_score():
    s = JudgeScore(
        dim_name="humanness",
        unit_id="paragraph:x",
        score_0_100=None,
        label=JudgeLabel.ERROR,
        reasoning="API timeout after 3 retries",
        evidence_refs=[],
        model="claude-sonnet-4-6",
        cost_usd=0.0,
        derived=False,
    )
    assert s.score_0_100 is None
    assert s.label == JudgeLabel.ERROR
```

The file should retain `test_paragraph_construction`, `test_chapter_aggregates_text`, and `test_run_status_enum` unchanged.

- [ ] **Step 22: `tests/test_judge_base.py`** — remove the now-unused imports and the `FakeJudge`/`test_judge_returns_score` pair that moved to mash-core in Task 1 Step 9. Before:

```python
import pytest

from book_mash.corpus.models import ClaimEntry
from book_mash.judges.base import JudgeDim
from book_mash.judges.claim_defensibility import ClaimDefensibilityJudge
from book_mash.judges.models import JudgeInput, JudgeLabel, JudgeScore
from book_mash.judges.registry import dim_registry, DIM_REGISTRY_VERSION


class FakeJudge(JudgeDim):
    name = "fake"
    unit_type = "paragraph"
    model_id = "test-model"

    async def judge(self, input: JudgeInput) -> JudgeScore:
        return JudgeScore(
            dim_name=self.name,
            unit_id=input.unit_id,
            score_0_100=75.0,
            label=JudgeLabel.MODERATE,
            reasoning="fake reasoning",
            evidence_refs=[],
            model=self.model_id,
            cost_usd=0.0,
            derived=False,
        )


async def test_judge_returns_score():
    judge = FakeJudge()
    input = JudgeInput(
        unit_id="paragraph:x",
        unit_type="paragraph",
        unit_text="hello",
        dim_name="fake",
    )
    score = await judge.judge(input)
    assert score.score_0_100 == 75.0
    assert score.label == JudgeLabel.MODERATE
```

After:

```python
import pytest

from book_mash.corpus.models import ClaimEntry
from book_mash.judges.claim_defensibility import ClaimDefensibilityJudge
from book_mash.judges.registry import dim_registry, DIM_REGISTRY_VERSION
from mash_core import JudgeInput
```

The rest of the file (`test_registry_version_is_string`, `test_registry_starts_empty`, the `claim_defensibility_judge` fixture, `_make_input`, `_claim`, and the four `test_context_cache_key_*` tests) is unchanged — `JudgeInput` is still used by `_make_input`.

- [ ] **Step 23: Sanity-check nothing still points at the old private modules**

```bash
cd /Users/timur_isachenko/Dev/LifeOS/book-mash
grep -rn "book_mash\.judges\.\(base\|models\|_model_factory\|_model_settings\|_pricing\|_retry\)" book_mash/
```

Expected: exactly one hit — `book_mash/judges/base.py` importing `book_mash/judges/models.py` (old-file-to-old-file internal wiring; both deleted wholesale in Task 6). No other file should reference the old modules. If any other file shows up, it's a miss — fix it before continuing.

- [ ] **Step 24: Run the full suite and commit**

```bash
poetry run pytest
```

Expected: fully green, 0 failures — 84 passed / 3 deselected. (Not 87: Steps 21-22 delete the 3 tests that moved to mash-core in Task 1, so 87 baseline minus 3 moved = 84. The point of pulling the test-file edits into this task is 0 *failures*, not an unchanged count.)

```bash
git add book_mash/ tests/
git commit -m "refactor: import judge-core types from mash_core instead of book_mash.judges"
```

---

### Task 6: Delete the extracted files

**Revision note:** this task originally also moved/trimmed 8 test files' imports; those steps were pulled into Task 5 (see Task 5's revision note) to keep the suite green after every task. This task is now deletions only — the old modules and their now-superseded test files, everything already re-pointed by Task 5.

**Files:**
- Delete: `tests/test_model_factory.py` (moved to mash-core in Task 2)
- Delete: `tests/test_retry.py` (moved to mash-core in Task 2)
- Delete: `book_mash/judges/base.py`
- Delete: `book_mash/judges/models.py`
- Delete: `book_mash/judges/_model_factory.py`
- Delete: `book_mash/judges/_model_settings.py`
- Delete: `book_mash/judges/_pricing.py`
- Delete: `book_mash/judges/_retry.py`

- [ ] **Step 1: Delete the moved test files**

```bash
cd /Users/timur_isachenko/Dev/LifeOS/book-mash
git rm tests/test_model_factory.py tests/test_retry.py
```

- [ ] **Step 2: Delete the extracted source files**

```bash
git rm book_mash/judges/base.py book_mash/judges/models.py \
       book_mash/judges/_model_factory.py book_mash/judges/_model_settings.py \
       book_mash/judges/_pricing.py book_mash/judges/_retry.py
```

- [ ] **Step 3: Run the full verification suite**

```bash
poetry run pytest
poetry run mypy book_mash
poetry run ruff check book_mash tests
```

Expected: all green — same pass count as before this plan started (minus the tests that moved to mash-core, which now run as part of mash-core's own suite instead).

- [ ] **Step 4: Commit**

```bash
git add tests/ book_mash/
git commit -m "chore: delete judge-core files now that book_mash imports from mash_core"
```

---

### Task 7: Final cross-repo verification

**Files:** none (verification only)

- [ ] **Step 1: Run mash-core's full suite one more time (independent of book-mash)**

```bash
cd /Users/timur_isachenko/Dev/LifeOS/mash-core
poetry run pytest && poetry run mypy mash_core && poetry run ruff check mash_core tests
```

Expected: all green, no dependency on book-mash's presence.

- [ ] **Step 2: Run book-mash's full suite one more time**

```bash
cd /Users/timur_isachenko/Dev/LifeOS/book-mash
poetry run pytest && poetry run mypy book_mash && poetry run ruff check book_mash tests
```

Expected: all green.

- [ ] **Step 3: Confirm the pytest count matches expectations**

The book-mash suite should have exactly 2 fewer test files (`test_model_factory.py`, `test_retry.py` removed) and 3 fewer test functions in `test_models.py`/`test_judge_base.py` (the two moved `JudgeScore` tests + the one moved `FakeJudge` test) than it had before Task 1. Those same tests should now appear, passing, in mash-core's suite. No test coverage was dropped — it moved.

- [ ] **Step 4: No commit for this task** — it's verification-only. If anything failed, go back to the task that introduced the failure, fix it there, and re-commit that task (don't paper over it with a new fixup commit at the end).
