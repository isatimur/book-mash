"""Configurable judge-model factory — one place all six judges get their model.

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
The factory returns BOTH the model object AND a stable model-id STRING. That string
is what each judge exposes as ``model_id``; the runner threads it into the cache key
(``cache.py``: ``unit_hash | dim_name | dim_version | model_id``) and into the
per-result ``JudgeScore.model`` field. Non-Anthropic providers get a
provider-prefixed id (e.g. ``openrouter:deepseek/deepseek-chat``) so a provider
switch ALWAYS busts the cache — an Anthropic-keyed cached score is never reused for
an OpenRouter run, and vice-versa, even in the unlikely case two providers expose
the same bare model slug.

pydantic-ai 0.0.40 API (verified against the installed version)
--------------------------------------------------------------
``poetry run python -c "from pydantic_ai.models.openai import OpenAIModel; import
inspect; print(inspect.signature(OpenAIModel.__init__))"`` reports::

    OpenAIModel.__init__(self, model_name, *, provider='openai' | Provider | None,
                         base_url=None, api_key=None, openai_client=None, ...)

So an OpenAI-compatible endpoint is constructed directly via
``OpenAIModel(model_name, base_url=..., api_key=...)`` (no separate provider object
required). pydantic-ai requests structured output for ``result_type`` via the
OpenAI tool/JSON path, so the judges' pydantic ``result_type`` works unchanged.

Extensibility — future multi-model panel
----------------------------------------
The judges are model-agnostic: each calls :func:`build_judge_model` in ``__init__``
and stores whatever ``(model, model_id)`` it gets. To add an N-model panel later
(run N models per unit, take the median score), introduce a sibling factory that
returns a list of ``(model, model_id)`` pairs and have the runner fan a unit out
across them and aggregate — no change to the six judge classes' contract is needed,
because they already treat the model + id as injected values rather than constants.
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
