"""Tests for the configurable judge-model factory (book_mash/judges/_model_factory.py).

No network calls: pydantic-ai constructs the model object (and its SDK client)
lazily without issuing a request, so we can assert on the model type / model_name /
base_url with only a dummy API key in the environment.

The cache-key contract is the load-bearing property under test: the model-id STRING
returned by the factory is what flows into cache.py's key and JudgeScore.model, so a
provider switch must produce a DIFFERENT string (otherwise an Anthropic-keyed cached
score would be wrongly reused for an OpenRouter run, and vice-versa).
"""

import pytest

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel

from book_mash.judges._model_factory import (
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
    string, so existing cache keys and JudgeScore.model values are unchanged."""
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
    """(c) The model-id string differs between providers, so cache.py keys differ and
    a provider switch cannot reuse the other provider's cached scores."""
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
