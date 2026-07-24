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
