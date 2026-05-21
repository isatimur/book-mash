# USD per token, per model. Revise when Anthropic pricing changes.
_PRICE_PER_TOKEN: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
    "claude-haiku-4-5": {"input": 1.0 / 1_000_000, "output": 5.0 / 1_000_000},
}


def estimate_cost(model_id: str, input_tokens: int | None, output_tokens: int | None) -> float:
    rates = _PRICE_PER_TOKEN.get(model_id)
    if rates is None:
        return 0.0
    return (input_tokens or 0) * rates["input"] + (output_tokens or 0) * rates["output"]
