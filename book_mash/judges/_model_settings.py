"""Shared model settings for every judge Agent.

Two reliability properties live here so all six dimensions (humanness, voice,
usefulness, evidence_density, claim_defensibility, redundancy) get them
identically — there is one definition, not six copies that can drift apart.

1. ``temperature=0.0`` — the design doc promises temperature=0 / reproducibility,
   but no judge was setting a temperature, so calls ran at the Anthropic provider
   default (~1.0) and were non-reproducible (a re-judged paragraph swung 5->82
   between runs; see docs/judge-module-evaluation.md, cross-cutting issue X1 and
   recommendation P1). temperature=0 makes cold re-runs near-reproducible as the
   contract intended. (Anthropic does not guarantee *bit*-exact determinism even
   at 0, but it removes the dominant sampling-variance source.)

2. ``timeout`` — a per-request httpx timeout. A cold run once hung for 75 minutes
   on a single stalled HTTPS request to the Anthropic API because there was no
   client-side timeout; one stalled connection blocked the whole run. A bounded
   timeout makes a stalled call fail fast. A timed-out request raises an
   APITimeout-class error, which ``run_with_backoff`` treats as retryable
   (bounded ``max_attempts``); if it keeps timing out the exception re-raises and
   the judge's existing handler records a clean ERROR-labeled score instead of
   hanging forever.

pydantic-ai 0.0.40: ``ModelSettings`` is a TypedDict; ``AnthropicModel`` reads
``temperature`` and ``timeout`` from it and passes them straight into the
Anthropic SDK ``messages.create`` call (verified in
pydantic_ai/models/anthropic.py). Pass it to ``Agent(model_settings=...)``.
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
