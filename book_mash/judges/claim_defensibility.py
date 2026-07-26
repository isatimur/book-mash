import hashlib
import json
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent

from book_mash.corpus.models import ClaimEntry
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


class _ClaimDefensibilityOutput(BaseModel):
    score_0_100: float
    label: Literal["strong", "moderate", "weak", "fail"]
    reasoning: str
    overstated_claims: list[str]  # claim phrases that overstate ledger evidence
    unsupported_claims: list[str]  # claim phrases with no ledger backing at all


# Bump this string whenever the prompt, the assembled ledger context, or the
# retrieval feeding it changes, so cached claim_defensibility scores are
# invalidated WITHOUT disturbing any other dimension's cache. It is folded into
# this judge's context_cache_key (which is part of the per-unit cache hash), so
# the bump is scoped to claim_defensibility only.
CLAIM_DEFENSIBILITY_PROMPT_VERSION = "2"


_SYSTEM_PROMPT = """\
You evaluate whether the claims made in a paragraph of book prose are defensible
against the claims ledger that the book maintains.

You receive:
- the paragraph
- the relevant ledger entries. Each entry includes its declared support strength
  (strong / moderate / weak), the claim statement, and its supporting quotes and
  sources (verbatim quotes from named speakers/sources backing the claim).

For each claim made in the prose:
- match it to a ledger entry using BOTH the claim statement AND its supporting
  quotes/sources. Prose may echo a supporting quote (e.g. "bless one platform",
  "root of trust") rather than the claim statement's wording — treat a match
  against a supporting quote or source as valid ledger backing.
- compare its rhetorical strength to the matched ledger entry's strength
- prose at or below ledger strength = OK
- prose above ledger strength = overstated
- prose with no matching ledger entry (statement or supporting quote) = unsupported

Rubric:
- strong (80-100): all claims at or below ledger strength
- moderate (50-79): slight overstatement on a minor claim
- weak (20-49): central claim overstates ledger evidence
- fail (0-19): the paragraph makes a claim with no ledger backing at all

The `fail` label is a ship-blocker. Use it whenever there is a fabricated or unsupported claim
of any kind. Before flagging a claim as unsupported, confirm it is backed by
neither a claim statement nor any supporting quote in the provided ledger entries.

Report all overstated and unsupported claims explicitly.
"""


def _format_claim(c: ClaimEntry) -> str:
    """Render one ledger entry for the judge prompt: statement, support level, and
    its supporting quotes/sources so prose can be matched against evidence."""
    lines = [f"- {c.id} (support_level: {c.support_level}): {c.text}"]
    for src, quote in zip(c.source_descriptions, c.quotes):
        lines.append(f"    - source {src}: \"{quote}\"")
    # If the source/quote lists are uneven, surface any remaining bare quotes too.
    for quote in c.quotes[len(c.source_descriptions):]:
        lines.append(f"    - supporting quote: \"{quote}\"")
    if c.reusable_phrasing:
        lines.append(f"    - reusable phrasing: {c.reusable_phrasing}")
    return "\n".join(lines)


def build_prompt(unit_text: str, ledger: list[ClaimEntry]) -> str:
    """Assemble the full claim_defensibility user prompt. Exposed for testing the
    retrieval/prompt-assembly path without an LLM call."""
    ledger_block = "\n".join(_format_claim(c) for c in ledger)
    return (
        f"Relevant ledger entries:\n{ledger_block or '(none)'}\n\n"
        f"Paragraph to evaluate:\n{unit_text}"
    )


def _build_agent() -> tuple[Agent[None, _ClaimDefensibilityOutput], str]:
    # Model is provider-configurable via _model_factory (Anthropic default,
    # OpenRouter/OpenAI-compatible opt-in). Factory returns the model object AND
    # the stable model-id string that flows into the cache key + JudgeScore.model.
    model, model_id = build_judge_model()
    return (
        Agent(
            model=model,
            system_prompt=_SYSTEM_PROMPT,
            result_type=_ClaimDefensibilityOutput,
            result_retries=2,
            model_settings=JUDGE_MODEL_SETTINGS,
        ),
        model_id,
    )


@register_dim
class ClaimDefensibilityJudge(JudgeDim):
    name = "claim_defensibility"
    unit_type = "paragraph"
    # ClassVar default for back-compat (planner introspection, tests); the
    # instance attribute set in __init__ is the live value the runner reads.
    model_id = DEFAULT_JUDGE_MODEL_ID

    def __init__(self):
        self._agent, self.model_id = _build_agent()

    def context_cache_key(self, input: JudgeInput) -> str:
        ledger: list[ClaimEntry] = input.context.get("relevant_ledger", [])
        serialized = json.dumps(
            {
                "v": CLAIM_DEFENSIBILITY_PROMPT_VERSION,
                "claims": [
                    {
                        "id": c.id,
                        "text": c.text,
                        "support_level": c.support_level,
                        "quotes": c.quotes,
                        "source_descriptions": c.source_descriptions,
                    }
                    for c in ledger
                ],
            },
            sort_keys=True,
        )
        return hashlib.sha256(serialized.encode()).hexdigest()

    async def judge(self, input: JudgeInput) -> JudgeScore:
        ledger: list[ClaimEntry] = input.context.get("relevant_ledger", [])
        prompt = build_prompt(input.unit_text, ledger)
        try:
            result = await run_with_backoff(lambda: self._agent.run(prompt))
            out: _ClaimDefensibilityOutput = result.data
            usage = result.usage()
            cost = estimate_cost(self.model_id, usage.request_tokens, usage.response_tokens)
            refs = [f"overstated:{c}" for c in out.overstated_claims] + [
                f"unsupported:{c}" for c in out.unsupported_claims
            ]
            return JudgeScore(
                dim_name=self.name,
                unit_id=input.unit_id,
                score_0_100=out.score_0_100,
                label=JudgeLabel(out.label),
                reasoning=out.reasoning,
                evidence_refs=refs,
                model=self.model_id,
                cost_usd=cost,
                derived=False,
            )
        except Exception as e:
            return JudgeScore(
                dim_name=self.name,
                unit_id=input.unit_id,
                score_0_100=None,
                label=JudgeLabel.ERROR,
                reasoning=f"{type(e).__name__}: {e}",
                evidence_refs=[],
                model=self.model_id,
                cost_usd=0.0,
                derived=False,
            )
