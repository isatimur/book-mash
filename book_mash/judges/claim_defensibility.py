import hashlib
import json
import os
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel

from book_mash.corpus.models import ClaimEntry
from book_mash.judges._pricing import estimate_cost
from book_mash.judges.base import JudgeDim
from book_mash.judges.models import JudgeInput, JudgeLabel, JudgeScore
from book_mash.judges.registry import register_dim
from book_mash.judges._retry import run_with_backoff


class _ClaimDefensibilityOutput(BaseModel):
    score_0_100: float
    label: Literal["strong", "moderate", "weak", "fail"]
    reasoning: str
    overstated_claims: list[str]  # claim phrases that overstate ledger evidence
    unsupported_claims: list[str]  # claim phrases with no ledger backing at all


_SYSTEM_PROMPT = """\
You evaluate whether the claims made in a paragraph of book prose are defensible
against the claims ledger that the book maintains.

You receive:
- the paragraph
- the relevant ledger entries (with their declared strength: strong / moderate / weak)

For each claim made in the prose:
- compare its rhetorical strength to the ledger entry's strength
- prose at or below ledger strength = OK
- prose above ledger strength = overstated
- prose with no matching ledger entry = unsupported

Rubric:
- strong (80-100): all claims at or below ledger strength
- moderate (50-79): slight overstatement on a minor claim
- weak (20-49): central claim overstates ledger evidence
- fail (0-19): the paragraph makes a claim with no ledger backing at all

The `fail` label is a ship-blocker. Use it whenever there is a fabricated or unsupported claim
of any kind. False positives here are tolerable; false negatives are not.

Report all overstated and unsupported claims explicitly.
"""


def _build_agent() -> Agent[None, _ClaimDefensibilityOutput]:
    return Agent(
        model=AnthropicModel("claude-sonnet-4-6", api_key=os.environ["ANTHROPIC_API_KEY"]),
        system_prompt=_SYSTEM_PROMPT,
        result_type=_ClaimDefensibilityOutput,
        result_retries=2,
    )


@register_dim
class ClaimDefensibilityJudge(JudgeDim):
    name = "claim_defensibility"
    unit_type = "paragraph"
    model_id = "claude-sonnet-4-6"

    def __init__(self):
        self._agent = _build_agent()

    def context_cache_key(self, input: JudgeInput) -> str:
        ledger: list[ClaimEntry] = input.context.get("relevant_ledger", [])
        serialized = json.dumps(
            [{"id": c.id, "text": c.text, "support_level": c.support_level} for c in ledger],
            sort_keys=True,
        )
        return hashlib.sha256(serialized.encode()).hexdigest()

    async def judge(self, input: JudgeInput) -> JudgeScore:
        ledger: list[ClaimEntry] = input.context.get("relevant_ledger", [])
        ledger_block = "\n".join(
            f"- {c.id} (support_level: {c.support_level}): {c.text}" for c in ledger
        )
        prompt = (
            f"Relevant ledger entries:\n{ledger_block or '(none)'}\n\n"
            f"Paragraph to evaluate:\n{input.unit_text}"
        )
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
