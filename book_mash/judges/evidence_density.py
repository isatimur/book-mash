import os

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel

from book_mash.corpus.models import ClaimEntry
from book_mash.judges._pricing import estimate_cost
from book_mash.judges.base import JudgeDim
from book_mash.judges.models import JudgeInput, JudgeLabel, JudgeScore
from book_mash.judges.registry import register_dim
from book_mash.judges._retry import run_with_backoff


class _CandidateClaim(BaseModel):
    text: str  # the exact claim made in the prose
    closest_ledger_id: str | None  # null if no plausible match


class _EvidenceDensityOutput(BaseModel):
    candidate_claims: list[_CandidateClaim]


_SYSTEM_PROMPT = """\
You extract distinct factual claims from a section of book prose and match each to an entry
in a claims ledger (or to null if no match).

A "claim" is a falsifiable statement about how the world is - not opinions, transitions, or
rhetorical framing.

For each claim:
- write the exact claim from the prose
- match it to the closest ledger entry id, or null if there is no plausible match

Be conservative. Better to miss a claim than to invent one.
"""


def _build_agent() -> Agent[None, _EvidenceDensityOutput]:
    return Agent(
        model=AnthropicModel("claude-sonnet-4-6", api_key=os.environ["ANTHROPIC_API_KEY"]),
        system_prompt=_SYSTEM_PROMPT,
        result_type=_EvidenceDensityOutput,
        result_retries=2,
    )


def _label_for_density(claims_count: int, word_count: int) -> str:
    if claims_count == 0 or word_count == 0:
        return "fail"
    words_per_claim = word_count / claims_count
    if words_per_claim <= 300:
        return "strong"
    if words_per_claim <= 700:
        return "moderate"
    if words_per_claim <= 1000:
        return "weak"
    return "fail"


def _score_for_density(claims_count: int, word_count: int) -> float:
    label = _label_for_density(claims_count, word_count)
    return {"strong": 90.0, "moderate": 65.0, "weak": 35.0, "fail": 10.0}[label]


@register_dim
class EvidenceDensityJudge(JudgeDim):
    name = "evidence_density"
    unit_type = "section"
    model_id = "claude-sonnet-4-6"

    def __init__(self):
        self._agent = _build_agent()

    async def judge(self, input: JudgeInput) -> JudgeScore:
        claims_index: list[ClaimEntry] = input.context.get("claims_index", [])
        ledger_listing = "\n".join(f"- {c.id}: {c.text}" for c in claims_index)
        prompt = (
            f"Claims ledger:\n{ledger_listing}\n\n"
            f"Section prose:\n{input.unit_text}"
        )
        try:
            result = await run_with_backoff(lambda: self._agent.run(prompt))
            out: _EvidenceDensityOutput = result.data
            grounded = [c for c in out.candidate_claims if c.closest_ledger_id]
            ungrounded = [c for c in out.candidate_claims if not c.closest_ledger_id]
            word_count = len(input.unit_text.split())
            label = _label_for_density(len(grounded), word_count)
            score = _score_for_density(len(grounded), word_count)
            usage = result.usage()
            cost = estimate_cost(self.model_id, usage.request_tokens, usage.response_tokens)
            refs = (
                [f"grounded:{c.closest_ledger_id}" for c in grounded]
                + [f"ungrounded:{c.text[:80]}" for c in ungrounded]
            )
            return JudgeScore(
                dim_name=self.name,
                unit_id=input.unit_id,
                score_0_100=score,
                label=JudgeLabel(label),
                reasoning=(
                    f"{len(grounded)} grounded claims / {word_count} words "
                    f"({word_count // max(len(grounded), 1)} words per grounded claim)"
                ),
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
