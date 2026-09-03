
import hashlib
import json

from pydantic import BaseModel
from pydantic_ai import Agent

from book_mash.corpus.models import ClaimEntry
from book_mash.judges.registry import register_dim
from mash_core import (
    audited_agent_run,
    DEFAULT_JUDGE_MODEL_ID,
    JUDGE_MODEL_SETTINGS,
    JudgeDim,
    JudgeInput,
    JudgeLabel,
    JudgeScore,
    build_judge_model,
    estimate_cost,
)


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


def _build_agent() -> tuple[Agent[None, _EvidenceDensityOutput], str]:
    # Model is provider-configurable via mash_core.model_factory (Anthropic default,
    # OpenRouter/OpenAI-compatible opt-in). Factory returns the model object AND
    # the stable model-id string that flows into the cache key + JudgeScore.model.
    model, model_id = build_judge_model()
    return (
        Agent(
            model=model,
            system_prompt=_SYSTEM_PROMPT,
            result_type=_EvidenceDensityOutput,
            result_retries=2,
            model_settings=JUDGE_MODEL_SETTINGS,
        ),
        model_id,
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
    # ClassVar default for back-compat (planner introspection, tests); the
    # instance attribute set in __init__ is the live value the runner reads.
    model_id = DEFAULT_JUDGE_MODEL_ID

    def __init__(self):
        self._agent, self.model_id = _build_agent()

    def context_cache_key(self, input: JudgeInput) -> str:
        # The score is a function of the section text AND the ledger it is matched
        # against. Without the ledger in the key, a ledger change never invalidates a
        # cached section score: on 2026-09-03 fourteen new entries written to rescue
        # ten zero-claim sections were never seen by the judge, because every one of
        # those sections replayed its previous score from the cache.
        claims_index: list[ClaimEntry] = input.context.get("claims_index", [])
        serialized = json.dumps([{"id": c.id, "text": c.text} for c in claims_index], sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    async def judge(self, input: JudgeInput) -> JudgeScore:
        claims_index: list[ClaimEntry] = input.context.get("claims_index", [])
        ledger_listing = "\n".join(f"- {c.id}: {c.text}" for c in claims_index)
        prompt = (
            f"Claims ledger:\n{ledger_listing}\n\n"
            f"Section prose:\n{input.unit_text}"
        )
        try:
            result = await audited_agent_run(self._agent, prompt, model_id=self.model_id)
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
