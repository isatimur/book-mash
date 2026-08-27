from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent

from book_mash.judges.embeddings import EmbeddingClient, cosine_similarity
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


class _RedundancyOutput(BaseModel):
    score_0_100: float
    label: Literal["strong", "moderate", "weak", "fail"]
    reasoning: str
    overlapping_arguments: list[str]  # arguments that restate something from an earlier chapter
    overlap_chapter_ids: list[str]    # the earlier chapter ids that overlap


_SYSTEM_PROMPT = """\
You judge whether a chapter restates arguments already made in earlier chapters.

You receive:
- the chapter to evaluate
- summary digests of earlier chapters in declared order
- a similarity prefilter has already flagged candidate earlier chapters; focus on those

Arguments that *deepen* a prior chapter's claim with new specifics are OK.
Arguments that *restate* a prior chapter without adding anything are redundancy.

Rubric:
- strong (80-100): each major argument is new or genuinely deepens a prior one
- moderate (50-79): one or two restatements of minor points
- weak (20-49): central argument has already been made
- fail (0-19): >40% of the chapter is restatement

List the specific overlapping arguments and which earlier chapters they overlap with.
"""


def _build_agent() -> tuple[Agent[None, _RedundancyOutput], str]:
    # Model is provider-configurable via mash_core.model_factory (Anthropic default,
    # OpenRouter/OpenAI-compatible opt-in). Factory returns the model object AND
    # the stable model-id string that flows into the cache key + JudgeScore.model.
    model, model_id = build_judge_model()
    return (
        Agent(
            model=model,
            system_prompt=_SYSTEM_PROMPT,
            result_type=_RedundancyOutput,
            result_retries=2,
            model_settings=JUDGE_MODEL_SETTINGS,
        ),
        model_id,
    )


@register_dim
class RedundancyJudge(JudgeDim):
    name = "redundancy"
    unit_type = "chapter"
    # ClassVar default for back-compat (planner introspection, tests); the
    # instance attribute set in __init__ is the live value the runner reads.
    model_id = DEFAULT_JUDGE_MODEL_ID

    def __init__(self):
        self._agent, self.model_id = _build_agent()

    async def judge(self, input: JudgeInput) -> JudgeScore:
        earlier_summaries: list[dict] = input.context.get("earlier_chapter_summaries", [])
        candidate_ids: list[str] = input.context.get("candidate_overlap_chapter_ids", [])
        if not earlier_summaries:
            return JudgeScore(
                dim_name=self.name,
                unit_id=input.unit_id,
                score_0_100=100.0,
                label=JudgeLabel.STRONG,
                reasoning="No earlier chapters to compare against.",
                evidence_refs=[],
                model=self.model_id,
                cost_usd=0.0,
                derived=False,
            )
        candidates_block = "\n\n".join(
            f"Chapter {s['id']} (similarity prefilter: {'flagged' if s['id'] in candidate_ids else 'not flagged'}):\n"
            f"{s['summary']}"
            for s in earlier_summaries
        )
        prompt = (
            f"Earlier chapters:\n{candidates_block}\n\n"
            f"Chapter to evaluate:\n{input.unit_text}"
        )
        try:
            result = await audited_agent_run(self._agent, prompt, model_id=self.model_id)
            out: _RedundancyOutput = result.data
            usage = result.usage()
            cost = estimate_cost(self.model_id, usage.request_tokens, usage.response_tokens)
            refs = [f"overlaps:{cid}" for cid in out.overlap_chapter_ids] + [
                f"argument:{a}" for a in out.overlapping_arguments
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


async def prefilter_candidates(
    embedding_client: EmbeddingClient,
    chapter_summary: str,
    earlier_summaries: list[dict],
    threshold: float = 0.75,
) -> list[str]:
    """Returns chapter ids whose cosine similarity to `chapter_summary` exceeds `threshold`."""
    if not earlier_summaries:
        return []
    texts = [chapter_summary] + [s["summary"] for s in earlier_summaries]
    embs = await embedding_client.embed(texts)
    target = embs[0]
    return [
        earlier_summaries[i]["id"]
        for i in range(len(earlier_summaries))
        if cosine_similarity(target, embs[i + 1]) >= threshold
    ]
