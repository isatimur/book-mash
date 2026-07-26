from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent

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


class _UsefulnessOutput(BaseModel):
    score_0_100: float
    label: Literal["strong", "moderate", "weak", "fail"]
    reasoning: str
    actionable_takeaway: str  # what a reader could change on Monday - empty string if none


_SYSTEM_PROMPT = """\
You are a senior practitioner reviewing a book on AI engineering with one question in mind:
"could a working engineer change something on Monday because of this paragraph?"

High usefulness:
- contains a decision, threshold, principle, or concrete example a reader can apply
- gives a quantitative or qualitative test ("X is the wrong choice when Y")
- names a specific tool, technique, or trap with enough detail to act on

Low usefulness:
- describes a landscape without commitments
- restates a prior chapter's point
- transition prose, meta-commentary, "in this chapter we will..."
- decorative metaphor with no operational lift

Rubric:
- strong (80-100): contains a decision, threshold, principle, or concrete example a reader can apply
- moderate (50-79): provides framing useful for later application
- weak (20-49): descriptive only, no actionable lift
- fail (0-19): filler - meta-talk about the book, transitions, restatements

If the score is moderate or above, write the actionable takeaway in one sentence in `actionable_takeaway`.
If weak or fail, leave `actionable_takeaway` empty.
"""


def _build_agent() -> tuple[Agent[None, _UsefulnessOutput], str]:
    # Model is provider-configurable via mash_core.model_factory (Anthropic default,
    # OpenRouter/OpenAI-compatible opt-in). Factory returns the model object AND
    # the stable model-id string that flows into the cache key + JudgeScore.model.
    model, model_id = build_judge_model()
    return (
        Agent(
            model=model,
            system_prompt=_SYSTEM_PROMPT,
            result_type=_UsefulnessOutput,
            result_retries=2,
            model_settings=JUDGE_MODEL_SETTINGS,
        ),
        model_id,
    )


@register_dim
class UsefulnessJudge(JudgeDim):
    name = "usefulness"
    unit_type = "paragraph"
    # ClassVar default for back-compat (planner introspection, tests); the
    # instance attribute set in __init__ is the live value the runner reads.
    model_id = DEFAULT_JUDGE_MODEL_ID

    def __init__(self):
        self._agent, self.model_id = _build_agent()

    async def judge(self, input: JudgeInput) -> JudgeScore:
        chapter_title = input.context.get("chapter_title", "")
        prompt = f"Chapter: {chapter_title}\n\nParagraph:\n{input.unit_text}"
        try:
            result = await run_with_backoff(lambda: self._agent.run(prompt))
            out: _UsefulnessOutput = result.data
            usage = result.usage()
            cost = estimate_cost(self.model_id, usage.request_tokens, usage.response_tokens)
            refs = [f"takeaway:{out.actionable_takeaway}"] if out.actionable_takeaway else []
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
