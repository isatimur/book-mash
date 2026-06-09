import os
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel

from book_mash.judges.base import JudgeDim
from book_mash.judges.models import JudgeInput, JudgeLabel, JudgeScore
from book_mash.judges.registry import register_dim
from book_mash.judges._pricing import estimate_cost
from book_mash.judges._retry import run_with_backoff


class _HumannessOutput(BaseModel):
    score_0_100: float
    label: Literal["strong", "moderate", "weak", "fail"]
    reasoning: str
    worst_phrase: str  # the worst phrase quoted from the paragraph


_SYSTEM_PROMPT = """\
You are a senior book editor with a hostile red pen and a low tolerance for AI-flavored prose.

You score a single paragraph on HUMANNESS — does this sound like a thinking author with a point of view, or is it generic AI-flavored hedging that could appear in any blog post?

Indicators of low humanness (AI slop):
- "in today's rapidly evolving landscape"
- "by leveraging X, you can unlock Y"
- list-of-three patterns with no real distinction between items
- empty hedging ("it's worth noting", "it's important to remember")
- listicle stems with no point of view
- smooth transitions that hide the absence of a claim

Indicators of high humanness:
- specific numbers, names, prices, dates
- a claim that could be wrong (something to disagree with)
- a point of view that survives a hostile editor
- prose that loses something concrete if removed

Rubric:
- strong (80-100): specific, has a point of view, survives a hostile editor's red pen
- moderate (50-79): readable but generic in spots
- weak (20-49): pattern-matchable AI prose; removing it loses nothing
- fail (0-19): pure scaffolding language, listicle stems, empty hedging

REQUIRED: you must quote the worst phrase from the paragraph in `worst_phrase`.
Even a "strong" score must name the weakest moment.
"""


def _build_agent() -> Agent:
    # pydantic-ai 0.0.40: AnthropicModel takes api_key= directly (no AnthropicProvider);
    # system_prompt= (not instructions=); result_type= is correct; result_retries= (not output_retries=)
    return Agent(
        model=AnthropicModel(
            "claude-sonnet-4-6",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        ),
        system_prompt=_SYSTEM_PROMPT,
        result_type=_HumannessOutput,
        result_retries=2,
    )


@register_dim
class HumannessJudge(JudgeDim):
    name = "humanness"
    unit_type = "paragraph"
    model_id = "claude-sonnet-4-6"

    def __init__(self):
        self._agent = _build_agent()

    async def judge(self, input: JudgeInput) -> JudgeScore:
        surrounding = input.context.get("surrounding_paragraphs", [])
        user_prompt = _format_user_prompt(input.unit_text, surrounding)
        try:
            result = await run_with_backoff(lambda: self._agent.run(user_prompt))
            out: _HumannessOutput = result.data
            # pydantic-ai 0.0.40 Usage: request_tokens / response_tokens (not input_/output_tokens)
            usage = result.usage()
            cost = estimate_cost(self.model_id, usage.request_tokens, usage.response_tokens)
            return JudgeScore(
                dim_name=self.name,
                unit_id=input.unit_id,
                score_0_100=out.score_0_100,
                label=JudgeLabel(out.label),
                reasoning=out.reasoning,
                evidence_refs=[f"phrase:{out.worst_phrase!r}"],
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


def _format_user_prompt(paragraph_text: str, surrounding: list[str]) -> str:
    parts = []
    if surrounding:
        parts.append("Surrounding paragraphs for context:")
        for s in surrounding:
            parts.append(f"  - {s[:300]}{'...' if len(s) > 300 else ''}")
        parts.append("")
    parts.append("Paragraph to judge:")
    parts.append(paragraph_text)
    return "\n".join(parts)
