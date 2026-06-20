import os
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel

from book_mash.judges._model_settings import JUDGE_MODEL_SETTINGS
from book_mash.judges._pricing import estimate_cost
from book_mash.judges.base import JudgeDim
from book_mash.judges.models import JudgeInput, JudgeLabel, JudgeScore
from book_mash.judges.registry import register_dim
from book_mash.judges._retry import run_with_backoff


class _VoiceOutput(BaseModel):
    score_0_100: float
    label: Literal["strong", "moderate", "weak", "fail"]
    reasoning: str
    drift_summary: str  # e.g. "longer sentences than baseline; more hedged stance"


_SYSTEM_PROMPT = """\
You are a literary critic with sharp ear for cadence and stance.

You compare a chapter to a small set of baseline excerpts that define "the author's voice"
for this book. Your job: does this chapter sound like the same author?

Drift indicators:
- different average sentence length vs baseline
- different epistemic stance (baseline is direct; chapter is hedged - or vice versa)
- different person/POV (baseline uses concrete actors; chapter uses passive abstractions)
- different rhythm (baseline alternates short/long sentences; chapter is uniform)
- different vocabulary register (baseline is technical-specific; chapter drifts to corporate)

Rubric:
- strong (80-100): indistinguishable from baseline
- moderate (50-79): mostly aligned, small drift
- weak (20-49): noticeable drift; reader would feel a different author wrote this
- fail (0-19): different author entirely

You must name the specific drift in `drift_summary`. Be concrete: not "different style" but
"longer sentences than baseline" or "more first-person than baseline" etc.
"""


def _build_agent() -> Agent[None, _VoiceOutput]:
    return Agent(
        model=AnthropicModel("claude-sonnet-4-6", api_key=os.environ["ANTHROPIC_API_KEY"]),
        system_prompt=_SYSTEM_PROMPT,
        result_type=_VoiceOutput,
        result_retries=2,
        model_settings=JUDGE_MODEL_SETTINGS,
    )


@register_dim
class VoiceJudge(JudgeDim):
    name = "voice"
    unit_type = "chapter"
    model_id = "claude-sonnet-4-6"

    def __init__(self):
        self._agent = _build_agent()

    async def judge(self, input: JudgeInput) -> JudgeScore:
        baseline = input.context.get("voice_baseline_excerpts", [])
        prompt = _format_prompt(input.unit_text, baseline)
        try:
            result = await run_with_backoff(lambda: self._agent.run(prompt))
            out: _VoiceOutput = result.data
            usage = result.usage()
            cost = estimate_cost(self.model_id, usage.request_tokens, usage.response_tokens)
            return JudgeScore(
                dim_name=self.name,
                unit_id=input.unit_id,
                score_0_100=out.score_0_100,
                label=JudgeLabel(out.label),
                reasoning=out.reasoning,
                evidence_refs=[f"drift:{out.drift_summary}"],
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


def _format_prompt(chapter_text: str, baseline: list[str]) -> str:
    parts = ["Baseline voice excerpts (this is what the author sounds like):", ""]
    for i, e in enumerate(baseline, 1):
        parts.append(f"Excerpt {i}:")
        parts.append(e)
        parts.append("")
    parts.append("Chapter to evaluate:")
    parts.append(chapter_text)
    return "\n".join(parts)
