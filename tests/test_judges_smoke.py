import pytest

from book_mash.judges.humanness import HumannessJudge
from book_mash.judges.voice import VoiceJudge
from mash_core import JudgeInput, JudgeLabel


@pytest.mark.live
async def test_humanness_flags_ai_slop():
    judge = HumannessJudge()
    input = JudgeInput(
        unit_id="paragraph:test",
        unit_type="paragraph",
        unit_text=(
            "In today's rapidly evolving landscape of AI engineering, developers "
            "face unprecedented challenges. By leveraging cutting-edge tools, teams "
            "can unlock new possibilities and drive transformative outcomes."
        ),
        dim_name="humanness",
        context={"surrounding_paragraphs": []},
    )
    score = await judge.judge(input)
    assert score.label in (JudgeLabel.WEAK, JudgeLabel.FAIL)
    assert score.evidence_refs  # must quote the worst phrase


@pytest.mark.live
async def test_humanness_passes_specific_prose():
    judge = HumannessJudge()
    input = JudgeInput(
        unit_id="paragraph:test",
        unit_type="paragraph",
        unit_text=(
            "Recall.ai charges $0.65 per bot-hour for diarized real-time transcription. "
            "Twelve concurrent meetings for an hour costs about $7.80 — cheap enough that "
            "the latency budget, not the bill, is what shapes how often you can interrupt."
        ),
        dim_name="humanness",
        context={"surrounding_paragraphs": []},
    )
    score = await judge.judge(input)
    assert score.label in (JudgeLabel.MODERATE, JudgeLabel.STRONG)
    assert score.evidence_refs  # worst_phrase is required even on a strong score


@pytest.mark.live
async def test_voice_flags_drift_from_baseline():
    judge = VoiceJudge()
    baseline_excerpts = [
        "The shift is concrete. A 2024 developer who reviewed pull requests on Tuesday "
        "spends Wednesday reviewing agent trajectories. Same skill, different surface.",
        "Trust is not vibes. Trust is the residual that a reader carries from page 60 to "
        "page 200, and the only way to earn it is to never lie."
    ]
    input = JudgeInput(
        unit_id="chapter:test",
        unit_type="chapter",
        unit_text=(
            "In today's rapidly evolving landscape, AI engineering teams must "
            "navigate a complex array of challenges. From orchestrating multi-agent "
            "workflows to ensuring observability, the path forward requires careful "
            "consideration of multiple competing factors. As we look ahead, several "
            "trends emerge that will shape the future of intelligent systems."
        ),
        dim_name="voice",
        context={"voice_baseline_excerpts": baseline_excerpts},
    )
    score = await judge.judge(input)
    assert score.label in (JudgeLabel.WEAK, JudgeLabel.FAIL, JudgeLabel.MODERATE)
