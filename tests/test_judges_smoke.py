import pytest

from book_mash.judges.humanness import HumannessJudge
from book_mash.judges.models import JudgeInput, JudgeLabel


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
