from mash_core import JudgeLabel, JudgeScore
from book_mash.rollups import weighted_mean, rollup_paragraph_to_section, rollup_section_to_chapter


def s(dim: str, score: float | None, char_count: int = 100, unit_id: str = "p") -> JudgeScore:
    return JudgeScore(
        dim_name=dim,
        unit_id=unit_id,
        score_0_100=score,
        label=JudgeLabel.MODERATE if score is not None else JudgeLabel.ERROR,
        reasoning="x",
        evidence_refs=[],
        model="m",
        cost_usd=0.0,
        derived=False,
    )


def test_weighted_mean_basic():
    assert weighted_mean([80.0, 60.0], [1, 1]) == 70.0
    assert weighted_mean([80.0, 60.0], [3, 1]) == 75.0


def test_weighted_mean_skips_none():
    # 80 (weight 1) and None (weight 1) -> 80 since None is skipped
    assert weighted_mean([80.0, None], [1, 1]) == 80.0


def test_weighted_mean_all_none_returns_none():
    assert weighted_mean([None, None], [1, 1]) is None


def test_paragraph_to_section_rollup():
    scores = [
        s("humanness", 80.0, char_count=100, unit_id="p1"),
        s("humanness", 60.0, char_count=300, unit_id="p2"),
    ]
    weights = [100, 300]
    rolled = rollup_paragraph_to_section(scores, weights, "section:x")
    assert rolled.dim_name == "humanness"
    assert rolled.score_0_100 == 65.0  # (80*100 + 60*300) / 400
    assert rolled.unit_id == "section:x"
    assert rolled.derived is False  # rollup is a real native section score
