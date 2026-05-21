from book_mash.judges.models import JudgeLabel, JudgeScore


def weighted_mean(values: list[float | None], weights: list[int]) -> float | None:
    pairs = [(v, w) for v, w in zip(values, weights) if v is not None]
    if not pairs:
        return None
    total_w = sum(w for _, w in pairs)
    if total_w == 0:
        return None
    return sum(v * w for v, w in pairs) / total_w


def _label_for_score(score: float | None) -> JudgeLabel:
    if score is None:
        return JudgeLabel.ERROR
    if score >= 80:
        return JudgeLabel.STRONG
    if score >= 50:
        return JudgeLabel.MODERATE
    if score >= 20:
        return JudgeLabel.WEAK
    return JudgeLabel.FAIL


def rollup_paragraph_to_section(scores: list[JudgeScore], weights: list[int], section_id: str) -> JudgeScore:
    dim = scores[0].dim_name
    mean = weighted_mean([s.score_0_100 for s in scores], weights)
    n_errors = sum(1 for s in scores if s.label == JudgeLabel.ERROR)
    return JudgeScore(
        dim_name=dim,
        unit_id=section_id,
        score_0_100=mean,
        label=_label_for_score(mean),
        reasoning=f"weighted mean of {len(scores) - n_errors} paragraph scores ({n_errors} errors excluded)",
        evidence_refs=[],
        model="rollup",
        cost_usd=0.0,
        derived=False,
    )


def rollup_section_to_chapter(scores: list[JudgeScore], weights: list[int], chapter_id: str) -> JudgeScore:
    dim = scores[0].dim_name
    mean = weighted_mean([s.score_0_100 for s in scores], weights)
    n_errors = sum(1 for s in scores if s.label == JudgeLabel.ERROR)
    return JudgeScore(
        dim_name=dim,
        unit_id=chapter_id,
        score_0_100=mean,
        label=_label_for_score(mean),
        reasoning=f"weighted mean of {len(scores) - n_errors} section scores ({n_errors} errors excluded)",
        evidence_refs=[],
        model="rollup",
        cost_usd=0.0,
        derived=False,
    )


def broadcast_chapter_to_paragraphs(chapter_score: JudgeScore, paragraph_ids: list[str]) -> list[JudgeScore]:
    """Voice/redundancy chapter-native scores broadcast down with derived=True."""
    return [
        JudgeScore(
            dim_name=chapter_score.dim_name,
            unit_id=pid,
            score_0_100=chapter_score.score_0_100,
            label=chapter_score.label,
            reasoning=f"inherited from {chapter_score.unit_id}",
            evidence_refs=chapter_score.evidence_refs,
            model=chapter_score.model,
            cost_usd=0.0,
            derived=True,
        )
        for pid in paragraph_ids
    ]
