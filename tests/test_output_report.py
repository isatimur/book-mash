from mash_core import JudgeLabel, JudgeScore
from book_mash.output.report import render_report
from book_mash.runners.models import Run, RunStatus


def make_score(dim: str, unit_id: str, score: float, label: JudgeLabel = JudgeLabel.MODERATE) -> JudgeScore:
    return JudgeScore(
        dim_name=dim, unit_id=unit_id, score_0_100=score, label=label,
        reasoning="r", evidence_refs=[], model="m", cost_usd=0.0, derived=False,
    )


def test_renders_report_md(tmp_path):
    run = Run(
        id="r1", corpus_snapshot_hash="sha256:abc", book_mash_version="0.1.0",
        dim_registry_version="0.1.0", started_at="t1", finished_at="t2",
        total_cost_usd=4.21, status=RunStatus.COMPLETED,
        rollups={
            "book": {"humanness": 65.0, "voice": 80.0, "usefulness": 70.0,
                     "evidence_density": 55.0, "claim_defensibility": 90.0, "redundancy": 75.0},
            "chapter:ch1": {"humanness": 70.0, "n_paragraphs": 5},
            "chapter:ch2": {"humanness": 60.0, "n_paragraphs": 5},
        },
        scores=[
            make_score("humanness", "paragraph:ch1#L1-L2", 25.0, JudgeLabel.WEAK),
            make_score("claim_defensibility", "paragraph:ch2#L5-L6", 10.0, JudgeLabel.FAIL),
        ],
    )
    render_report(run, tmp_path)
    p = tmp_path / "report.md"
    assert p.exists()
    text = p.read_text()
    assert "# book-mash run" in text
    assert "Heatmap" in text
    assert "Ship-blockers" in text
    assert "paragraph:ch2#L5-L6" in text
