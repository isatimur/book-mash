from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from book_mash.config import load_config
from book_mash.judges.models import JudgeLabel, JudgeScore
from book_mash.runners.measurement import run_measurement


FIXTURE_CONFIG = Path(__file__).parent / "fixtures" / "mini_book" / "book-mash.toml"


@pytest.fixture(autouse=True)
def stub_api_keys(monkeypatch):
    # Judge __init__ builds a pydantic-ai agent that reads ANTHROPIC_API_KEY.
    # Constructing the model with a stub key makes no network call; judge methods are mocked.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-stub")


def make_mock_score(dim: str, unit_id: str, score: float = 75.0) -> JudgeScore:
    return JudgeScore(
        dim_name=dim,
        unit_id=unit_id,
        score_0_100=score,
        label=JudgeLabel.MODERATE,
        reasoning="mocked",
        evidence_refs=[],
        model="mock",
        cost_usd=0.001,
        derived=False,
    )


@pytest.fixture
def mock_all_judges():
    call_counter = {"count": 0}

    async def fake_judge(self, input):
        call_counter["count"] += 1
        return make_mock_score(self.name, input.unit_id)

    with patch("book_mash.judges.humanness.HumannessJudge.judge", new=fake_judge), \
         patch("book_mash.judges.voice.VoiceJudge.judge", new=fake_judge), \
         patch("book_mash.judges.usefulness.UsefulnessJudge.judge", new=fake_judge), \
         patch("book_mash.judges.evidence_density.EvidenceDensityJudge.judge", new=fake_judge), \
         patch("book_mash.judges.claim_defensibility.ClaimDefensibilityJudge.judge", new=fake_judge), \
         patch("book_mash.judges.redundancy.RedundancyJudge.judge", new=fake_judge):
        yield call_counter


@pytest.fixture
def mock_embeddings():
    async def fake_embed(texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    mock_client = AsyncMock()
    mock_client.embed = fake_embed
    with patch("book_mash.runners.measurement.pick_embedding_client", return_value=mock_client):
        yield


async def test_measurement_runs_end_to_end(tmp_path, mock_all_judges, mock_embeddings):
    cfg = load_config(str(FIXTURE_CONFIG))
    cfg.runs_dir = str(tmp_path)
    run = await run_measurement(cfg)
    assert run.status.value == "completed"
    chapter_ids = {s.unit_id for s in run.scores if s.unit_id.startswith("chapter:")}
    assert len(chapter_ids) == 2
    assert run.total_cost_usd > 0
    assert "book" in run.rollups
    assert any(s.derived for s in run.scores)


async def test_measurement_idempotent_with_warm_cache(tmp_path, mock_all_judges, mock_embeddings):
    cfg = load_config(str(FIXTURE_CONFIG))
    cfg.runs_dir = str(tmp_path)
    run1 = await run_measurement(cfg)
    calls_after_run1 = mock_all_judges["count"]
    assert calls_after_run1 > 0  # first run actually called judges
    run2 = await run_measurement(cfg)
    calls_after_run2 = mock_all_judges["count"]
    # second run on unchanged corpus must be fully served from the warm cache
    assert calls_after_run2 == calls_after_run1
    scores_1 = sorted([(s.unit_id, s.dim_name) for s in run1.scores])
    scores_2 = sorted([(s.unit_id, s.dim_name) for s in run2.scores])
    assert scores_1 == scores_2
