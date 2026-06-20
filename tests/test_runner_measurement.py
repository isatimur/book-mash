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


async def test_measurement_runs_without_embedding_key(tmp_path, mock_all_judges):
    # No mock_embeddings fixture — pick_embedding_client raises RuntimeError (no key).
    # The runner must tolerate this: skip the redundancy prefilter, still complete.
    cfg = load_config(str(FIXTURE_CONFIG))
    cfg.runs_dir = str(tmp_path)

    def _raise_no_key():
        raise RuntimeError("No embedding API key found (set OPENAI_API_KEY)")

    with patch("book_mash.runners.measurement.pick_embedding_client", side_effect=_raise_no_key):
        run = await run_measurement(cfg)

    assert run.status.value == "completed"
    # redundancy still produced chapter-level scores (it runs on Anthropic, no embeddings needed)
    redundancy_chapter_scores = [
        s for s in run.scores
        if s.dim_name == "redundancy" and s.unit_id.startswith("chapter:") and not s.derived
    ]
    assert len(redundancy_chapter_scores) == 2


async def test_hung_unit_cannot_wedge_the_batch(tmp_path, mock_embeddings, monkeypatch):
    # Regression for the claim_defensibility deadlock (post-10de5d0): a single
    # claim_def unit that never resolves was holding heavy_sem (_HEAVY_CONCURRENCY=1)
    # forever, so every other claim_def unit blocked behind it and the whole run hung
    # at 0 CPU / 0 connections. The hard wall-clock cap must force-cancel the stuck
    # unit, record it as an ERROR score, release the semaphore, and let the run finish.
    import asyncio

    import book_mash.runners.measurement as measurement

    # Tiny cap so the test is fast; the production constant is 180s.
    monkeypatch.setattr(measurement, "HARD_CAP_SECONDS", 0.2)

    hung_unit = {"id": None}

    async def fake_claim_judge(self, input):
        # The FIRST claim_defensibility unit hangs forever (never resolves, never
        # raises) — exactly the stalled-call shape. All later units must still run.
        if hung_unit["id"] is None:
            hung_unit["id"] = input.unit_id
            await asyncio.sleep(10**9)
        return make_mock_score(self.name, input.unit_id)

    async def fake_judge(self, input):
        return make_mock_score(self.name, input.unit_id)

    with patch("book_mash.judges.humanness.HumannessJudge.judge", new=fake_judge), \
         patch("book_mash.judges.voice.VoiceJudge.judge", new=fake_judge), \
         patch("book_mash.judges.usefulness.UsefulnessJudge.judge", new=fake_judge), \
         patch("book_mash.judges.evidence_density.EvidenceDensityJudge.judge", new=fake_judge), \
         patch("book_mash.judges.claim_defensibility.ClaimDefensibilityJudge.judge", new=fake_claim_judge), \
         patch("book_mash.judges.redundancy.RedundancyJudge.judge", new=fake_judge):
        cfg = load_config(str(FIXTURE_CONFIG))
        cfg.runs_dir = str(tmp_path)
        # The whole run must complete well within the cap budget; without the cap it
        # would hang forever. wait_for here is the test's own safety net.
        run = await asyncio.wait_for(run_measurement(cfg), timeout=30)

    assert run.status.value == "completed"

    claim_scores = [s for s in run.scores if s.dim_name == "claim_defensibility" and not s.derived]
    # The hung unit was recorded as an ERROR score (not lost, not hanging).
    errored = [s for s in claim_scores if s.label == JudgeLabel.ERROR]
    assert len(errored) == 1
    assert errored[0].unit_id == hung_unit["id"]
    assert errored[0].score_0_100 is None
    assert "hard timeout" in errored[0].reasoning
    # Every OTHER claim_def unit still got a real score — the hang did not wedge them.
    assert len(claim_scores) > 1
    non_errored = [s for s in claim_scores if s.label != JudgeLabel.ERROR]
    assert len(non_errored) == len(claim_scores) - 1


async def test_timed_out_unit_releases_heavy_semaphore(tmp_path, mock_embeddings, monkeypatch):
    # Tighter assertion on the wedge mechanism: a heavy unit that times out must hand
    # heavy_sem back so the NEXT heavy unit can acquire it. We observe this by checking
    # that heavy units after the hung one actually executed (inflight reached >0 again
    # after the timeout), proving the semaphore was released in finally.
    import asyncio

    import book_mash.runners.measurement as measurement

    monkeypatch.setattr(measurement, "HARD_CAP_SECONDS", 0.2)

    state = {"first_seen": False, "ran_after_hang": 0}

    async def fake_claim_judge(self, input):
        if not state["first_seen"]:
            state["first_seen"] = True
            await asyncio.sleep(10**9)  # hangs, holds heavy_sem until force-cancelled
        else:
            state["ran_after_hang"] += 1
        return make_mock_score(self.name, input.unit_id)

    async def fake_judge(self, input):
        return make_mock_score(self.name, input.unit_id)

    with patch("book_mash.judges.humanness.HumannessJudge.judge", new=fake_judge), \
         patch("book_mash.judges.voice.VoiceJudge.judge", new=fake_judge), \
         patch("book_mash.judges.usefulness.UsefulnessJudge.judge", new=fake_judge), \
         patch("book_mash.judges.evidence_density.EvidenceDensityJudge.judge", new=fake_judge), \
         patch("book_mash.judges.claim_defensibility.ClaimDefensibilityJudge.judge", new=fake_claim_judge), \
         patch("book_mash.judges.redundancy.RedundancyJudge.judge", new=fake_judge):
        cfg = load_config(str(FIXTURE_CONFIG))
        cfg.runs_dir = str(tmp_path)
        run = await asyncio.wait_for(run_measurement(cfg), timeout=30)

    assert run.status.value == "completed"
    # If the timed-out unit had NOT released heavy_sem, no later claim_def unit could
    # ever acquire it and ran_after_hang would stay 0.
    assert state["ran_after_hang"] > 0


async def test_heavy_judges_serialize_under_their_own_semaphore(tmp_path, mock_embeddings):
    # Regression for the run-4 coverage gap: humanness + claim_defensibility must
    # never run more than _HEAVY_CONCURRENCY at a time (so their large prompts stay
    # under the per-key TPM ceiling), while the small-prompt judges are unaffected.
    import asyncio

    from book_mash.runners.measurement import _HEAVY_CONCURRENCY, _HEAVY_JUDGES

    state = {"heavy_inflight": 0, "heavy_peak": 0, "heavy_total": 0}

    async def fake_judge(self, input):
        if self.name in _HEAVY_JUDGES:
            state["heavy_inflight"] += 1
            state["heavy_total"] += 1
            state["heavy_peak"] = max(state["heavy_peak"], state["heavy_inflight"])
            await asyncio.sleep(0.01)  # hold the slot so any overlap would surface
            state["heavy_inflight"] -= 1
        return make_mock_score(self.name, input.unit_id)

    with patch("book_mash.judges.humanness.HumannessJudge.judge", new=fake_judge), \
         patch("book_mash.judges.voice.VoiceJudge.judge", new=fake_judge), \
         patch("book_mash.judges.usefulness.UsefulnessJudge.judge", new=fake_judge), \
         patch("book_mash.judges.evidence_density.EvidenceDensityJudge.judge", new=fake_judge), \
         patch("book_mash.judges.claim_defensibility.ClaimDefensibilityJudge.judge", new=fake_judge), \
         patch("book_mash.judges.redundancy.RedundancyJudge.judge", new=fake_judge):
        cfg = load_config(str(FIXTURE_CONFIG))
        cfg.runs_dir = str(tmp_path)
        run = await run_measurement(cfg)

    assert run.status.value == "completed"
    # The fixture exercises enough heavy paragraphs that, without the cap, they would
    # overlap — so the peak being within the cap is a real constraint, not a vacuous one.
    assert state["heavy_total"] > _HEAVY_CONCURRENCY
    assert state["heavy_peak"] <= _HEAVY_CONCURRENCY
