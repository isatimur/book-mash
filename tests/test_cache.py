from book_mash.cache import JudgeScoreCache, content_hash
from book_mash.judges.models import JudgeLabel, JudgeScore


def make_score(unit_id: str = "p1", dim: str = "humanness") -> JudgeScore:
    return JudgeScore(
        dim_name=dim,
        unit_id=unit_id,
        score_0_100=80.0,
        label=JudgeLabel.STRONG,
        reasoning="r",
        evidence_refs=[],
        model="claude-sonnet-4-6",
        cost_usd=0.01,
        derived=False,
    )


def test_content_hash_is_stable():
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("a") != content_hash("b")


def test_cache_store_and_retrieve(tmp_path):
    cache = JudgeScoreCache(tmp_path / "cache.json")
    s = make_score()
    cache.put(unit_hash="h1", dim_name="humanness", dim_version="0.1.0", model_id="claude-sonnet-4-6", score=s)
    got = cache.get(unit_hash="h1", dim_name="humanness", dim_version="0.1.0", model_id="claude-sonnet-4-6")
    assert got is not None
    assert got.label == JudgeLabel.STRONG


def test_cache_miss_on_different_key(tmp_path):
    cache = JudgeScoreCache(tmp_path / "cache.json")
    cache.put(unit_hash="h1", dim_name="humanness", dim_version="0.1.0", model_id="claude-sonnet-4-6", score=make_score())
    assert cache.get(unit_hash="h2", dim_name="humanness", dim_version="0.1.0", model_id="claude-sonnet-4-6") is None
    assert cache.get(unit_hash="h1", dim_name="voice", dim_version="0.1.0", model_id="claude-sonnet-4-6") is None
    assert cache.get(unit_hash="h1", dim_name="humanness", dim_version="0.2.0", model_id="claude-sonnet-4-6") is None


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "cache.json"
    cache1 = JudgeScoreCache(path)
    cache1.put(unit_hash="h1", dim_name="humanness", dim_version="0.1.0", model_id="m", score=make_score())
    cache1.flush()
    cache2 = JudgeScoreCache(path)
    got = cache2.get(unit_hash="h1", dim_name="humanness", dim_version="0.1.0", model_id="m")
    assert got is not None
