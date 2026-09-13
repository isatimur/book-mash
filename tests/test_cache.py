from book_mash.cache import JudgeScoreCache, content_hash
from mash_core import JudgeLabel, JudgeScore


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


def test_concurrent_runs_do_not_erase_each_other(tmp_path):
    """Regression, 2026-09-09: the panel runs its three members in parallel. The member
    that started first also finished last, and its flush wrote back a dict loaded before
    the others had cached anything — erasing their entries silently. The erased model
    then re-bought 416 judgements on its next run and ran out of credit mid-flight."""
    path = tmp_path / "cache.json"
    first = JudgeScoreCache(path)   # starts 00:05, flushes 00:24
    second = JudgeScoreCache(path)  # starts 00:05, flushes 00:07
    third = JudgeScoreCache(path)   # starts 00:14, flushes 00:16

    first.put(unit_hash="h", dim_name="humanness", dim_version="0.1.0", model_id="deepseek", score=make_score())
    second.put(unit_hash="h", dim_name="humanness", dim_version="0.1.0", model_id="llama", score=make_score())
    third.put(unit_hash="h", dim_name="humanness", dim_version="0.1.0", model_id="qwen", score=make_score())

    second.flush()
    third.flush()
    first.flush()  # last writer must not discard the other two

    reloaded = JudgeScoreCache(path)
    for model in ("deepseek", "llama", "qwen"):
        assert reloaded.get(
            unit_hash="h", dim_name="humanness", dim_version="0.1.0", model_id=model
        ) is not None, f"{model}'s entry was erased by a later flush"


def test_flush_is_atomic_and_leaves_no_tmp_file(tmp_path):
    """The merge writes through a same-directory temp file. It must be renamed into
    place, never left behind: the runs directory is scanned by the publish scripts."""
    path = tmp_path / "cache.json"
    cache = JudgeScoreCache(path)
    cache.put(unit_hash="h1", dim_name="humanness", dim_version="0.1.0", model_id="m", score=make_score())
    cache.flush()
    assert path.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_corrupt_cache_on_disk_is_rebuilt_not_fatal(tmp_path):
    """A half-written cache from an older build must not fail the run."""
    path = tmp_path / "cache.json"
    path.write_text('{"truncated": ')
    cache = JudgeScoreCache(path)  # tolerated on read? if not, this is the contract we want
    cache.put(unit_hash="h1", dim_name="humanness", dim_version="0.1.0", model_id="m", score=make_score())
    cache.flush()
    reloaded = JudgeScoreCache(path)
    assert reloaded.get(unit_hash="h1", dim_name="humanness", dim_version="0.1.0", model_id="m") is not None
