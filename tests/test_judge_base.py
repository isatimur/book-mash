import pytest

from book_mash.corpus.models import ClaimEntry
from book_mash.judges.claim_defensibility import ClaimDefensibilityJudge
from book_mash.judges.registry import dim_registry, DIM_REGISTRY_VERSION
from mash_core import JudgeInput


def test_registry_version_is_string():
    assert isinstance(DIM_REGISTRY_VERSION, str)
    assert "." in DIM_REGISTRY_VERSION


def test_registry_starts_empty():
    # The 6 dims register themselves on import in later tasks.
    # In this task, the registry exists but no real dims are registered yet.
    assert isinstance(dim_registry, dict)


@pytest.fixture
def claim_defensibility_judge(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return ClaimDefensibilityJudge()


def _make_input(ledger: list[ClaimEntry]) -> JudgeInput:
    return JudgeInput(
        unit_id="paragraph:ch1#L1-L3",
        unit_type="paragraph",
        unit_text="some prose",
        dim_name="claim_defensibility",
        context={"relevant_ledger": ledger},
    )


def _claim(id: str, text: str, support_level: str = "moderate") -> ClaimEntry:
    return ClaimEntry(id=id, text=text, support_level=support_level, file_path="claims/c.md")


def test_context_cache_key_empty_ledger_is_stable(claim_defensibility_judge):
    key1 = claim_defensibility_judge.context_cache_key(_make_input([]))
    key2 = claim_defensibility_judge.context_cache_key(_make_input([]))
    assert key1 == key2
    assert len(key1) == 64  # sha256 hex digest


def test_context_cache_key_changes_on_text(claim_defensibility_judge):
    key_a = claim_defensibility_judge.context_cache_key(_make_input([_claim("c1", "Claim A")]))
    key_b = claim_defensibility_judge.context_cache_key(_make_input([_claim("c1", "Claim B")]))
    assert key_a != key_b


def test_context_cache_key_changes_on_support_level(claim_defensibility_judge):
    key_a = claim_defensibility_judge.context_cache_key(_make_input([_claim("c1", "Same text", "moderate")]))
    key_b = claim_defensibility_judge.context_cache_key(_make_input([_claim("c1", "Same text", "strong")]))
    assert key_a != key_b


def test_context_cache_key_stable_across_calls(claim_defensibility_judge):
    ledger = [_claim("c1", "Claim one"), _claim("c2", "Claim two", "strong")]
    inp = _make_input(ledger)
    assert claim_defensibility_judge.context_cache_key(inp) == claim_defensibility_judge.context_cache_key(inp)
