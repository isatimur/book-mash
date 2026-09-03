from book_mash.judges.evidence_density import _label_for_density


def test_label_strong_above_threshold():
    # 1 claim per 250 words -> strong
    assert _label_for_density(claims_count=4, word_count=1000) == "strong"


def test_label_moderate():
    # 1 claim per 500 words
    assert _label_for_density(claims_count=2, word_count=1000) == "moderate"


def test_label_weak():
    # 1 claim per 800 words
    assert _label_for_density(claims_count=1, word_count=800) == "weak"


def test_label_fail():
    # 0 claims
    assert _label_for_density(claims_count=0, word_count=500) == "fail"
    # 1 per 1500 words
    assert _label_for_density(claims_count=1, word_count=1500) == "fail"


def test_cache_key_changes_when_ledger_changes():
    """A ledger edit must invalidate cached evidence_density scores; the section text alone is not the key."""
    from unittest.mock import patch
    from book_mash.corpus.models import ClaimEntry
    from book_mash.judges.evidence_density import EvidenceDensityJudge
    from mash_core import JudgeInput

    with patch("book_mash.judges.evidence_density._build_agent", return_value=(object(), "mock")):
        judge = EvidenceDensityJudge()
    def entry(i, text):
        return ClaimEntry(id=f"claims#{i}", text=text, support_level="moderate", file_path="ledger.md")
    mk = lambda ledger: JudgeInput(unit_id="section:x#y", unit_type="section", unit_text="same text",
                                   dim_name="evidence_density", context={"claims_index": ledger})
    k1 = judge.context_cache_key(mk([entry(1, "a")]))
    k2 = judge.context_cache_key(mk([entry(1, "a"), entry(2, "new claim")]))
    k3 = judge.context_cache_key(mk([entry(1, "a")]))
    assert k1 != k2 and k1 == k3 and k1 != ""
