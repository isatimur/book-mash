from pathlib import Path

from book_mash.corpus.claims_index import load_claims_index


CLAIMS_DIR = str(Path(__file__).parent / "fixtures" / "mini_book" / "claims")


def test_loads_two_claims():
    claims = load_claims_index(CLAIMS_DIR)
    assert len(claims) == 2


def test_claim_fields():
    claims = load_claims_index(CLAIMS_DIR)
    by_id = {c.id: c for c in claims}
    c1 = by_id["claim:ch1#delegation-shift"]
    assert c1.strength == "strong"
    assert "delegates that act" in c1.text
    assert "YT-#206" in c1.source_refs
    assert "YT-#225" in c1.source_refs


def test_empty_dir_returns_empty_list(tmp_path):
    claims = load_claims_index(str(tmp_path))
    assert claims == []
