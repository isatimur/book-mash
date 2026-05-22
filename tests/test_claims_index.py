from pathlib import Path

from book_mash.corpus.claims_index import load_claims_index


CLAIMS_DIR = str(Path(__file__).parent / "fixtures" / "mini_book" / "claims")


def test_loads_two_claims():
    claims = load_claims_index(CLAIMS_DIR)
    assert len(claims) == 2


def test_claim_fields():
    claims = load_claims_index(CLAIMS_DIR)
    by_id = {c.id: c for c in claims}
    c1 = by_id["claims#1"]
    assert c1.support_level == "strong"
    assert "delegates that act" in c1.text
    assert c1.candidate_chapters == [1]
    assert any("206" in s for s in c1.source_refs)
    assert any("225" in s for s in c1.source_refs)


def test_candidate_chapters_parsed_as_ints():
    claims = load_claims_index(CLAIMS_DIR)
    by_id = {c.id: c for c in claims}
    assert by_id["claims#2"].candidate_chapters == [2]


def test_empty_dir_returns_empty_list(tmp_path):
    claims = load_claims_index(str(tmp_path))
    assert claims == []
