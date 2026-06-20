"""Tests for the P0 claim_defensibility grounding fix.

These exercise the retrieval + prompt-assembly path directly. No LLM is called.
The key regression guarded here is the Sampath/Anthropic routing false positive:
ledger claims #32/#33 are tagged candidate_chapters 7/9, yet they support a
chapter-5 paragraph about Karan Sampath / Anthropic MCP. The judge must now SEE
those claims (full-ledger relevance retrieval, not chapter filtering) and their
supporting quotes.
"""

from pathlib import Path

import pytest

from book_mash.corpus.claim_retrieval import (
    DEFAULT_TOP_K,
    retrieve_relevant_claims,
    score_claims,
)
from book_mash.corpus.claims_index import load_claims_index
from book_mash.corpus.models import ClaimEntry
from book_mash.judges.claim_defensibility import build_prompt

# The real consumer ledger. The fix is about routing real cross-chapter claims,
# so the regression test is anchored to the real data that exposed the bug.
REAL_CLAIMS_DIR = "/Users/timur_isachenko/Dev/LifeOS/knowledge-bases/ai-engineer-book/claims"

SAMPATH_PARAGRAPH = (
    "Karan Sampath shared what Anthropic learned scaling MCP to enterprises: "
    "security teams need to establish a root of trust and bless one platform "
    "rather than trusting every individual tool. Standardization expands the "
    "attack surface when governance lags."
)


def _real_claims() -> list[ClaimEntry]:
    if not Path(REAL_CLAIMS_DIR).exists():
        pytest.skip("real consumer claims ledger not present")
    claims = load_claims_index(REAL_CLAIMS_DIR)
    if not claims:
        pytest.skip("real consumer claims ledger empty")
    return claims


# --- parser: quotes / sources / phrasing are now captured ---------------------


def test_parser_captures_quotes_and_sources_for_sampath_claims():
    by_id = {c.id: c for c in _real_claims()}
    c32, c33 = by_id["claims#32"], by_id["claims#33"]

    # The verbatim quotes the judge needs to match prose against.
    assert any("root of trust" in q for q in c32.quotes)
    assert any("bless one platform" in q for q in c33.quotes)
    # Source attribution labels are captured so "Anthropic / Sampath" prose matches.
    assert any("Karan Sampath, Anthropic" in s for s in c32.source_descriptions)
    assert any("Karan Sampath, Anthropic" in s for s in c33.source_descriptions)
    # Reusable phrasing captured too.
    assert c33.reusable_phrasing


# --- routing fix: full-ledger retrieval ignores candidate_chapters -------------


def test_sampath_paragraph_retrieves_claims_32_and_33_despite_chapter_tags():
    claims = _real_claims()
    by_id = {c.id: c for c in claims}
    # Precondition: these claims are tagged to chapters 7/9, NOT chapter 5 — under
    # the old candidate_chapters filter a chapter-5 paragraph would never see them.
    assert 5 not in by_id["claims#32"].candidate_chapters
    assert 5 not in by_id["claims#33"].candidate_chapters

    retrieved = retrieve_relevant_claims(SAMPATH_PARAGRAPH, claims, top_k=DEFAULT_TOP_K)
    ids = {c.id for c in retrieved}
    assert "claims#32" in ids, "routing fix failed: #32 not retrieved"
    assert "claims#33" in ids, "routing fix failed: #33 not retrieved"


def test_assembled_prompt_for_sampath_includes_supporting_quotes():
    claims = _real_claims()
    retrieved = retrieve_relevant_claims(SAMPATH_PARAGRAPH, claims)
    prompt = build_prompt(SAMPATH_PARAGRAPH, retrieved)

    # The judge can now match prose like "bless one platform" to the quote.
    assert "bless one platform" in prompt
    assert "establish a root of trust" in prompt
    assert "Karan Sampath, Anthropic" in prompt
    # Support level still surfaced for the strength comparison.
    assert "support_level: strong" in prompt
    # The paragraph itself is present.
    assert "Karan Sampath shared" in prompt


def test_retrieval_caps_at_top_k():
    claims = _real_claims()
    retrieved = retrieve_relevant_claims(SAMPATH_PARAGRAPH, claims, top_k=6)
    assert len(retrieved) <= 6


# --- retrieval scoring: deterministic, relevance-ordered ----------------------


def _claim(cid: str, text: str, quotes=None, phrasing="") -> ClaimEntry:
    return ClaimEntry(
        id=cid,
        text=text,
        support_level="strong",
        quotes=quotes or [],
        reusable_phrasing=phrasing,
        file_path="x.md",
    )


def test_relevance_ranks_matching_claim_first():
    claims = [
        _claim("claims#1", "Embeddings power semantic search over documents"),
        _claim(
            "claims#2",
            "A root of trust must be established at the platform",
            quotes=["bless one platform"],
        ),
        _claim("claims#3", "Specs are executable intent for delegated work"),
    ]
    para = "Security teams bless one platform and establish a root of trust."
    scored = score_claims(para, claims)
    assert scored[0][1].id == "claims#2"


def test_zero_overlap_claims_are_dropped():
    claims = [_claim("claims#1", "Quantum chromodynamics and gluon confinement")]
    para = "A paragraph about agent harness engineering and codebases."
    assert retrieve_relevant_claims(para, claims) == []


def test_retrieval_is_deterministic():
    claims = [
        _claim("claims#1", "root of trust platform governance"),
        _claim("claims#2", "root of trust platform governance"),  # tie
    ]
    para = "root of trust platform governance"
    first = [c.id for c in retrieve_relevant_claims(para, claims)]
    second = [c.id for c in retrieve_relevant_claims(para, claims)]
    assert first == second  # stable tie-break by id
    assert first == ["claims#1", "claims#2"]
