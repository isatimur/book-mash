"""Deterministic, local lexical retrieval of ledger claims relevant to a paragraph.

Used by the claim_defensibility judge to select which claims (drawn from the FULL
ledger, not a chapter-filtered slice) to put in front of the judge, together with
their supporting quotes. No embeddings, no network — pure token-overlap scoring so
results are reproducible and free.

Design note (P0 grounding fix, see docs/judge-module-evaluation.md):
the previous routing filtered claims by the hand-authored `candidate_chapters`
tag, which is editorial metadata, not ground truth. That withheld well-supported
claims (e.g. #32/#33 Sampath/Anthropic, tagged chapters 7/9) from paragraphs that
actually discuss them (e.g. chapter 5), producing verified false "unsupported"
flags. This module replaces that gate with relevance over the whole ledger.
"""

import math
import re
from collections import Counter

from book_mash.corpus.models import ClaimEntry

DEFAULT_TOP_K = 8

_TOKEN = re.compile(r"[a-z0-9]+")

# Common English + book-domain filler that carries no retrieval signal. Kept small
# and deterministic on purpose; this is lexical overlap, not a search engine.
_STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for with without from into
    over under again further is are was were be been being do does did doing have
    has had having this that these those it its they them their there here what
    which who whom how when where why all any both each few more most other some
    such no nor not only own same so than too very can will just as we you your our
    i me my he she his her him not about up down out off above below between through
    """.split()
)


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def _build_idf(claims: list[ClaimEntry]) -> dict[str, float]:
    """Inverse document frequency over the ledger so distinctive terms (names,
    products) outweigh generic ones. Smoothed; never zero."""
    n = len(claims) or 1
    df: Counter[str] = Counter()
    for c in claims:
        for tok in set(_tokenize(c.retrieval_text())):
            df[tok] += 1
    return {tok: math.log((n + 1) / (count + 1)) + 1.0 for tok, count in df.items()}


def score_claims(paragraph_text: str, claims: list[ClaimEntry]) -> list[tuple[float, ClaimEntry]]:
    """Score every claim against the paragraph by IDF-weighted token overlap.

    Returns (score, claim) pairs sorted by descending score, then by claim id for
    a stable, deterministic order on ties.
    """
    idf = _build_idf(claims)
    para_tokens = set(_tokenize(paragraph_text))
    scored: list[tuple[float, ClaimEntry]] = []
    for c in claims:
        claim_tokens = set(_tokenize(c.retrieval_text()))
        overlap = para_tokens & claim_tokens
        score = sum(idf.get(tok, 1.0) for tok in overlap)
        scored.append((score, c))
    scored.sort(key=lambda pair: (-pair[0], pair[1].id))
    return scored


def retrieve_relevant_claims(
    paragraph_text: str,
    claims: list[ClaimEntry],
    top_k: int = DEFAULT_TOP_K,
) -> list[ClaimEntry]:
    """Top-K claims from the FULL ledger most lexically relevant to the paragraph.

    Claims with zero overlap are dropped (a paragraph with no ledger-relevant
    content should see an empty/short set rather than K arbitrary claims). The
    candidate_chapters tag is intentionally NOT consulted.
    """
    scored = score_claims(paragraph_text, claims)
    return [c for score, c in scored[:top_k] if score > 0.0]
