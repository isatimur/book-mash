"""Dry-run planner: count units + estimate cost without invoking any judges.

The cost estimates use rough per-call token assumptions for the three
granularity levels. Actual costs vary with chapter length, claims-ledger size,
and surrounding-paragraph context — these numbers are directionally useful
for budget-vs-plan checks before paying for a real run, not exact.
"""

from dataclasses import dataclass

from book_mash.config import BookMashConfig
from book_mash.corpus.loader import load_chapters
from book_mash.judges import _pricing


# Rough token-per-call assumptions for each granularity level.
# Input includes prompt template + context + unit text; output is a structured JudgeScore.
_TOKENS_PER_CALL = {
    "paragraph": (1500, 250),
    "section": (3000, 300),
    "chapter": (8000, 400),
}

# Model assignments per dim. Mirrors the actual judge implementations.
# If a judge swaps models, update here too (the dry-run will then disagree
# with reality, which is itself a useful signal).
_DIM_MODEL: dict[str, str] = {
    "humanness": "claude-sonnet-4-6",
    "usefulness": "claude-sonnet-4-6",
    "claim_defensibility": "claude-sonnet-4-6",
    "evidence_density": "claude-sonnet-4-6",
    "voice": "claude-sonnet-4-6",
    "redundancy": "claude-sonnet-4-6",
}

# Granularity per dim — drives both the call count and the token assumption.
_DIM_GRANULARITY: dict[str, str] = {
    "humanness": "paragraph",
    "usefulness": "paragraph",
    "claim_defensibility": "paragraph",
    "evidence_density": "section",
    "voice": "chapter",
    "redundancy": "chapter",
}


@dataclass(frozen=True)
class DimPlan:
    dim_name: str
    granularity: str
    model_id: str
    call_count: int
    estimated_cost_usd: float


@dataclass(frozen=True)
class MeasurementPlan:
    n_chapters: int
    n_sections: int
    n_paragraphs: int
    dims: list[DimPlan]
    total_calls: int
    total_estimated_cost_usd: float
    budget_cap_usd: float
    embedder_available: bool
    estimated_embedding_calls: int


def _per_call_cost(granularity: str, model_id: str) -> float:
    in_tokens, out_tokens = _TOKENS_PER_CALL[granularity]
    return _pricing.estimate_cost(model_id, in_tokens, out_tokens)


def _embedder_available() -> bool:
    try:
        from book_mash.judges.embeddings import pick_embedding_client

        pick_embedding_client()
        return True
    except Exception:
        return False


def plan_measurement(cfg: BookMashConfig) -> MeasurementPlan:
    """Count units and estimate cost. No LLM or embedding calls."""
    chapters = load_chapters(cfg.chapters_glob, cfg.skip_sections)
    n_chapters = len(chapters)
    n_sections = sum(len(c.sections) for c in chapters)
    n_paragraphs = sum(len(s.paragraphs) for c in chapters for s in c.sections)

    unit_counts = {
        "paragraph": n_paragraphs,
        "section": n_sections,
        "chapter": n_chapters,
    }

    dims: list[DimPlan] = []
    for dim_name, model_id in _DIM_MODEL.items():
        granularity = _DIM_GRANULARITY[dim_name]
        calls = unit_counts[granularity]
        cost = calls * _per_call_cost(granularity, model_id)
        dims.append(
            DimPlan(
                dim_name=dim_name,
                granularity=granularity,
                model_id=model_id,
                call_count=calls,
                estimated_cost_usd=cost,
            )
        )

    embedder_available = _embedder_available()
    return MeasurementPlan(
        n_chapters=n_chapters,
        n_sections=n_sections,
        n_paragraphs=n_paragraphs,
        dims=dims,
        total_calls=sum(d.call_count for d in dims),
        total_estimated_cost_usd=sum(d.estimated_cost_usd for d in dims),
        budget_cap_usd=cfg.max_cost_usd,
        embedder_available=embedder_available,
        estimated_embedding_calls=n_chapters if embedder_available else 0,
    )
