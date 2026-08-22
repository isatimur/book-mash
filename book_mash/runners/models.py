from enum import Enum

from pydantic import BaseModel, Field

from mash_core import JudgeScore


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    HALTED_BUDGET = "halted_budget"
    ABORTED = "aborted"


class Run(BaseModel):
    id: str
    corpus_snapshot_hash: str
    book_mash_version: str
    dim_registry_version: str
    # Per-judge prompt-version strings (e.g. claim_defensibility's own
    # CLAIM_DEFENSIBILITY_PROMPT_VERSION), keyed by dim name. dim_registry_version
    # tracks which dimensions exist; this tracks whether an individual judge's
    # grading rubric changed. Without this, a rubric change and a genuine content
    # improvement are indistinguishable on a public score trend line — a gap a
    # retroactive ship-gate review found on 2026-08-22 (see ledger/verdicts.md).
    judge_prompt_versions: dict[str, str] = Field(default_factory=dict)
    started_at: str  # ISO 8601 UTC
    finished_at: str | None
    total_cost_usd: float
    status: RunStatus
    rollups: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    scores: list[JudgeScore] = Field(default_factory=list)
