from enum import Enum

from pydantic import BaseModel, Field

from book_mash.judges.models import JudgeScore


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
    started_at: str  # ISO 8601 UTC
    finished_at: str | None
    total_cost_usd: float
    status: RunStatus
    rollups: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    scores: list[JudgeScore] = Field(default_factory=list)
