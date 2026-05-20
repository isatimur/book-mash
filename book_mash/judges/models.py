from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class JudgeLabel(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    FAIL = "fail"
    ERROR = "error"


UnitType = Literal["paragraph", "section", "chapter"]


class JudgeScore(BaseModel):
    dim_name: str
    unit_id: str
    score_0_100: float | None  # None when label == ERROR
    label: JudgeLabel
    reasoning: str
    evidence_refs: list[str] = Field(default_factory=list)
    model: str
    cost_usd: float
    derived: bool = False  # True when broadcast from a higher-level native score


class JudgeInput(BaseModel):
    unit_id: str
    unit_type: UnitType
    unit_text: str
    dim_name: str
    context: dict = Field(default_factory=dict)


class JudgeResult(BaseModel):
    unit_id: str
    scores: list[JudgeScore]
