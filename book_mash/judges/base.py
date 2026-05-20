from abc import ABC, abstractmethod
from typing import ClassVar

from book_mash.judges.models import JudgeInput, JudgeScore


class JudgeDim(ABC):
    name: ClassVar[str]
    unit_type: ClassVar[str]  # "paragraph" | "section" | "chapter"
    model_id: ClassVar[str]

    @abstractmethod
    async def judge(self, input: JudgeInput) -> JudgeScore:
        ...

    def label_for_score(self, score: float) -> str:
        if score >= 80:
            return "strong"
        if score >= 50:
            return "moderate"
        if score >= 20:
            return "weak"
        return "fail"
