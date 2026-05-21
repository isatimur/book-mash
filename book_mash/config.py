import tomli
from pathlib import Path

from pydantic import BaseModel


class BookMashConfig(BaseModel):
    chapters_glob: str
    claims_dir: str
    evidence_dir: str
    voice_baseline_chapters: list[str]
    runs_dir: str
    max_cost_usd: float


def load_config(path: str) -> BookMashConfig:
    p = Path(path).resolve()
    base = p.parent
    with open(p, "rb") as f:
        raw = tomli.load(f)
    corpus = raw["corpus"]
    output = raw["output"]
    budget = raw.get("budget", {})
    return BookMashConfig(
        chapters_glob=str(base / corpus["chapters_glob"]),
        claims_dir=str(base / corpus["claims_dir"]),
        evidence_dir=str(base / corpus["evidence_dir"]),
        voice_baseline_chapters=corpus.get("voice_baseline_chapters", []),
        runs_dir=str(base / output["runs_dir"]),
        max_cost_usd=float(budget.get("max_cost_usd", 10.0)),
    )
