import hashlib
import json
from pathlib import Path

from book_mash.judges.models import JudgeScore


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class JudgeScoreCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._entries: dict[str, dict] = {}
        if self.path.exists():
            self._entries = json.loads(self.path.read_text())

    def _key(self, unit_hash: str, dim_name: str, dim_version: str, model_id: str) -> str:
        return f"{unit_hash}|{dim_name}|{dim_version}|{model_id}"

    def get(self, unit_hash: str, dim_name: str, dim_version: str, model_id: str) -> JudgeScore | None:
        raw = self._entries.get(self._key(unit_hash, dim_name, dim_version, model_id))
        if raw is None:
            return None
        return JudgeScore.model_validate(raw)

    def put(self, unit_hash: str, dim_name: str, dim_version: str, model_id: str, score: JudgeScore) -> None:
        self._entries[self._key(unit_hash, dim_name, dim_version, model_id)] = score.model_dump()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._entries, indent=2))
