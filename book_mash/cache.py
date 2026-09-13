import hashlib
import json
from pathlib import Path

from mash_core import JudgeScore


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class JudgeScoreCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._entries: dict[str, dict] = {}
        if self.path.exists():
            try:
                self._entries = json.loads(self.path.read_text())
            except (OSError, json.JSONDecodeError) as e:
                # Losing the cache costs money, it does not corrupt results, so rebuild
                # rather than fail the run — but say so loudly, because the next run
                # re-buys every judgement and that is the expensive, silent failure.
                print(
                    f"[cache] WARNING: {self.path} is unreadable ({e}); starting from "
                    f"empty. This run will re-judge every unit at full price."
                )
                self._entries = {}

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
        """Merge this run's entries into whatever is on disk, then replace atomically.

        Read-all-at-start / write-all-at-end silently ERASED concurrent runs. The three
        panel members are deliberately run in parallel, so on 2026-09-09 the member that
        started first also finished last (00:05:24 -> 00:24:09) and its flush wrote back
        a dict loaded at 00:05, discarding every entry the other two had cached at 00:07
        and 00:16. The loss is invisible: the file still looks healthy, the erased model
        simply pays to re-judge the same units on its next run.

        Merging disk under memory makes concurrent writers safe without a lock (which
        would serialize runs we parallelize on purpose). Entries are keyed by content
        hash + dim + dim version + model, so a collision means two equivalent results
        and either value is correct; preferring memory keeps the write idempotent.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        merged: dict[str, dict] = {}
        if self.path.exists():
            try:
                merged = json.loads(self.path.read_text())
            except (OSError, json.JSONDecodeError):
                # A corrupt or half-written cache is a performance problem, not a
                # correctness one: drop it and rebuild rather than fail the run.
                merged = {}
        merged.update(self._entries)
        # Same-directory temp + rename: atomic on one filesystem, so a reader never
        # sees a half-written cache and a crash mid-write cannot truncate the file.
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(merged, indent=2))
        tmp.replace(self.path)
