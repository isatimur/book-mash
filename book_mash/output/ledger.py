import json
from pathlib import Path

from book_mash.runners.models import Run


def write_ledger(run: Run, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "run_id": run.id,
        "corpus_snapshot_hash": run.corpus_snapshot_hash,
        "book_mash_version": run.book_mash_version,
        "dim_registry_version": run.dim_registry_version,
        "judge_prompt_versions": run.judge_prompt_versions,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "total_cost_usd": run.total_cost_usd,
        "status": run.status.value,
        "rollups": run.rollups,
        "scores": [s.model_dump(mode="json") for s in run.scores],
    }
    (run_dir / "scores.json").write_text(json.dumps(out, indent=2, default=str))
    small = dict(out)
    small.pop("scores")
    (run_dir / "run.json").write_text(json.dumps(small, indent=2, default=str))
