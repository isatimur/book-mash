import json
from pathlib import Path

from book_mash.output.ledger import write_ledger
from book_mash.runners.models import Run, RunStatus


def test_writes_scores_json(tmp_path):
    run = Run(
        id="2026-05-19-1547-a7f3",
        corpus_snapshot_hash="sha256:abc",
        book_mash_version="0.1.0",
        dim_registry_version="0.1.0",
        started_at="2026-05-19T15:47:12+00:00",
        finished_at="2026-05-19T15:52:08+00:00",
        total_cost_usd=4.21,
        status=RunStatus.COMPLETED,
        rollups={"book": {"humanness": 67.0}},
        scores=[],
    )
    write_ledger(run, tmp_path)
    p = tmp_path / "scores.json"
    assert p.exists()
    data = json.loads(p.read_text())
    assert data["run_id"] == "2026-05-19-1547-a7f3"
    assert data["status"] == "completed"
    assert data["rollups"]["book"]["humanness"] == 67.0
