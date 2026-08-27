"""Contract DoD item 6: the audit + PII logs are gitignored and a fresh
``git status`` never shows them as trackable."""

import subprocess
from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent
_LOG_FILES = [
    "logs/llm-call-audit.jsonl",
    "logs/llm-call-pii-flags.jsonl",
    "logs/llm-call-coverage-gaps.jsonl",
    "logs/.llm-call-marker.json",
]


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=_REPO, capture_output=True, text=True)


def test_log_paths_are_gitignored():
    for rel in _LOG_FILES:
        rc = _git("check-ignore", "-q", rel).returncode
        assert rc == 0, f"{rel} is NOT gitignored"


def test_git_status_never_shows_log_files():
    (_REPO / "logs").mkdir(exist_ok=True)
    created = []
    try:
        for rel in _LOG_FILES:
            p = _REPO / rel
            if not p.exists():
                p.write_text('{"probe": true}\n')
                created.append(p)
        status = _git("status", "--porcelain").stdout
        for rel in _LOG_FILES:
            assert rel not in status, f"{rel} appeared in git status: {status!r}"
    finally:
        for p in created:
            p.unlink(missing_ok=True)
