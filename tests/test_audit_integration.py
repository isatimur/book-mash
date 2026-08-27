"""book-mash audit integration — the judge -> wrapper path.

The default (non-live) test drives a real judge through pydantic-ai's TestModel
and asserts an audit record is produced: it proves book-mash's judges route
their outbound call through the audit wrapper. It does NOT satisfy contract
DoD item 1 (which needs a real provider's response metadata) — the ``live``
test below does that; run it with a key:

    ANTHROPIC_API_KEY=... poetry run pytest -m live tests/test_audit_integration.py
"""

import json
import os

import pytest
from pydantic_ai.models.test import TestModel

import mash_core.audit as audit
from book_mash.judges import voice as voice_mod
from book_mash.judges.voice import VoiceJudge
from mash_core import JudgeInput

_INPUT = JudgeInput(
    unit_id="chapter:test",
    unit_type="chapter",
    unit_text="A short chapter under test. It makes one concrete claim.",
    dim_name="voice",
    context={"voice_baseline_excerpts": ["Baseline sentence one.", "Baseline two."]},
)


async def test_judge_call_routes_through_wrapper(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_AUDIT_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_AUDIT_REPO", "book-mash")
    monkeypatch.setenv("AUDIT_WRAPPER_ENABLED", "1")
    # make the judge build a deterministic TestModel instead of a real provider
    monkeypatch.setattr(voice_mod, "build_judge_model",
                        lambda: (TestModel(), "claude-sonnet-4-6"))

    judge = VoiceJudge()
    await judge.judge(_INPUT)

    records = [json.loads(x) for x in (tmp_path / audit.AUDIT_LOG_NAME).read_text().splitlines() if x]
    assert len(records) == 1
    rec = records[0]
    assert rec["repo"] == "book-mash"
    assert rec["provider"] == "anthropic"
    assert rec["caller"].startswith("book_mash.judges.voice")
    assert rec["tokens_in"] > 0 and rec["tokens_out"] > 0
    assert set(rec) == {"ts", "repo", "caller", "provider", "model",
                        "prompt", "response", "tokens_in", "tokens_out"}


@pytest.mark.live
async def test_live_judge_audit_matches_provider_metadata(tmp_path, monkeypatch):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("no ANTHROPIC_API_KEY")
    monkeypatch.setenv("LLM_AUDIT_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_AUDIT_REPO", "book-mash")
    monkeypatch.setenv("AUDIT_WRAPPER_ENABLED", "1")

    judge = VoiceJudge()
    await judge.judge(_INPUT)

    rec = json.loads((tmp_path / audit.AUDIT_LOG_NAME).read_text().splitlines()[0])
    assert rec["tokens_in"] and rec["tokens_in"] > 0
    assert rec["tokens_out"] and rec["tokens_out"] > 0
    assert rec["repo"] == "book-mash"
    assert rec["provider"] == "anthropic"
