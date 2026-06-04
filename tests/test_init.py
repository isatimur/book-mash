"""Tests for the init wizard. Pure render + load round-trip; no interactive driving."""

from pathlib import Path

from book_mash.config import load_config
from book_mash.init import render_toml


def test_render_toml_with_baseline():
    out = render_toml(
        chapters_glob="public/drafting/*.md",
        claims_dir="claims/",
        evidence_dir="evidence/",
        voice_baseline_chapters=["Chapter 1.md", "Chapter 7.md"],
        runs_dir=".book-mash-runs/",
        max_cost_usd=10.0,
    )
    assert "[corpus]" in out
    assert "[output]" in out
    assert "[budget]" in out
    assert 'chapters_glob = "public/drafting/*.md"' in out
    assert 'voice_baseline_chapters = ["Chapter 1.md", "Chapter 7.md"]' in out
    assert "max_cost_usd = 10.0" in out


def test_render_toml_without_baseline_omits_key():
    """When the user passes no baseline, the key is omitted — letting the
    config loader default cleanly."""
    out = render_toml(
        chapters_glob="*.md",
        claims_dir="c/",
        evidence_dir="e/",
        voice_baseline_chapters=[],
        runs_dir="runs/",
        max_cost_usd=5.0,
    )
    assert "voice_baseline_chapters" not in out


def test_render_then_load_round_trip(tmp_path: Path):
    """End-to-end: render → write → load_config → BookMashConfig.

    Catches schema drift between the wizard and the loader.
    """
    out = render_toml(
        chapters_glob="public/drafting/*.md",
        claims_dir="claims/",
        evidence_dir="evidence/",
        voice_baseline_chapters=["Chapter 1.md"],
        runs_dir=".book-mash-runs/",
        max_cost_usd=12.5,
    )
    target = tmp_path / "book-mash.toml"
    target.write_text(out)

    cfg = load_config(str(target))
    assert cfg.max_cost_usd == 12.5
    assert cfg.voice_baseline_chapters == ["Chapter 1.md"]
    # Paths are resolved relative to the toml file location
    assert cfg.claims_dir.endswith("/claims")
    assert cfg.runs_dir.endswith("/.book-mash-runs")
