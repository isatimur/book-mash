from pathlib import Path

from book_mash.config import load_config


FIXTURE_CONFIG = Path(__file__).parent / "fixtures" / "mini_book" / "book-mash.toml"


def test_load_config():
    cfg = load_config(str(FIXTURE_CONFIG))
    assert cfg.chapters_glob.endswith("public/drafting/*.md")
    assert cfg.claims_dir.endswith("claims")
    assert cfg.voice_baseline_chapters == ["chapter-01.md"]
    assert cfg.runs_dir.endswith(".book-mash-runs")
    assert cfg.max_cost_usd == 1.0


def test_paths_resolve_relative_to_config_file():
    cfg = load_config(str(FIXTURE_CONFIG))
    # chapters_glob is relative to the directory containing book-mash.toml
    assert "fixtures/mini_book" in cfg.chapters_glob
