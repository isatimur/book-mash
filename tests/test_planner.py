"""Tests for the dry-run planner — counts units + estimates cost without LLM calls."""

from pathlib import Path

from book_mash.config import load_config
from book_mash.runners.planner import plan_measurement


_FIXTURE_TOML = Path(__file__).parent / "fixtures" / "mini_book" / "book-mash.toml"


def _cfg():
    return load_config(str(_FIXTURE_TOML))


def test_plan_counts_units_from_corpus():
    plan = plan_measurement(_cfg())
    # mini_book is 2 chapters with at least one section and one paragraph each
    assert plan.n_chapters == 2
    assert plan.n_sections >= 2
    assert plan.n_paragraphs >= 2


def test_plan_dim_count_matches_six():
    plan = plan_measurement(_cfg())
    assert len(plan.dims) == 6
    names = {d.dim_name for d in plan.dims}
    assert names == {
        "humanness",
        "voice",
        "usefulness",
        "evidence_density",
        "claim_defensibility",
        "redundancy",
    }


def test_plan_paragraph_dims_call_count_equals_n_paragraphs():
    plan = plan_measurement(_cfg())
    para_dims = [d for d in plan.dims if d.granularity == "paragraph"]
    assert len(para_dims) == 3  # humanness + usefulness + claim_defensibility
    for d in para_dims:
        assert d.call_count == plan.n_paragraphs


def test_plan_chapter_dims_call_count_equals_n_chapters():
    plan = plan_measurement(_cfg())
    chapter_dims = [d for d in plan.dims if d.granularity == "chapter"]
    assert len(chapter_dims) == 2  # voice + redundancy
    for d in chapter_dims:
        assert d.call_count == plan.n_chapters


def test_plan_total_calls_sums_dims():
    plan = plan_measurement(_cfg())
    assert plan.total_calls == sum(d.call_count for d in plan.dims)


def test_plan_total_cost_sums_dims():
    plan = plan_measurement(_cfg())
    expected = sum(d.estimated_cost_usd for d in plan.dims)
    assert plan.total_estimated_cost_usd == expected


def test_plan_costs_are_positive_when_units_exist():
    plan = plan_measurement(_cfg())
    if plan.n_paragraphs > 0:
        for d in plan.dims:
            assert d.estimated_cost_usd > 0
        assert plan.total_estimated_cost_usd > 0


def test_plan_carries_budget_cap_from_config():
    cfg = _cfg()
    plan = plan_measurement(cfg)
    assert plan.budget_cap_usd == cfg.max_cost_usd
