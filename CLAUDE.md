# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (dev environment)
poetry install

# Run tests (fast, no LLM calls)
poetry run pytest

# Run a single test file
poetry run pytest tests/test_cache.py

# Run live tests (hit real LLM APIs — costs money)
poetry run pytest -m live

# Type check
poetry run mypy book_mash

# Lint
poetry run ruff check book_mash tests

# CLI (from consumer repo after install, or via poetry run for local dev)
poetry run book-mash init
poetry run book-mash measure --config ./book-mash.toml --dry-run
poetry run book-mash measure --config ./book-mash.toml
```

Required env: `ANTHROPIC_API_KEY`. Optional: `OPENAI_API_KEY` — enables embedding-based redundancy prefilter (otherwise redundancy falls back to direct Haiku comparisons at ~3× cost).

## Architecture

The engine scores a manuscript in a single pipeline pass (`run_measurement`) and writes three output layers: a JSON ledger, a Markdown report, and per-chapter annotation sidecars.

book-mash depends on `mash_core`, a sibling package that holds the provider-agnostic judge contract (`JudgeDim`, `JudgeScore`, `JudgeInput`), the model factory, retry/backoff, and pricing logic; it's currently a local path dependency (`../mash-core` in `pyproject.toml`) since it isn't published anywhere yet.

### Data flow

```
book-mash.toml
  → load_config()           → BookMashConfig
  → load_chapters()         → list[Chapter → Section → Paragraph]
  → load_claims_index()     → list[Claim]
  → run_measurement()       → Run (with all JudgeScores + rollups)
  → write_ledger / render_report / write_annotations
```

### Score anatomy

- **Native scores**: produced directly by a judge at its declared `unit_type` (paragraph / section / chapter).
- **Rollup scores**: `paragraph → section → chapter` via `rollup.py` `weighted_mean` (weights = char counts). Applied to humanness, usefulness, claim_defensibility, evidence_density.
- **Broadcast scores**: voice and redundancy are chapter-native; their scores propagate down to every paragraph with `derived=True` via `broadcast_chapter_to_paragraphs`.

### Concurrency model (`runners/measurement.py`)

Two semaphores: `_CONCURRENCY=3` for all judges, `_HEAVY_CONCURRENCY=1` for humanness + claim_defensibility (large prompts that blow the TPM rate limit). Heavy judges hold both semaphores; light judges hold only the global one — no deadlock risk.

### Cache (`cache.py`)

Key: `sha256(unit_text + optional_ctx_key) | dim_name | dim_version | model_id`. Lives at `<runs_dir>/cache.json` (shared across runs). Editing one chapter invalidates only that chapter's hashes; reruns typically reuse ~90% of prior calls.

### Judge pattern

`JudgeDim` (ABC), `JudgeScore`, and `JudgeInput` live in the `mash_core` package, not in `book_mash/judges/` — import them from `mash_core`. Each judge in `judges/` subclasses `JudgeDim` and declares `name`, `unit_type`, `model_id` as `ClassVar`. `judge(input: JudgeInput) -> JudgeScore` is the only required method. `context_cache_key()` is overridden when the judge's score depends on context beyond the unit text (e.g. a mutable claims ledger), so cache misses on ledger changes are correct.

### Adding a new judge

1. Create `judges/<dim>.py`, subclass `JudgeDim`, set the three `ClassVar`s, implement `judge()`.
2. Register it in `runners/measurement.py:_build_judges()`.
3. Update `runners/planner.py` so `--dry-run` accounts for the new dim.
4. Bump `DIM_REGISTRY_VERSION` in `judges/registry.py` to invalidate cached scores for this dim.

### Key models

- `JudgeScore` — the atomic unit of all output. A `mash_core` type, re-exported and used throughout book-mash. Fields: `dim_name`, `unit_id`, `score_0_100` (None on error), `label` (strong/moderate/weak/fail/error), `reasoning`, `evidence_refs`, `model`, `cost_usd`, `derived`.
- `JudgeInput` — what a judge receives, also a `mash_core` type: `unit_id`, `unit_type`, `unit_text`, `dim_name`, `context` dict.
- `Run` — top-level result, defined in book-mash: version stamps, `total_cost_usd`, `status` (completed / halted_budget), `rollups` dict, and the full `scores` list.

### Test conventions

- Tests default to `asyncio_mode = "auto"` (pytest-asyncio).
- Tests that call real LLMs must be marked `@pytest.mark.live`; they are excluded from the default run (`addopts = "-m 'not live'"`).
- Fixtures live in `tests/fixtures/`.
