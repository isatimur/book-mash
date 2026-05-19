# book-mash

MASH-style multi-judge measurement engine for book manuscripts.

Generic engine — any book project can use it by providing a `book-mash.toml` config.

See the design spec in the consumer repo at `docs/superpowers/specs/2026-05-19-book-mash-design.md` for architecture, judge dimensions, and rollup semantics.

## Quickstart

```bash
poetry install
poetry run book-mash measure --config /path/to/book-mash.toml
```

Requires `ANTHROPIC_API_KEY` in environment. Optional: `VOYAGE_API_KEY` (preferred for embeddings) or `OPENAI_API_KEY` (fallback).
