# mash-core extraction — design

Status: proposed, pending user approval. No code has been written or moved.

## Why

`book-mash` and Swiirl Pulse's `mash` (Meeting Agent Simulation Harness) independently built
"score something against a rubric via an LLM judge" plumbing. book-mash's version
(`book_mash/judges/{_model_factory,_model_settings,_pricing,_retry,base}.py` + the
`JudgeScore` types in `models.py`) is the cleaner, more generic of the two — mash's own
judge/eval code isn't even a stable extraction source right now (it lives in a separate
repo, `ai-worker`, and only exists as source on a non-current branch,
`feature/ai-eval-framework`; the branch `mash` currently imports from has only stale
`.pyc` files for that path).

Goal: pull book-mash's judge-core cluster into its own project now, so it exists as a
real, independent, pip-installable thing — not because mash is ready to consume it today,
but because book-mash's version is already good enough to stand alone, and waiting for a
second consumer to materialize before extracting anything indefinitely defers the "make
it a separated project" ask.

## Prior art check (why not just adopt an existing framework)

Researched Promptfoo, DeepEval, Ragas, and Inspect AI (2026 landscape). All four are
regression-test/benchmark harnesses: fixed test cases, pytest- or YAML-config-style,
built for "does this output still pass" gating. None target the actual shape of this
problem — cost-capped, cache-keyed judge scoring rolled up through a document hierarchy
(paragraph → section → chapter, or utterance → turn → meeting) against a live, changing
corpus. Building mash-core isn't reinventing an existing wheel.

One real correction from that research: both `book-mash` and `mash` already depend on
**pydantic-ai**, which is itself a provider-agnostic model layer (Anthropic / OpenAI /
OpenRouter-compatible providers, structured Pydantic output). `_model_factory.py` is
already essentially a thin pydantic-ai wrapper selecting a provider from config — it
should stay framed that way, not as a competing provider-abstraction. The genuinely novel,
worth-extracting surface is the judge-specific layer built on top of pydantic-ai:
`JudgeDim`, `JudgeScore`, the per-model pricing/cost-ledger table, and the judge-tuned
retry/backoff wrapper.

## What moves

From `book-mash/book_mash/judges/`:

- `_model_factory.py` — pydantic-ai model construction per provider config (Anthropic /
  OpenRouter / OpenAI-compatible), selected via env var.
- `_model_settings.py` — per-judge model settings (incl. the `max_tokens=2048` cap fixed
  this session).
- `_pricing.py` — `_PRICE_PER_TOKEN` table for cost tracking.
- `_retry.py` — `run_with_backoff`, judge-call-tuned exponential backoff.
- `base.py` — the `JudgeDim` ABC (~20 lines, already clean).
- The `JudgeScore`-family types currently in `models.py`.

Moved essentially verbatim; package renamed; book-mash's six judge implementations
import from the new package instead of local relative imports.

## Naming

`mash-core` has no exact PyPI/GitHub collision, but bare "mash" is noisy — an unrelated
shell utility, an ML/CV utils lib (Moonshine Labs), and a multi-agent SDK all use it.
Not a blocker, but worth a more distinctive name before publishing (e.g. `mash-judge-core`
or similar) to keep `pip install` / search unambiguous. Left open for the user to decide;
default to `mash-core` if no preference is expressed.

## Location & structure

New sibling project: `/Users/timur_isachenko/Dev/LifeOS/mash-core/` (or renamed
equivalent), own git repo, own `pyproject.toml`, MIT license (matches book-mash).
Package name `mash_core`. book-mash adds it as a Poetry path/git dependency.

## Non-goals (explicitly not moving)

- mash's autoresearch loop, Postgres/SQLAlchemy persistence, bot-adapter, transcript
  replay parsers.
- book-mash's rollup/broadcast logic and manuscript corpus model (`Chapter`/`Section`/
  `Paragraph`, claim retrieval).

None of this is generic enough today, and mash's own judge code isn't in a state where
designing a shared seam against it is possible yet — this stays deferred, not designed
around speculatively.

## Migration plan

1. `git init` new repo, scaffold `pyproject.toml` / `LICENSE` / `README.md` mirroring
   book-mash's conventions.
2. Move the 6 files in, preserving origin note in README (fresh copy, not a git-history
   carry-over, since it's crossing repos).
3. Add direct unit tests (currently only covered indirectly via book-mash's judge tests).
4. Wire book-mash to depend on it locally (path dependency first); update judge files'
   imports.
5. Full book-mash test suite green (pytest, mypy, ruff) before calling it done.

## Effort

Half a day to a day. Small, mechanical, low-risk.

## Open question carried forward (not yet answered by user)

Whether to also generalize `JudgeDim`/`JudgeScore` now with mash's shape in mind
(a documented seam, ~few extra days, some speculative risk) versus keep it a verbatim
lift of book-mash's version until a second real consumer exists. Default, absent a
reply: verbatim lift — narrower, ships faster, avoids guessing an interface against code
that can't be tested against it yet.
