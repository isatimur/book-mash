# book-mash

A multi-judge measurement engine for book manuscripts. Six independent judges score prose against three craft dimensions (humanness, voice, usefulness) and three epistemic dimensions (evidence density, claim defensibility, redundancy), at paragraph / section / chapter / book granularity. Output is a JSON ledger + Markdown report + per-chapter sidecar annotations.

The intent is to turn vibes ("this chapter feels weak") into a defensible, reproducible signal — and to give the author a measurement layer they can re-run after every editorial pass.

Generic engine: any book project that exposes a `book-mash.toml` can use it. The first consumer is [`ai-engineer-book`](https://github.com/isatimur/ai-engineering-book-lab).

**Canonical design:** [`docs/design.md`](docs/design.md) (375 lines: architecture, primitives, rubrics, output schemas, error handling, testing).

---

## Install

```bash
poetry install
```

Required: `ANTHROPIC_API_KEY` in environment.
Optional: `OPENAI_API_KEY` — enables embedding-based redundancy prefilter (otherwise redundancy falls back to direct Haiku comparisons, ~3× cost on long manuscripts).

Python 3.13. Single dependency on Anthropic SDK + Pydantic + Typer + Rich. No external state — runs against a local `book-mash.toml` + the consumer repo's filesystem.

## Quickstart

From the consumer book repo:

```bash
poetry run --directory ~/Dev/LifeOS/book-mash book-mash measure --config ./book-mash.toml
```

A full v0.1 run on a 10-chapter / 80k-word manuscript: **~$3–6** in Anthropic credits, **3–5 min** wall clock with default 8-way concurrency.

---

## Config — `book-mash.toml`

```toml
[corpus]
chapters_glob = "public/drafting/*.md"             # any glob, relative to the toml's directory
claims_dir    = "claims/"                          # for evidence-density + claim-defensibility judges
evidence_dir  = "evidence/"                        # for claim-defensibility judge
voice_baseline_chapters = ["Chapter 1.md", "Chapter 7.md"]  # the "this is the right voice" exemplars

[output]
runs_dir = ".book-mash-runs/"                      # add to your repo's .gitignore

[budget]
max_cost_usd = 15.0                                # hard halt if cumulative cost crosses this
```

All paths are resolved relative to the location of the toml file. Required keys: `chapters_glob`, `claims_dir`, `evidence_dir`, `runs_dir`. Optional: `voice_baseline_chapters` (no baseline → voice judge skipped), `max_cost_usd` (defaults to $10).

---

## The six judges

| # | Dimension | Granularity | Model | What it scores |
|---|---|---|---|---|
| 1 | **humanness** | paragraph | Sonnet | Anti-AI-slop. Does this read like a thinking author, or generic AI-flavoured hedging? `reasoning` must quote the worst phrase. |
| 2 | **voice** | chapter (broadcast to sections/paragraphs as `derived=true`) | Sonnet | Does this chapter sound like the same author as the baseline chapters? Names specific drift ("longer sentences than baseline", "softer epistemic stance"). |
| 3 | **usefulness** | paragraph | Haiku | Could a practitioner change something on Monday because of this paragraph? Filler and meta-talk are `fail`. |
| 4 | **evidence_density** | section | Haiku + deterministic match | Grounded claims per ~500 words, ratio against `claims_dir`. Number is reproducible — Haiku lists candidates, code computes the ratio. |
| 5 | **claim_defensibility** | paragraph (claim-bearing only) | Sonnet | For each claim the prose makes, does it match what the claims ledger actually supports? `fail` here is a ship-blocker. |
| 6 | **redundancy** | chapter (broadcast down) | Haiku + embeddings | Does this chapter restate arguments already made earlier? Cosine-similarity prefilter narrows pairs before Haiku confirms. |

All judges: `temperature=0`, structured output, version-stamped on every run. Re-running on an unchanged corpus with a warm cache produces identical scores. Cold-cache reruns are approximately reproducible.

Rubric bands across every dim: `strong` (80–100) · `moderate` (50–79) · `weak` (20–49) · `fail` (0–19). Full rubric definitions in [`docs/design.md`](docs/design.md#the-six-judge-dimensions).

---

## CLI

```bash
book-mash measure --config ./book-mash.toml          # the v0.1 entrypoint
book-mash gate                                       # v0.2 — stub
book-mash autoresearch                               # v0.3 — stub
```

Only `measure` is implemented. `gate` and `autoresearch` are scaffolded with `@app.command()` decorators but print "not yet implemented" — they're road-mapped for v0.2 (continuous re-score after each research_pass) and v0.3 (Sonnet rewrites passage → judges score → accept-if-better).

Future flags (post-v0.1): `--chapters 5,6,8` and `--dims humanness,voice` for narrowing during development.

---

## Output

Each run writes to `<runs_dir>/<date>-<HHMM>-<short-hash>/`:

```
2026-05-19-1547-a7f3/
  run.json                                  # the Run object (small, indexable)
  scores.json                               # full JudgeScore ledger (source of truth)
  report.md                                 # rendered Markdown report
  annotations/
    chapter-01.annotations.md               # per-chapter sidecar, non-mutating
    chapter-02.annotations.md
  cache.json                                # (unit_hash, dim, version) → score
```

**`scores.json`** carries every judge score with reasoning, evidence refs, model, cost, and `derived=true` for broadcast scores. Schema in [`docs/design.md`](docs/design.md#scoresjson-shape).

**`report.md`** opens with a one-paragraph TL;DR ("Chapters 5 and 8 are the bottleneck on humanness; Ch 3 ↔ Ch 6 has the worst redundancy"), then a chapters × dims heatmap, per-chapter detail with the 3 weakest paragraphs quoted under each, voice drift table, redundancy pairs, and a ship-blockers list (every `fail` on `claim_defensibility` first).

**Annotations** are owned by the engine — re-running regenerates them, so resolved issues disappear naturally. They sit next to the draft and open in any editor.

**Cache** is keyed on `(unit_hash, dim_name, dim_version, model_id)`. Editing one chapter and re-running typically reuses ~90% of judge calls and costs **~$0.50** instead of $4–6. This is what makes the v0.2 continuous-gate flow economically viable.

---

## Determinism contract

- All judges run at `temperature=0` with structured Pydantic outputs.
- Every `Run` is version-stamped with `book_mash_version` + `dim_registry_version`.
- The cache is the real determinism contract: same `(unit_hash, dim_name, dim_version, model_id)` → bit-identical `JudgeScore`. Editing chapter text invalidates only the units that changed.
- Cold-cache reruns are *approximately* reproducible (API-side stability at `temperature=0` is not a hard guarantee, but `score_0_100` typically lands within ±2).

---

## Error handling

Philosophy: **partial runs are fine; silent lies are not.**

| Failure | Behaviour |
|---|---|
| Judge call rate-limit / 5xx / timeout | 3 retries with exponential backoff (1s → 4s → 16s), then `JudgeScore(label="error", score=null)`. Run continues; rollups exclude null scores and annotate the chapter rollup with `n_errors`. |
| Cumulative cost crosses `max_cost_usd` | Run halts cleanly. Partial ledger written. `Run.status = "halted_budget"`. |
| Malformed chapter markdown | Permissive loader — unparseable sections become a single paragraph holding the section's text. Warning surfaces in the report. |
| Judge references a missing claim ledger entry | Score is *not* invalidated — the missing reference is itself a finding, surfaced under "broken claim references". Catches `claims/` rot. |

**Hard fails** (run aborts cleanly): missing/malformed config; no chapters matched by `chapters_glob`; API key missing for a model the dim registry requires.

---

## Roadmap

- **v0.1 (current):** `measure` — single full-corpus run. Six judges, three output layers. Ships.
- **v0.2 (planned):** `gate` — re-score after each research_pass, regression-detection on the existing cache. ~1 day of work; the pipeline is already idempotent.
- **v0.3 (planned):** `autoresearch` — Sonnet mutates a weak passage, the same six judges score the candidate, accept-if-better loop with bandit-like budget allocation. Ports the autoresearch loop from `mash/` (Plan 3 / Plan 6).
- **Deferred:** web UI, multi-book comparison, sharing dim definitions back to `mash/` via an extracted `eval-core/`.

---

## Project tree

```
book-mash/
  pyproject.toml              # Poetry; Python 3.13
  README.md                   # this file
  docs/
    design.md                 # canonical architecture spec (375 lines)
  book_mash/
    cli.py                    # Typer entrypoint
    config.py                 # book-mash.toml loader
    corpus/                   # Chapter→Section→Paragraph markdown parser
    judges/                   # six judge implementations + base class + pricing
    runners/
      measurement.py          # the v0.1 pipeline (260 lines)
      gate.py                 # v0.2 stub
      autoresearch.py         # v0.3 stub
    output/                   # ledger, report, annotations renderers
  tests/                      # ~10 test files: config, models, judges, rollups, output
  scripts/                    # repo-local utilities
```

---

## Why this exists

Editorial gut-check is the single most expensive resource in book-writing — the author can read each chapter maybe a handful of times before fatigue sets in. Once execution gets cheap (drafting prose with AI assistance, generating diagrams, source-anchoring at scale), the bottleneck is review. `book-mash` is the bet that the same scaffolding-makes-reliability argument the consumer book itself makes — evals as a control system, not a chore — applies one level up to manuscript review. Judges replace gut-check on the routine cases so author attention concentrates on the consequential ones.

The engine is generic; the dim registry, rubrics, and output schemas are deliberately separable from the consumer's corpus shape. Other long-form prose projects (technical books, research reports, internal manuscripts) should be able to wire up a `book-mash.toml` and use the same engine with minimal config-time work.
