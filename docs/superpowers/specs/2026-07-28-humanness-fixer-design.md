# humanness fixer — design

Status: approved by user. No code has been written yet. Spans three repos:
`de-slop` (github.com/isatimur/de-slop), `mash-core`, and `book-mash`.

## What

Book-mash's `humanness` judge currently only *measures* — it scores a paragraph
and stops. This adds a *fixer*: when a paragraph scores weak/fail on humanness,
run it through a rewrite-and-reverify loop, and either surface the suggested fix
as an annotation (default) or write it back into the manuscript (opt-in).

The rewrite loop is powered by `de-slop` — a separate, pre-existing, atomic
project (github.com/isatimur/de-slop, MIT) that already does exactly this job
for arbitrary prose, with its own rubric that closely matches book-mash's
`humanness` rubric (same strong/moderate/weak/fail bands, same "hostile
editor" framing).

## Why

De-slop's own design doc states it was deliberately decoupled from its
originating project's judges specifically so it could be reused elsewhere —
that reuse never happened until now. Book-mash independently reimplemented an
AI-slop *detector* (the `humanness` judge) without that reuse; this design
closes the loop de-slop was built for, and gives de-slop a real second
consumer instead of standing alone — the same validation mash-core got from
book-mash depending on it.

## Cross-repo research (grounds this design in what's real, not assumed)

- De-slop's detector layer (`flag()`, `score()` in `scripts/flag_slop.py`) is
  **already** a real, versioned (`__version__ = "0.4.1"`), zero-dependency,
  pip/git-installable Python library — `pyproject.toml` declares it as a real
  package (`py-modules = ["flag_slop"]`). No new API needed there.
- What's **not** packaged: `references/rubric.md`, `guardrails.md`,
  `examples.md` — the markdown content that drives the actual rewrite loop.
  These exist only as loose repo files consumed by `SKILL.md`'s
  agent-executed instructions, not as importable package data.
- The rewrite/judge/self-score loop itself (de-slop's steps 2-5) has **no
  code** — it's natural-language instructions for an LLM agent to execute
  interactively. There is no `rewrite()` function. Book-mash's fixer must be
  its own Agent, *prompted from* de-slop's rubric content, not a call into
  de-slop's execution.
- De-slop's own design principle #6: **"Report, don't overwrite. The human
  or calling agent decides what to accept."** This design's default behavior
  (annotations, not auto-mutation) exists specifically to honor that
  principle rather than contradict it. Book-mash's own README roadmap sketch
  for "autoresearch" implied auto-accept-if-better; that behavior is
  preserved but demoted to an explicit opt-in (`--auto-accept`), not the
  default.

## Part 1: de-slop changes (own repo)

Package `references/rubric.md`, `guardrails.md`, `examples.md` as importable
package data (`package_data`/`MANIFEST.in`), and add:

```python
def load_reference(name: str) -> str:
    """Load a reference doc by name (e.g. "rubric", "guardrails", "examples").
    Raises FileNotFoundError for unknown names."""
```

in `scripts/flag_slop.py`, using `importlib.resources` so it works whether
installed from a wheel or run from a checkout. No change to `flag()`/`score()`
signatures — they're already stable. Version bump to `0.5.0`, changelog entry
documenting programmatic reference-content consumption as a supported use
case (distinct from, and additive to, the existing CLI/skill usage).

## Part 2: mash-core changes (own repo)

A minimal, generic fixer contract — deliberately small, since there is one
real implementation and no second consumer yet (the same restraint applied to
`JudgeDim` originally):

```python
class FixResult(BaseModel):
    unit_id: str
    original_text: str
    fixed_text: str | None   # None: hollow span, capped at 3 passes, or already strong
    accepted: bool
    reason: str
    cost_usd: float


class FixerDim(ABC):
    name: ClassVar[str]

    @abstractmethod
    async def fix(self, input: JudgeInput) -> FixResult:
        ...
```

Reuses `JudgeInput` (already in mash-core). No registry, no plugin/discovery
mechanism — book-mash imports `FixerDim`/`FixResult` directly and implements
one class. Exported from `mash_core`'s public `__init__.py` alongside the
existing judge-core exports.

## Part 3: book-mash changes (own repo)

- **New dependency**: `de-slop`, wired as a local path/git dependency in
  `pyproject.toml`, same pattern as `mash-core` (`{path = "../de-slop", develop
  = true}` in `[tool.poetry.dependencies]`; bare `"de-slop"` in
  `[project.dependencies]`).
- **New file**: `book_mash/judges/humanness_fixer.py` — `HumannessFixer`
  implementing `mash_core.FixerDim`. On `fix(input)`:
  1. Pre-flag via `de_slop.flag_slop.flag(input.unit_text)` — candidate spans.
  2. Judge the paragraph via an Agent built with
     `mash_core.build_judge_model()`, prompted with
     `de_slop.flag_slop.load_reference("rubric")`.
  3. Triage: rewordable → attempt rewrite; hollow → return
     `FixResult(fixed_text=None, accepted=False, reason="hollow: ...")`.
  4. Rewrite loop, prompted with `load_reference("guardrails")` and
     `load_reference("examples")`, capped at 3 passes (matching de-slop's own
     cap), self-scoring each attempt against the rubric.
  5. Re-score the best candidate via the **existing** `HumannessJudge`
     (book-mash's own judge, unchanged) to confirm the rewrite actually
     improves book-mash's own score, not just de-slop's internal self-score.
  6. `accepted = True` only if the re-judged score exceeds the original
     score; otherwise `accepted=False` with the reason recorded (e.g. "self-
     scored strong but book-mash re-judge did not confirm improvement").
- **Output — default (non-destructive)**: accepted `FixResult`s render into
  the existing per-chapter annotation sidecars (`output/annotations.py`) as a
  new "suggested rewrite" block per flagged paragraph — original text, fixed
  text, old/new score, side by side. The manuscript file itself is never
  touched.
- **`--auto-accept` flag**: when set, accepted fixes are written back into the
  chapter markdown directly. The run log records exactly which paragraphs
  changed and their old/new scores, so the change is auditable even though
  the file was mutated.
- **New CLI command**: `book-mash fix --config ./book-mash.toml [--dry-run]
  [--auto-accept]`, mirroring `measure`'s structure. `--dry-run` counts
  candidate paragraphs (weak/fail humanness from the most recent cache) and
  estimates cost — no LLM calls — same contract as `measure --dry-run`.
- **Budget**: `fix` gets its own `max_cost_usd` check, separate from
  `measure`'s. Materially more expensive per paragraph than measuring alone
  (rewrite + self-score, up to 3 passes, then a re-judge call) — worth a
  distinct budget rather than sharing `measure`'s cap.

## Non-goals

- No fixers for the other five dimensions (voice, usefulness, evidence
  density, claim defensibility, redundancy) — the interface stays open,
  nothing else is built against it yet.
- No changes to `measure`'s existing behavior or output shape — `fix` is a
  new, additive command.
- De-slop's CLI/skill-facing behavior (the interactive agent-executed loop)
  is unchanged — this only adds a second, programmatic consumption path
  alongside it.
- No PyPI publication work for de-slop as part of this — it stays a git
  dependency for now, same as mash-core.
