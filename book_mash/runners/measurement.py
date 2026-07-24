import asyncio
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from book_mash.cache import JudgeScoreCache, content_hash
from book_mash.config import BookMashConfig
from book_mash.corpus.claim_retrieval import retrieve_relevant_claims
from book_mash.corpus.claims_index import load_claims_index
from book_mash.corpus.loader import compute_snapshot_hash, load_chapters
from book_mash.corpus.models import Chapter
from book_mash.judges.base import JudgeDim
from book_mash.judges.claim_defensibility import ClaimDefensibilityJudge
from book_mash.judges.embeddings import pick_embedding_client
from book_mash.judges.evidence_density import EvidenceDensityJudge
from book_mash.judges.humanness import HumannessJudge
from book_mash.judges.models import JudgeInput, JudgeLabel, JudgeScore
from book_mash.judges.redundancy import RedundancyJudge, prefilter_candidates
from book_mash.judges.registry import DIM_REGISTRY_VERSION
from book_mash.judges.usefulness import UsefulnessJudge
from book_mash.judges.voice import VoiceJudge
from book_mash.rollups import (
    broadcast_chapter_to_paragraphs,
    rollup_paragraph_to_section,
    rollup_section_to_chapter,
    weighted_mean,
)
from book_mash.runners.models import Run, RunStatus
from book_mash.version import __version__


# Lowered from 8 to reduce Anthropic rate-limit failures that left prior runs
# partial (humanness + claim_defensibility under-covered). As of the retry layer
# (judges/_retry.py), transient 429/529s are retried with backoff rather than
# turned into permanent coverage gaps, so this is now a throughput knob, not a
# coverage one — raise if your tier allows.
# Throughput knob, not a coverage one (the retry layer recovers transient 429s).
# Sized for Anthropic's tight TPM ceiling; override via env for providers with
# generous limits (e.g. OpenRouter/DeepSeek): BOOK_MASH_CONCURRENCY=16.
_CONCURRENCY = int(os.environ.get("BOOK_MASH_CONCURRENCY", "3"))

# The retry layer recovers *transient* throttles, but run 4 (0cb7) still left
# humanness at 70% and claim_defensibility at 71% while the small-prompt judges
# hit 100%. The cause is sustained, not transient: those two judges carry by far
# the largest prompts (humanness bundles the surrounding paragraphs; claim_
# defensibility bundles the relevant ledger), so running several at once pushes
# the per-key tokens-per-minute rate past the ceiling faster than backoff can
# recover. Serialize just those two through a separate, tighter semaphore so the
# heavy calls drip under the TPM ceiling while usefulness/voice/evidence/
# redundancy keep the full _CONCURRENCY throughput.
_HEAVY_JUDGES = frozenset({"humanness", "claim_defensibility"})
# Serializes the two large-prompt judges under Anthropic's TPM ceiling. The hard
# per-unit cap below already prevents a stuck heavy unit from wedging the batch, so
# this is purely a rate knob — raise it for generous providers:
# BOOK_MASH_HEAVY_CONCURRENCY=12.
_HEAVY_CONCURRENCY = int(os.environ.get("BOOK_MASH_HEAVY_CONCURRENCY", "1"))

# Hard per-unit wall-clock cap. Belt-and-suspenders on top of the per-request
# httpx timeout (judges/_model_settings.py, 120s) and run_with_backoff's bounded
# retries. The per-request timeout only bounds a single in-flight HTTP request; it
# cannot catch a unit that wedges with NOTHING in flight (a code-level asyncio
# stall: 0 CPU, 0 network connections). That failure mode is dangerous here because
# claim_defensibility/humanness run under _HEAVY_CONCURRENCY=1 — a single stuck unit
# holds heavy_sem forever and every other heavy unit blocks behind it, hanging the
# whole run. This cap guarantees each unit either finishes or is force-cancelled and
# recorded as an ERROR score within HARD_CAP_SECONDS, and the semaphore it held is
# always released (see run_judge's finally), so no single unit can wedge the batch
# for any reason (deadlock, half-open socket, model stall).
HARD_CAP_SECONDS = 180.0


async def run_measurement(cfg: BookMashConfig) -> Run:
    chapters = load_chapters(cfg.chapters_glob, cfg.skip_sections)
    claims_index = load_claims_index(cfg.claims_dir)
    voice_baseline_excerpts = _load_voice_baseline(chapters, cfg.voice_baseline_chapters)
    snapshot_hash = compute_snapshot_hash(chapters)

    run_id = _make_run_id(snapshot_hash)
    run_dir = Path(cfg.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = JudgeScoreCache(run_dir.parent / "cache.json")

    started = datetime.now(UTC).isoformat()
    judges = _build_judges()

    try:
        embedder = pick_embedding_client()
    except RuntimeError:
        embedder = None
    chapter_summaries = [{"id": c.id, "summary": _summarize_chapter(c)} for c in chapters]

    all_scores: list[JudgeScore] = []
    cost = 0.0
    sem = asyncio.Semaphore(_CONCURRENCY)
    heavy_sem = asyncio.Semaphore(_HEAVY_CONCURRENCY)

    @asynccontextmanager
    async def _slots(judge_name: str):
        # Heavy judges hold their own slot first, then a global slot. Light judges
        # only ever take the global slot, so there is no circular wait (no deadlock)
        # and at most _HEAVY_CONCURRENCY heavy calls are in flight at once.
        #
        # Acquire/release explicitly in finally (rather than `async with sem`) so that
        # if the wrapped judge call is force-cancelled by the hard cap, the slot is
        # still handed back. `async with` would also release on cancellation, but the
        # explicit finally makes the guarantee unmissable and local to one place: a
        # timed-out unit can NEVER leave heavy_sem/sem held and wedge the rest of the
        # batch behind it.
        heavy = judge_name in _HEAVY_JUDGES
        if heavy:
            await heavy_sem.acquire()
        await sem.acquire()
        try:
            yield
        finally:
            sem.release()
            if heavy:
                heavy_sem.release()

    async def run_judge(judge: JudgeDim, input: JudgeInput) -> JudgeScore:
        nonlocal cost
        ctx_key = judge.context_cache_key(input)
        unit_hash = content_hash(input.unit_text + "|ctx:" + ctx_key) if ctx_key else content_hash(input.unit_text)
        cached = cache.get(unit_hash, judge.name, DIM_REGISTRY_VERSION, judge.model_id)
        if cached is not None:
            return cached
        async with _slots(judge.name):
            if cost >= cfg.max_cost_usd:
                return JudgeScore(
                    dim_name=judge.name, unit_id=input.unit_id, score_0_100=None,
                    label=JudgeLabel.ERROR, reasoning="halted: budget", evidence_refs=[],
                    model=judge.model_id, cost_usd=0.0, derived=False,
                )
            try:
                # Hard wall-clock cap per unit. If the judge coroutine stalls for any
                # reason (deadlock, half-open socket the per-request timeout misses,
                # model never responding), wait_for cancels it and we record an ERROR
                # score instead of hanging the whole run. The _slots finally releases
                # the semaphore on this cancellation path, so the next unit proceeds.
                score = await asyncio.wait_for(judge.judge(input), timeout=HARD_CAP_SECONDS)
            except (asyncio.TimeoutError, TimeoutError):
                return JudgeScore(
                    dim_name=judge.name, unit_id=input.unit_id, score_0_100=None,
                    label=JudgeLabel.ERROR,
                    reasoning=f"hard timeout after {HARD_CAP_SECONDS:.0f}s",
                    evidence_refs=[], model=judge.model_id, cost_usd=0.0, derived=False,
                )
            cost += score.cost_usd
            # Never cache errors: a cached 402/timeout would replay forever and
            # make heal-reruns no-ops. Errors should be retried on the next run.
            if score.label != JudgeLabel.ERROR:
                cache.put(unit_hash, judge.name, DIM_REGISTRY_VERSION, judge.model_id, score)
            return score

    para_tasks: list = []
    for chapter in chapters:
        for section in chapter.sections:
            for i, paragraph in enumerate(section.paragraphs):
                surrounding = _surrounding(section.paragraphs, i)
                para_tasks.append(run_judge(judges["humanness"], JudgeInput(
                    unit_id=paragraph.id, unit_type="paragraph", unit_text=paragraph.text,
                    dim_name="humanness", context={"surrounding_paragraphs": surrounding},
                )))
                para_tasks.append(run_judge(judges["usefulness"], JudgeInput(
                    unit_id=paragraph.id, unit_type="paragraph", unit_text=paragraph.text,
                    dim_name="usefulness", context={"chapter_title": chapter.title},
                )))
                # P0 grounding fix: retrieve relevant claims from the FULL ledger by
                # lexical relevance (NOT the editorial candidate_chapters tag), so that
                # well-supported cross-chapter content (e.g. ch-5 Sampath/Anthropic ->
                # claims #32/#33) is no longer routed away from the judge. The judge
                # also receives each claim's supporting quotes (see claim_defensibility).
                relevant_ledger = retrieve_relevant_claims(paragraph.text, claims_index)
                para_tasks.append(run_judge(judges["claim_defensibility"], JudgeInput(
                    unit_id=paragraph.id, unit_type="paragraph", unit_text=paragraph.text,
                    dim_name="claim_defensibility", context={"relevant_ledger": relevant_ledger},
                )))
    para_scores = await asyncio.gather(*para_tasks)
    all_scores.extend(para_scores)

    sec_tasks: list = []
    for chapter in chapters:
        for section in chapter.sections:
            section_text = "\n\n".join(p.text for p in section.paragraphs)
            sec_tasks.append(run_judge(judges["evidence_density"], JudgeInput(
                unit_id=section.id, unit_type="section", unit_text=section_text,
                dim_name="evidence_density",
                context={"claims_index": claims_index},
            )))
    sec_scores = await asyncio.gather(*sec_tasks)
    all_scores.extend(sec_scores)

    chapter_tasks: list = []
    earlier_summaries: list[dict] = []
    for i, chapter in enumerate(chapters):
        chapter_tasks.append(run_judge(judges["voice"], JudgeInput(
            unit_id=f"chapter:{chapter.id}", unit_type="chapter", unit_text=chapter.full_text,
            dim_name="voice", context={"voice_baseline_excerpts": voice_baseline_excerpts},
        )))
        if embedder is not None:
            candidate_ids = await prefilter_candidates(
                embedder, chapter_summaries[i]["summary"], earlier_summaries
            )
        else:
            candidate_ids = []
        chapter_tasks.append(run_judge(judges["redundancy"], JudgeInput(
            unit_id=f"chapter:{chapter.id}", unit_type="chapter", unit_text=chapter.full_text,
            dim_name="redundancy",
            context={
                "earlier_chapter_summaries": list(earlier_summaries),
                "candidate_overlap_chapter_ids": candidate_ids,
            },
        )))
        earlier_summaries.append(chapter_summaries[i])
    chapter_scores = await asyncio.gather(*chapter_tasks)
    all_scores.extend(chapter_scores)

    rollup_scores: list[JudgeScore] = []
    for dim in ("humanness", "usefulness", "claim_defensibility"):
        for chapter in chapters:
            section_rollups = []
            for section in chapter.sections:
                para_scores_for_dim = [
                    s for s in para_scores
                    if s.dim_name == dim and any(p.id == s.unit_id for p in section.paragraphs)
                ]
                weights = []
                ordered = []
                for s in para_scores_for_dim:
                    p = next(p for p in section.paragraphs if p.id == s.unit_id)
                    ordered.append(s)
                    weights.append(p.char_count)
                if ordered:
                    sec_rollup = rollup_paragraph_to_section(ordered, weights, section.id)
                    rollup_scores.append(sec_rollup)
                    section_rollups.append((sec_rollup, section))
            if section_rollups:
                ch_rollup = rollup_section_to_chapter(
                    [r for r, _ in section_rollups],
                    [sum(p.char_count for p in s.paragraphs) for _, s in section_rollups],
                    f"chapter:{chapter.id}",
                )
                rollup_scores.append(ch_rollup)

    for chapter in chapters:
        sec_ed = [s for s in sec_scores if s.dim_name == "evidence_density" and
                  any(sec.id == s.unit_id for sec in chapter.sections)]
        weights = []
        for s in sec_ed:
            sec = next(sec for sec in chapter.sections if sec.id == s.unit_id)
            weights.append(sum(p.char_count for p in sec.paragraphs))
        if sec_ed:
            rollup_scores.append(rollup_section_to_chapter(sec_ed, weights, f"chapter:{chapter.id}"))

    derived_scores: list[JudgeScore] = []
    for chapter_score in chapter_scores:
        chapter_id = chapter_score.unit_id.split(":")[1]
        chapter = next(c for c in chapters if c.id == chapter_id)
        para_ids = [p.id for s in chapter.sections for p in s.paragraphs]
        derived_scores.extend(broadcast_chapter_to_paragraphs(chapter_score, para_ids))

    all_scores.extend(rollup_scores)
    all_scores.extend(derived_scores)

    rollups = _compute_run_rollups(all_scores, chapters)

    cache.flush()
    finished = datetime.now(UTC).isoformat()
    status = RunStatus.HALTED_BUDGET if cost >= cfg.max_cost_usd else RunStatus.COMPLETED

    return Run(
        id=run_id,
        corpus_snapshot_hash=snapshot_hash,
        book_mash_version=__version__,
        dim_registry_version=DIM_REGISTRY_VERSION,
        started_at=started,
        finished_at=finished,
        total_cost_usd=round(cost, 4),
        status=status,
        rollups=rollups,
        scores=all_scores,
    )


def _build_judges() -> dict[str, JudgeDim]:
    return {
        "humanness": HumannessJudge(),
        "voice": VoiceJudge(),
        "usefulness": UsefulnessJudge(),
        "evidence_density": EvidenceDensityJudge(),
        "claim_defensibility": ClaimDefensibilityJudge(),
        "redundancy": RedundancyJudge(),
    }


def _surrounding(paragraphs, i: int) -> list[str]:
    return [p.text for p in paragraphs[max(0, i - 1): i + 2] if p.id != paragraphs[i].id]


def _summarize_chapter(chapter: Chapter) -> str:
    return f"{chapter.title}\n\n" + "\n\n".join(
        s.paragraphs[0].text[:200] if s.paragraphs else s.heading for s in chapter.sections
    )


def _load_voice_baseline(chapters: list[Chapter], baseline_files: list[str]) -> list[str]:
    excerpts: list[str] = []
    for baseline_file in baseline_files:
        for c in chapters:
            if c.file_path.endswith(baseline_file):
                for s in c.sections:
                    for p in s.paragraphs[:1]:
                        excerpts.append(p.text)
                break
    return excerpts


def _make_run_id(snapshot_hash: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d-%H%M%S")
    short = snapshot_hash.split(":")[1][:4]
    return f"{timestamp}-{short}"


def _compute_run_rollups(scores: list[JudgeScore], chapters: list[Chapter]) -> dict[str, dict[str, float | int]]:
    rollups: dict[str, dict[str, float | int]] = {"book": {}}
    dim_names = ["humanness", "voice", "usefulness", "evidence_density", "claim_defensibility", "redundancy"]
    for dim in dim_names:
        chapter_means: list[float] = []
        for c in chapters:
            ch_scores = [s for s in scores if s.dim_name == dim and s.unit_id == f"chapter:{c.id}" and not s.derived]
            if ch_scores and ch_scores[0].score_0_100 is not None:
                chapter_means.append(ch_scores[0].score_0_100)
        book_mean = weighted_mean(chapter_means, [1] * len(chapter_means))
        if book_mean is not None:
            rollups["book"][dim] = round(book_mean, 1)
    for c in chapters:
        rollups[f"chapter:{c.id}"] = {}
        for dim in dim_names:
            ch = [s for s in scores if s.dim_name == dim and s.unit_id == f"chapter:{c.id}" and not s.derived]
            if ch and ch[0].score_0_100 is not None:
                rollups[f"chapter:{c.id}"][dim] = round(ch[0].score_0_100, 1)
        rollups[f"chapter:{c.id}"]["n_paragraphs"] = sum(len(s.paragraphs) for s in c.sections)
    return rollups
