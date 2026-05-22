import asyncio
from datetime import UTC, datetime
from pathlib import Path

from book_mash.cache import JudgeScoreCache, content_hash
from book_mash.config import BookMashConfig
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


_CONCURRENCY = 8


async def run_measurement(cfg: BookMashConfig) -> Run:
    chapters = load_chapters(cfg.chapters_glob)
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

    async def run_judge(judge: JudgeDim, input: JudgeInput) -> JudgeScore:
        nonlocal cost
        unit_hash = content_hash(input.unit_text)
        cached = cache.get(unit_hash, judge.name, DIM_REGISTRY_VERSION, judge.model_id)
        if cached is not None:
            return cached
        async with sem:
            if cost >= cfg.max_cost_usd:
                return JudgeScore(
                    dim_name=judge.name, unit_id=input.unit_id, score_0_100=None,
                    label=JudgeLabel.ERROR, reasoning="halted: budget", evidence_refs=[],
                    model=judge.model_id, cost_usd=0.0, derived=False,
                )
            score = await judge.judge(input)
            cost += score.cost_usd
            cache.put(unit_hash, judge.name, DIM_REGISTRY_VERSION, judge.model_id, score)
            return score

    para_tasks: list = []
    for chapter in chapters:
        relevant_ledger = [c for c in claims_index if c.id.startswith(f"claim:{chapter.id}")]
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
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d-%H%M")
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
