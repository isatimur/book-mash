from pathlib import Path

from mash_core import JudgeLabel, JudgeScore
from book_mash.runners.models import Run


_DIM_ORDER = ["humanness", "voice", "usefulness", "evidence_density", "claim_defensibility", "redundancy"]
_LABEL_SYMBOL = {
    JudgeLabel.STRONG: "[strong]",
    JudgeLabel.MODERATE: "[moderate]",
    JudgeLabel.WEAK: "[weak]",
    JudgeLabel.FAIL: "[fail]",
    JudgeLabel.ERROR: "[error]",
}


def render_report(run: Run, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    parts = [
        f"# book-mash run {run.id}",
        "",
        _render_tldr(run),
        "",
        "## Heatmap",
        _render_heatmap(run),
        "",
        "## Per-chapter detail",
        _render_per_chapter(run),
        "",
        "## Ship-blockers",
        _render_ship_blockers(run.scores),
        "",
        "## Run metadata",
        _render_metadata(run),
    ]
    (run_dir / "report.md").write_text("\n".join(parts))


def _render_tldr(run: Run) -> str:
    book = run.rollups.get("book", {})
    if not book:
        return "_No book-level rollup available._"
    sorted_dims = sorted(book.items(), key=lambda kv: kv[1])
    weakest = sorted_dims[0] if sorted_dims else (None, None)
    strongest = sorted_dims[-1] if sorted_dims else (None, None)
    return (
        f"**TL;DR:** Across the book, the weakest dim is **{weakest[0]}** ({weakest[1]:.0f}), "
        f"strongest is **{strongest[0]}** ({strongest[1]:.0f}). "
        f"Run cost: ${run.total_cost_usd:.2f}. Status: {run.status.value}."
    )


def _render_heatmap(run: Run) -> str:
    chapter_keys = sorted([k for k in run.rollups if k.startswith("chapter:")])
    if not chapter_keys:
        return "_No chapter rollups._"
    header = "| Chapter | " + " | ".join(_DIM_ORDER) + " |"
    sep = "|---|" + "---|" * len(_DIM_ORDER)
    rows = [header, sep]
    for ck in chapter_keys:
        ch_rollup = run.rollups[ck]
        row = [ck.split(":")[1]]
        for d in _DIM_ORDER:
            val = ch_rollup.get(d)
            row.append(f"{val:.0f}" if isinstance(val, (int, float)) else "-")
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows)


def _render_per_chapter(run: Run) -> str:
    chapter_keys = sorted([k for k in run.rollups if k.startswith("chapter:")])
    parts = []
    for ck in chapter_keys:
        ch_id = ck.split(":")[1]
        parts.append(f"### {ch_id}")
        weakest = sorted(
            [s for s in run.scores if s.unit_id.startswith(f"paragraph:{ch_id}") and not s.derived and s.score_0_100 is not None],
            key=lambda s: s.score_0_100 or 0.0,  # None already filtered above
        )[:3]
        if weakest:
            parts.append("**Weakest paragraphs:**")
            for s in weakest:
                parts.append(f"- {_LABEL_SYMBOL[s.label]} `{s.unit_id}` - {s.dim_name} - {s.score_0_100:.0f}")
                parts.append(f"  > {s.reasoning}")
        parts.append("")
    return "\n".join(parts)


def _render_ship_blockers(scores: list[JudgeScore]) -> str:
    blockers = [s for s in scores if s.label == JudgeLabel.FAIL and s.dim_name == "claim_defensibility"]
    others = [s for s in scores if s.label == JudgeLabel.FAIL and s.dim_name != "claim_defensibility"]
    if not blockers and not others:
        return "_No ship-blockers found._"
    parts = []
    if blockers:
        parts.append("### Claim defensibility failures (must resolve)")
        for s in blockers:
            parts.append(f"- `{s.unit_id}` - {s.reasoning}")
    if others:
        parts.append("### Other failures")
        for s in others:
            parts.append(f"- `{s.unit_id}` - {s.dim_name} - {s.reasoning}")
    return "\n".join(parts)


def _render_metadata(run: Run) -> str:
    return (
        f"- snapshot: `{run.corpus_snapshot_hash}`\n"
        f"- book_mash: `{run.book_mash_version}`\n"
        f"- dim_registry: `{run.dim_registry_version}`\n"
        f"- started: `{run.started_at}` - finished: `{run.finished_at}`\n"
        f"- cost: `${run.total_cost_usd:.2f}` - status: `{run.status.value}`"
    )
