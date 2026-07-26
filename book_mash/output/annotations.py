from pathlib import Path

from book_mash.corpus.models import Chapter
from mash_core import JudgeLabel, JudgeScore


_FINDING_LABELS = {JudgeLabel.WEAK, JudgeLabel.FAIL}


def write_annotations(
    chapters: list[Chapter],
    scores: list[JudgeScore],
    run_dir: Path,
    run_id: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for chapter in chapters:
        findings = [
            s for s in scores
            if s.unit_id.startswith(f"paragraph:{chapter.id}")
            and not s.derived
            and s.label in _FINDING_LABELS
        ]
        if not findings:
            continue
        findings.sort(key=lambda s: (s.unit_id, s.dim_name))
        parts = [f"# {chapter.title} - Annotations from run {run_id}", ""]
        last_unit = None
        para_text_lookup = {
            p.id: p.text for s in chapter.sections for p in s.paragraphs
        }
        for s in findings:
            if s.unit_id != last_unit:
                last_unit = s.unit_id
                parts.append("")
            score_str = f"{s.score_0_100:.0f}" if s.score_0_100 is not None else "-"
            parts.append(f"## {s.unit_id} - {s.dim_name} - {s.label.value} ({score_str})")
            quoted = para_text_lookup.get(s.unit_id, "")
            for line in quoted.splitlines():
                parts.append(f"> {line}")
            parts.append(f"**Judge:** {s.reasoning}")
            if s.evidence_refs:
                parts.append(f"**Refs:** {', '.join(s.evidence_refs)}")
            parts.append("")
        (run_dir / f"chapter-{chapter.id}.annotations.md").write_text("\n".join(parts))
