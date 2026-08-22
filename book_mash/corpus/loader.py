import glob
import hashlib
import re
from pathlib import Path

from book_mash.corpus.models import Chapter, Paragraph, Section


_H1 = re.compile(r"^#\s+(.+)$")
_H2 = re.compile(r"^##\s+(.+)$")
_H3 = re.compile(r"^###\s+(.+)$")
_CODE_FENCE = re.compile(r"^```")
_BLOCKQUOTE = re.compile(r"^>\s")
_CHAPTER_NUM = re.compile(r"[Cc]hapter\s+(\d+)")


def load_chapters(chapters_glob: str, skip_sections: list[str] | None = None) -> list[Chapter]:
    paths = sorted(glob.glob(chapters_glob))
    skip = set(skip_sections or [])
    return [_load_chapter(p, skip) for p in paths]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _load_chapter(file_path: str, skip_sections: set[str] | None = None) -> Chapter:
    text = Path(file_path).read_text(encoding="utf-8")
    lines = text.splitlines()

    title = ""
    sections: list[Section] = []
    current_section: dict | None = None  # accumulator
    current_para_lines: list[str] = []
    current_para_start: int | None = None
    in_code = False
    in_skipped_section = False
    chapter_id = _slug(Path(file_path).stem)
    _skip = skip_sections or set()

    def flush_paragraph():
        nonlocal current_para_lines, current_para_start
        if not current_para_lines or current_section is None:
            current_para_lines = []
            current_para_start = None
            return
        para_text = "\n".join(current_para_lines).strip()
        if not para_text:
            current_para_lines = []
            current_para_start = None
            return
        start = current_para_start or 1
        end = start + len(current_para_lines) - 1
        is_bq = all(_BLOCKQUOTE.match(ln) for ln in current_para_lines if ln.strip())
        is_cf = any(_CODE_FENCE.match(ln) for ln in current_para_lines)
        para_id = f"paragraph:{chapter_id}#L{start}-L{end}"
        current_section["paragraphs"].append(Paragraph(
            id=para_id,
            text=para_text,
            line_range=(start, end),
            char_count=len(para_text),
            is_blockquote=is_bq,
            is_code_fence=is_cf,
        ))
        current_para_lines = []
        current_para_start = None

    def flush_section():
        nonlocal current_section
        if current_section is None:
            return
        last_end = current_section["paragraphs"][-1].line_range[1] if current_section["paragraphs"] else current_section["start"]
        sec = Section(
            id=current_section["id"],
            heading=current_section["heading"],
            depth=current_section["depth"],
            paragraphs=current_section["paragraphs"],
            line_range=(current_section["start"], last_end),
        )
        sections.append(sec)
        current_section = None

    for i, line in enumerate(lines, start=1):
        if _CODE_FENCE.match(line):
            in_code = not in_code
            current_para_lines.append(line)
            if current_para_start is None:
                current_para_start = i
            continue

        if in_code:
            current_para_lines.append(line)
            continue

        m1 = _H1.match(line)
        if m1:
            title = m1.group(1).strip()
            continue

        m2 = _H2.match(line) or _H3.match(line)
        if m2:
            flush_paragraph()
            flush_section()
            heading = m2.group(1).strip()
            if heading in _skip:
                in_skipped_section = True
                current_section = None
                continue
            in_skipped_section = False
            depth = 2 if _H2.match(line) else 3
            sec_id = f"section:{chapter_id}#{_slug(heading)}"
            current_section = {
                "id": sec_id,
                "heading": heading,
                "depth": depth,
                "paragraphs": [],
                "start": i,
            }
            continue

        if in_skipped_section:
            continue

        if line.strip() == "":
            flush_paragraph()
            continue

        if current_section is None:
            # paragraph before any h2 — make a default "preamble" section
            current_section = {
                "id": f"section:{chapter_id}#preamble",
                "heading": "(preamble)",
                "depth": 2,
                "paragraphs": [],
                "start": i,
            }
        if current_para_start is None:
            current_para_start = i
        current_para_lines.append(line)

    flush_paragraph()
    flush_section()

    num_match = _CHAPTER_NUM.search(title)
    number = int(num_match.group(1)) if num_match else 0

    return Chapter(
        id=chapter_id,
        number=number,
        title=title or chapter_id,
        file_path=file_path,
        sections=sections,
        full_text=text,
        line_count=len(lines),
    )


def compute_snapshot_hash(chapters: list[Chapter]) -> str:
    h = hashlib.sha256()
    for c in chapters:
        h.update(c.file_path.encode("utf-8"))
        h.update(b"\0")
        h.update(c.full_text.encode("utf-8"))
        h.update(b"\0")
    return f"sha256:{h.hexdigest()}"
