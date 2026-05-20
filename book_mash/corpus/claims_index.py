import glob
import re
from pathlib import Path

from book_mash.corpus.models import ClaimEntry


_CLAIM_HEADER = re.compile(r"^##\s+(claim:[a-z0-9#\-]+)$", re.MULTILINE)
_STRENGTH = re.compile(r"\*\*Strength:\*\*\s+(\w+)")
_TEXT = re.compile(r"\*\*Text:\*\*\s+(.+)")
_SOURCES = re.compile(r"\*\*Sources:\*\*\s+(.+)")


def load_claims_index(claims_dir: str) -> list[ClaimEntry]:
    if not Path(claims_dir).exists():
        return []
    entries: list[ClaimEntry] = []
    for path in sorted(glob.glob(f"{claims_dir}/*.md")):
        text = Path(path).read_text(encoding="utf-8")
        blocks = _split_by_claim_header(text)
        for claim_id, body in blocks:
            strength = _STRENGTH.search(body)
            text_match = _TEXT.search(body)
            sources_match = _SOURCES.search(body)
            sources = [s.strip() for s in sources_match.group(1).split(",")] if sources_match else []
            entries.append(ClaimEntry(
                id=claim_id,
                text=text_match.group(1).strip() if text_match else "",
                strength=strength.group(1).strip() if strength else "moderate",
                source_refs=sources,
                file_path=path,
            ))
    return entries


def _split_by_claim_header(text: str) -> list[tuple[str, str]]:
    parts = []
    matches = list(_CLAIM_HEADER.finditer(text))
    for i, m in enumerate(matches):
        claim_id = m.group(1)
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts.append((claim_id, text[body_start:body_end]))
    return parts
