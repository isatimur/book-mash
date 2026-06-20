import glob
import re
from pathlib import Path

from book_mash.corpus.models import ClaimEntry


_CLAIM_HEADER = re.compile(r"^##\s+(\d+)\)\s+(.+?)\s*$", re.MULTILINE)
_SUPPORT = re.compile(r"\*\*Support level:\*\*\s*([A-Za-z]+)")
_CANDIDATE = re.compile(r"\*\*Candidate chapters:\*\*\s*(.+)")
_WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")
_PHRASING = re.compile(r"\*\*Reusable phrasing:\*\*\s*(.+)")
_QUOTE = re.compile(r"\*\*Quote:\*\*\s*[\"“]?(.+?)[\"”]?\s*$", re.MULTILINE)
# the speaker/org label sits inside the wikilink display text after the final "|", e.g.
# [[624-...-anthropic|#624 — Karan Sampath, Anthropic]]
_SOURCE_LABEL = re.compile(r"\[\[[^\]|]*\|([^\]]+)\]\]")


def load_claims_index(claims_dir: str) -> list[ClaimEntry]:
    if not Path(claims_dir).exists():
        return []
    entries: list[ClaimEntry] = []
    for path in sorted(glob.glob(f"{claims_dir}/*.md")):
        text = Path(path).read_text(encoding="utf-8")
        for number, claim_text, body in _split_by_claim_header(text):
            support = _SUPPORT.search(body)
            candidate = _CANDIDATE.search(body)
            phrasing = _PHRASING.search(body)
            entries.append(ClaimEntry(
                id=f"claims#{number}",
                text=claim_text.strip(),
                support_level=support.group(1).strip().lower() if support else "moderate",
                candidate_chapters=_parse_ints(candidate.group(1)) if candidate else [],
                source_refs=_WIKILINK.findall(body),
                quotes=[q.strip() for q in _QUOTE.findall(body) if q.strip()],
                source_descriptions=[s.strip() for s in _SOURCE_LABEL.findall(body) if s.strip()],
                reusable_phrasing=phrasing.group(1).strip() if phrasing else "",
                file_path=path,
            ))
    return entries


def _split_by_claim_header(text: str) -> list[tuple[str, str, str]]:
    parts: list[tuple[str, str, str]] = []
    matches = list(_CLAIM_HEADER.finditer(text))
    for i, m in enumerate(matches):
        number = m.group(1)
        claim_text = m.group(2)
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts.append((number, claim_text, text[body_start:body_end]))
    return parts


def _parse_ints(s: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", s)]
