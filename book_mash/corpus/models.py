from pydantic import BaseModel, Field


class Paragraph(BaseModel):
    id: str
    text: str
    line_range: tuple[int, int]
    char_count: int
    is_blockquote: bool
    is_code_fence: bool


class Section(BaseModel):
    id: str
    heading: str
    depth: int  # 2 = h2, 3 = h3
    paragraphs: list[Paragraph]
    line_range: tuple[int, int]


class Chapter(BaseModel):
    id: str
    number: int = 0
    title: str
    file_path: str
    sections: list[Section]
    full_text: str
    line_count: int


class ClaimEntry(BaseModel):
    id: str  # e.g. "claim:ch1#authority-blur"
    text: str
    strength: str  # "strong" | "moderate" | "weak"
    source_refs: list[str] = Field(default_factory=list)
    file_path: str


class Corpus(BaseModel):
    chapters: list[Chapter]
    claims_index: list[ClaimEntry] = Field(default_factory=list)
    voice_baseline: list[str] = Field(default_factory=list)  # excerpts from designated chapters
    corpus_snapshot_hash: str
