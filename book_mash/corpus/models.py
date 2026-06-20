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
    id: str  # "claims#<N>"
    text: str  # the claim prose (the ## N) heading)
    support_level: str  # "tentative" | "moderate" | "strong"
    candidate_chapters: list[int] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    quotes: list[str] = Field(default_factory=list)  # verbatim supporting quotes from sources
    source_descriptions: list[str] = Field(default_factory=list)  # speaker/org labels for each source
    reusable_phrasing: str = ""  # the claim's reusable phrasing line, if present
    file_path: str

    def retrieval_text(self) -> str:
        """Concatenated text used for lexical relevance retrieval against a paragraph."""
        parts = [self.text, self.reusable_phrasing, *self.quotes, *self.source_descriptions]
        return " ".join(p for p in parts if p)


class Corpus(BaseModel):
    chapters: list[Chapter]
    claims_index: list[ClaimEntry] = Field(default_factory=list)
    voice_baseline: list[str] = Field(default_factory=list)  # excerpts from designated chapters
    corpus_snapshot_hash: str
