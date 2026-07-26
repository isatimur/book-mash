from book_mash.corpus.models import Paragraph, Section, Chapter, Corpus
from book_mash.runners.models import Run, RunStatus


def test_paragraph_construction():
    p = Paragraph(
        id="paragraph:ch1#L10-L15",
        text="Hello world.",
        line_range=(10, 15),
        char_count=12,
        is_blockquote=False,
        is_code_fence=False,
    )
    assert p.id == "paragraph:ch1#L10-L15"
    assert p.line_range == (10, 15)


def test_chapter_aggregates_text():
    p1 = Paragraph(id="p1", text="One.", line_range=(1, 1), char_count=4,
                   is_blockquote=False, is_code_fence=False)
    p2 = Paragraph(id="p2", text="Two.", line_range=(2, 2), char_count=4,
                   is_blockquote=False, is_code_fence=False)
    s = Section(id="s1", heading="Intro", depth=2, paragraphs=[p1, p2], line_range=(1, 2))
    c = Chapter(id="ch1", title="One", file_path="/x/ch1.md", sections=[s],
                full_text="One.\nTwo.", line_count=2)
    assert len(c.sections) == 1
    assert c.sections[0].paragraphs[0].text == "One."


def test_run_status_enum():
    r = Run(
        id="2026-05-19-1547-a7f3",
        corpus_snapshot_hash="sha256:abc",
        book_mash_version="0.1.0",
        dim_registry_version="0.1.0",
        started_at="2026-05-19T15:47:12Z",
        finished_at=None,
        total_cost_usd=0.0,
        status=RunStatus.RUNNING,
        rollups={},
    )
    assert r.status == RunStatus.RUNNING
