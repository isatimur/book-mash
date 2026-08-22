from book_mash.corpus.models import Chapter, Paragraph, Section
from mash_core import JudgeLabel, JudgeScore
from book_mash.output.annotations import write_annotations


def test_writes_one_annotation_file_per_chapter(tmp_path):
    p = Paragraph(id="paragraph:ch1#L10-L15", text="x", line_range=(10, 15), char_count=1,
                  is_blockquote=False, is_code_fence=False)
    s = Section(id="sec1", heading="h", depth=2, paragraphs=[p], line_range=(10, 15))
    c = Chapter(id="ch1", title="One", file_path="/x/ch1.md", sections=[s], full_text="x", line_count=15)
    score = JudgeScore(
        dim_name="humanness", unit_id="paragraph:ch1#L10-L15", score_0_100=25.0,
        label=JudgeLabel.WEAK, reasoning="weak prose", evidence_refs=["phrase:'in today's evolving landscape'"],
        model="m", cost_usd=0.0, derived=False,
    )
    write_annotations([c], [score], tmp_path, run_id="r1")
    out = tmp_path / "chapter-ch1.annotations.md"
    assert out.exists()
    body = out.read_text()
    assert "paragraph:ch1#L10-L15" in body
    assert "humanness" in body
    assert "weak" in body
    assert "in today's evolving landscape" in body


def test_no_file_for_chapter_with_no_findings(tmp_path):
    p = Paragraph(id="paragraph:ch1#L1-L1", text="x", line_range=(1, 1), char_count=1,
                  is_blockquote=False, is_code_fence=False)
    s = Section(id="sec1", heading="h", depth=2, paragraphs=[p], line_range=(1, 1))
    c = Chapter(id="ch1", title="One", file_path="/x/ch1.md", sections=[s], full_text="x", line_count=1)
    strong_score = JudgeScore(
        dim_name="humanness", unit_id="paragraph:ch1#L1-L1", score_0_100=85.0,
        label=JudgeLabel.STRONG, reasoning="strong", evidence_refs=[], model="m", cost_usd=0.0, derived=False,
    )
    write_annotations([c], [strong_score], tmp_path, run_id="r1")
    assert not (tmp_path / "chapter-ch1.annotations.md").exists()
