from pathlib import Path

from book_mash.corpus.loader import load_chapters


FIXTURE_GLOB = str(Path(__file__).parent / "fixtures" / "mini_book" / "public" / "drafting" / "*.md")


def test_loads_two_chapters():
    chapters = load_chapters(FIXTURE_GLOB)
    assert len(chapters) == 2
    assert chapters[0].title == "Chapter 1 — The Premise"
    assert chapters[1].title == "Chapter 2 — A Drift"


def test_chapter_has_sections():
    chapters = load_chapters(FIXTURE_GLOB)
    ch1 = chapters[0]
    assert len(ch1.sections) == 2
    assert ch1.sections[0].heading == "A specific opening"
    assert ch1.sections[1].heading == "A second section"


def test_section_has_paragraphs():
    chapters = load_chapters(FIXTURE_GLOB)
    ch1 = chapters[0]
    sec1 = ch1.sections[0]
    assert len(sec1.paragraphs) == 2
    assert "first paragraph" in sec1.paragraphs[0].text
    assert sec1.paragraphs[0].line_range[0] > 0  # has real line numbers


def test_paragraph_char_count():
    chapters = load_chapters(FIXTURE_GLOB)
    p = chapters[0].sections[0].paragraphs[0]
    assert p.char_count == len(p.text)


def test_chapters_sorted_by_filename():
    chapters = load_chapters(FIXTURE_GLOB)
    assert chapters[0].file_path.endswith("chapter-01.md")
    assert chapters[1].file_path.endswith("chapter-02.md")


def test_chapter_has_number():
    chapters = load_chapters(FIXTURE_GLOB)
    assert chapters[0].number == 1
    assert chapters[1].number == 2
