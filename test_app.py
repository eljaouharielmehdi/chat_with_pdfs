from langchain_core.documents import Document

from app import get_text_chunks, format_source, select_context_docs


def test_get_text_chunks_splits_long_page_and_tags_source_and_page():
    pages = [("report.pdf", 1, "Lorem ipsum dolor sit amet. " * 100)]

    chunks, metadatas = get_text_chunks(pages)

    assert len(chunks) > 1
    assert all(meta == {"source": "report.pdf", "page": 1} for meta in metadatas)
    assert len(chunks) == len(metadatas)


def test_get_text_chunks_keeps_separate_pages_distinct():
    pages = [
        ("notes.pdf", 1, "First page content."),
        ("notes.pdf", 2, "Second page content."),
    ]

    chunks, metadatas = get_text_chunks(pages)

    pages_seen = {meta["page"] for meta in metadatas}
    assert pages_seen == {1, 2}


def test_get_text_chunks_returns_empty_lists_when_no_pages():
    chunks, metadatas = get_text_chunks([])

    assert chunks == []
    assert metadatas == []


def test_format_source_includes_page_number():
    assert format_source({"source": "report.pdf", "page": 3}) == "report.pdf (p. 3)"


def test_format_source_without_page_falls_back_to_filename():
    assert format_source({"source": "report.pdf"}) == "report.pdf"


def test_select_context_docs_respects_token_budget():
    big_doc = Document(page_content="word " * 5000, metadata={"source": "a.pdf", "page": 1})
    small_doc = Document(page_content="short", metadata={"source": "b.pdf", "page": 1})

    selected = select_context_docs([big_doc, small_doc], max_tokens=10)

    assert selected == [big_doc]


def test_select_context_docs_always_keeps_at_least_one_doc():
    huge_doc = Document(page_content="word " * 50000, metadata={"source": "a.pdf", "page": 1})

    selected = select_context_docs([huge_doc], max_tokens=1)

    assert selected == [huge_doc]
