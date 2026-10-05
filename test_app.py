from app import get_text_chunks


def test_get_text_chunks_splits_long_text_and_tags_source():
    docs = [("report.pdf", "Lorem ipsum dolor sit amet. " * 100)]

    chunks, metadatas = get_text_chunks(docs)

    assert len(chunks) > 1
    assert all(meta == {"source": "report.pdf"} for meta in metadatas)
    assert len(chunks) == len(metadatas)


def test_get_text_chunks_skips_documents_with_no_text():
    docs = [("scanned.pdf", ""), ("notes.pdf", "Some real extracted text.")]

    chunks, metadatas = get_text_chunks(docs)

    assert all(meta["source"] != "scanned.pdf" for meta in metadatas)
    assert any(meta["source"] == "notes.pdf" for meta in metadatas)


def test_get_text_chunks_returns_empty_lists_when_nothing_extractable():
    docs = [("scanned.pdf", ""), ("blank.pdf", "   ")]

    chunks, metadatas = get_text_chunks(docs)

    assert chunks == []
    assert metadatas == []
