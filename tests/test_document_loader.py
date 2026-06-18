import pytest

from backend.app.rag.document_loader import (
    UnsupportedDocumentError,
    load_document_bytes,
    load_document_text,
)


def test_load_txt_bytes():
    text = load_document_bytes(b"hello world", "note.txt")
    assert text == "hello world"


def test_load_md_file(tmp_path):
    p = tmp_path / "standard.md"
    p.write_text("# Title\n\nClause body", encoding="utf-8")
    text = load_document_text(p)
    assert "Clause body" in text


def test_unsupported_extension_raises():
    with pytest.raises(UnsupportedDocumentError):
        load_document_bytes(b"data", "report.docx")


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_document_text(tmp_path / "nope.md")


def test_real_sample_standard_loads():
    text = load_document_text("data/sample_standards/pipeline_surface_defect_standard.md")
    assert "Crack" in text
