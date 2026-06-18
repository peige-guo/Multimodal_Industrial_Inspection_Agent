"""Load inspection standard documents into plain text.

Supports `.txt`, `.md`, and `.pdf`. PDF parsing uses PyMuPDF (fitz) and is
imported lazily so the rest of the system works even if PyMuPDF is absent.
"""

from __future__ import annotations

from pathlib import Path

TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS


class UnsupportedDocumentError(ValueError):
    """Raised when a document extension is not supported."""


def _extract_pdf_text(data: bytes) -> str:
    try:
        import fitz  # type: ignore  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise RuntimeError(
            "PyMuPDF (pymupdf) is required to parse PDF documents."
        ) from exc

    text_parts: list[str] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def load_document_bytes(data: bytes, filename: str) -> str:
    """Extract text from raw document bytes, dispatching on file extension."""
    suffix = Path(filename).suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return data.decode("utf-8", errors="replace")
    if suffix in PDF_EXTENSIONS:
        return _extract_pdf_text(data)
    raise UnsupportedDocumentError(
        f"Unsupported document extension '{suffix}'. "
        f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )


def load_document_text(path: str | Path) -> str:
    """Read and extract text from a document on disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    return load_document_bytes(path.read_bytes(), path.name)
