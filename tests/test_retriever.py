from backend.app.rag.document_loader import load_document_text
from backend.app.rag.retriever import StandardRetriever, chunk_standard

SAMPLE = "data/sample_standards/pipeline_surface_defect_standard.md"


def test_chunk_preserves_clause_ids():
    text = load_document_text(SAMPLE)
    clauses = chunk_standard(text)
    ids = {c.clause_id for c in clauses}
    assert "3.2" in ids
    assert "2.1" in ids


def test_retrieve_crack_clause():
    retriever = StandardRetriever.from_text(load_document_text(SAMPLE))
    results = retriever.retrieve("crack length 5mm load bearing", top_k=3)
    assert results
    combined = " ".join(r.text.lower() for r in results)
    assert "crack" in combined


def test_no_match_returns_empty():
    retriever = StandardRetriever.from_text(load_document_text(SAMPLE))
    results = retriever.retrieve("xylophone unicorn spacecraft", top_k=3)
    assert results == []


def test_plain_text_paragraph_chunking():
    clauses = chunk_standard("First paragraph here.\n\nSecond paragraph there.")
    assert len(clauses) == 2
