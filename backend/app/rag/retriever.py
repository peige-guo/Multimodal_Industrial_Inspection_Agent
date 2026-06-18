"""Chunk standard documents into clauses and retrieve relevant ones.

The MVP uses transparent lexical matching (token overlap with light term
weighting) instead of embeddings, so retrieval is deterministic, dependency
free, and easy to test. The interface is intentionally simple so it can later
be swapped for a FAISS/Chroma embedding retriever.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.app.schemas.inspection import StandardEvidence

# Matches markdown headings like "### 3.2 Serious Crack" and extracts the
# clause id ("3.2") and title ("Serious Crack").
_HEADING_RE = re.compile(r"^#{1,6}\s+(?P<id>\d+(?:\.\d+)*)?\s*(?P<title>.*)$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Very small stopword list; lexical retrieval benefits from dropping these.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "of", "to", "in", "on", "and", "or", "for",
    "with", "as", "at", "by", "be", "this", "that", "it", "from", "shall",
    "must", "when", "if", "not", "no", "than", "more", "less",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


@dataclass
class StandardClause:
    """A single retrievable unit of a standard document."""

    text: str
    clause_id: str | None = None
    title: str | None = None
    tokens: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.tokens:
            combined = f"{self.title or ''} {self.text}"
            self.tokens = set(_tokenize(combined))


def chunk_standard(text: str) -> list[StandardClause]:
    """Split a standard document into clauses, preserving heading ids/titles.

    A new clause starts at each markdown heading. Body lines accumulate into
    the current clause. Documents without headings yield paragraph chunks.
    """
    lines = text.splitlines()
    clauses: list[StandardClause] = []
    cur_id: str | None = None
    cur_title: str | None = None
    cur_body: list[str] = []

    def flush() -> None:
        body = "\n".join(cur_body).strip()
        if body or cur_title:
            clauses.append(
                StandardClause(
                    text=body or (cur_title or ""),
                    clause_id=cur_id,
                    title=cur_title,
                )
            )

    saw_heading = False
    for line in lines:
        heading = _HEADING_RE.match(line.strip()) if line.strip().startswith("#") else None
        if heading:
            saw_heading = True
            flush()
            cur_id = heading.group("id") or None
            cur_title = (heading.group("title") or "").strip() or None
            cur_body = []
        else:
            cur_body.append(line)
    flush()

    if not saw_heading:
        # Fall back to paragraph splitting when there are no headings.
        clauses = [
            StandardClause(text=para.strip())
            for para in re.split(r"\n\s*\n", text)
            if para.strip()
        ]
    return clauses


class StandardRetriever:
    """Lexical retriever over standard clauses."""

    def __init__(self, clauses: list[StandardClause]):
        self.clauses = clauses
        # Document frequency for light idf-style weighting.
        self._doc_freq: dict[str, int] = {}
        for clause in clauses:
            for token in clause.tokens:
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1

    @classmethod
    def from_text(cls, text: str) -> "StandardRetriever":
        return cls(chunk_standard(text))

    def _score(self, query_tokens: list[str], clause: StandardClause) -> float:
        n = max(len(self.clauses), 1)
        score = 0.0
        for token in set(query_tokens):
            if token in clause.tokens:
                df = self._doc_freq.get(token, 1)
                # Rarer terms contribute more; common terms contribute less.
                score += 1.0 + (n / df) ** 0.5 * 0.1
        return score

    def retrieve(self, query: str, top_k: int = 3) -> list[StandardEvidence]:
        """Return up to ``top_k`` clauses scored against the query.

        Clauses with zero overlap are excluded, so a no-match query yields an
        empty list (which downstream logic uses to trigger human review).
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        scored = [
            (self._score(query_tokens, clause), clause) for clause in self.clauses
        ]
        scored = [(s, c) for s, c in scored if s > 0]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            StandardEvidence(
                clause_id=clause.clause_id,
                title=clause.title,
                text=clause.text,
                score=round(score, 4),
            )
            for score, clause in scored[:top_k]
        ]
