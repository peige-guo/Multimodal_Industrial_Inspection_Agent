"""Canonical defect-type vocabulary shared across vision backends.

All detectors (heuristic, YOLO, VLM) normalize their raw output to the same
small set of canonical defect types so downstream RAG and decision logic stay
backend-agnostic.
"""

from __future__ import annotations

# Keyword -> canonical defect type. Order matters: first match wins.
KEYWORD_DEFECTS: list[tuple[tuple[str, ...], str]] = [
    (("crack", "fracture", "fissure"), "crack"),
    (("corrosion", "rust", "pitting", "oxid"), "corrosion"),
    (("scratch", "scuff", "abrasion"), "scratch"),
    (("dent", "deform"), "dent"),
    (("good", "ok", "normal", "pass", "no_defect"), "none"),
]

CANONICAL_DEFECTS = {"crack", "corrosion", "scratch", "dent", "none", "unknown"}

DESCRIPTIONS = {
    "crack": "Linear surface discontinuity consistent with a crack.",
    "corrosion": "Material degradation with rust/pitting indicative of corrosion.",
    "scratch": "Shallow linear surface mark without material separation.",
    "dent": "Localized surface deformation consistent with a dent.",
    "none": "No clear surface defect detected.",
    "unknown": "Surface anomaly detected; type could not be confidently classified.",
}


def canonical_defect_type(raw: str | None) -> str:
    """Map an arbitrary label/class name to a canonical defect type.

    Returns ``"unknown"`` when the input does not match any known keyword.
    """
    if not raw:
        return "unknown"
    text = str(raw).strip().lower()
    if text in CANONICAL_DEFECTS:
        return text
    for keywords, defect in KEYWORD_DEFECTS:
        if any(k in text for k in keywords):
            return defect
    return "unknown"


def describe(defect_type: str) -> str:
    """Return a human-readable description for a canonical defect type."""
    return DESCRIPTIONS.get(defect_type, DESCRIPTIONS["unknown"])
