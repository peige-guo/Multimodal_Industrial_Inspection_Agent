"""Vision adapter for defect detection.

The MVP ships a deterministic ``HeuristicDefectDetector`` that derives a
``DefectObservation`` from the filename and optional caller-supplied hints.
This keeps the full workflow testable without binding to any model. A real
VLM/YOLO/SAM implementation can later subclass ``DefectDetector`` and be wired
in via ``get_default_detector``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from backend.app.schemas.inspection import BoundingBox, DefectObservation

# Keyword -> canonical defect type. Order matters: first match wins.
_KEYWORD_DEFECTS: list[tuple[tuple[str, ...], str]] = [
    (("crack", "fracture", "fissure"), "crack"),
    (("corrosion", "rust", "pitting", "oxid"), "corrosion"),
    (("scratch", "scuff", "abrasion"), "scratch"),
    (("dent", "deform"), "dent"),
    (("good", "ok", "normal", "pass"), "none"),
]

_DESCRIPTIONS = {
    "crack": "Linear surface discontinuity consistent with a crack.",
    "corrosion": "Material degradation with rust/pitting indicative of corrosion.",
    "scratch": "Shallow linear surface mark without material separation.",
    "dent": "Localized surface deformation consistent with a dent.",
    "none": "No clear surface defect detected.",
    "unknown": "Surface anomaly detected; type could not be confidently classified.",
}


class DefectDetector(ABC):
    """Interface for any defect detection backend."""

    @abstractmethod
    def detect(
        self,
        *,
        filename: str,
        image_bytes: Optional[bytes] = None,
        hints: Optional[dict] = None,
    ) -> DefectObservation:
        """Return a single defect observation for the given image."""
        raise NotImplementedError


class HeuristicDefectDetector(DefectDetector):
    """Deterministic, dependency-free detector for the MVP and tests."""

    def detect(
        self,
        *,
        filename: str,
        image_bytes: Optional[bytes] = None,
        hints: Optional[dict] = None,
    ) -> DefectObservation:
        hints = hints or {}
        name = (filename or "").lower()

        defect_type = self._classify(name, hints)
        confidence = float(hints.get("confidence", self._default_confidence(name)))
        confidence = max(0.0, min(1.0, confidence))

        bbox = None
        if "bounding_box" in hints and hints["bounding_box"]:
            bbox = BoundingBox(**hints["bounding_box"])

        return DefectObservation(
            defect_type=defect_type,
            description=_DESCRIPTIONS.get(defect_type, _DESCRIPTIONS["unknown"]),
            confidence=confidence,
            location=hints.get("location"),
            bounding_box=bbox,
            length_mm=hints.get("length_mm"),
            area_ratio=hints.get("area_ratio"),
            load_bearing_area=bool(hints.get("load_bearing_area", False)),
        )

    @staticmethod
    def _classify(name: str, hints: dict) -> str:
        if hints.get("defect_type"):
            return str(hints["defect_type"]).lower()
        for keywords, defect in _KEYWORD_DEFECTS:
            if any(k in name for k in keywords):
                return defect
        return "unknown"

    @staticmethod
    def _default_confidence(name: str) -> float:
        # Filenames that name a defect are treated as higher confidence;
        # otherwise we return a low confidence so human review is triggered.
        for keywords, _defect in _KEYWORD_DEFECTS:
            if any(k in name for k in keywords):
                return 0.86
        return 0.55


def get_default_detector() -> DefectDetector:
    """Factory for the detector used by the workflow."""
    return HeuristicDefectDetector()
