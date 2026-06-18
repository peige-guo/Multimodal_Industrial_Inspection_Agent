"""Vision adapter for defect detection.

The MVP ships a deterministic ``HeuristicDefectDetector`` that derives a
``DefectObservation`` from the filename and optional caller-supplied hints.
This keeps the full workflow testable without binding to any model. Phase-2
model backends (``YOLODefectDetector``, ``VLMDefectDetector``) subclass
``DefectDetector`` and are selected at runtime by ``get_default_detector``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from backend.app.schemas.inspection import BoundingBox, DefectObservation
from backend.app.vision.labels import KEYWORD_DEFECTS, canonical_defect_type, describe


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
            description=describe(defect_type),
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
            return canonical_defect_type(str(hints["defect_type"]))
        return canonical_defect_type(name)

    @staticmethod
    def _default_confidence(name: str) -> float:
        # Filenames that name a defect are treated as higher confidence;
        # otherwise we return a low confidence so human review is triggered.
        for keywords, _defect in KEYWORD_DEFECTS:
            if any(k in name for k in keywords):
                return 0.86
        return 0.55


def get_default_detector() -> DefectDetector:
    """Return the detector selected by configuration (env-driven).

    Delegates to :func:`backend.app.vision.factory.build_detector`, which falls
    back to the heuristic detector when a requested backend or its dependencies
    are unavailable. Imported lazily to avoid a circular import.
    """
    from backend.app.vision.factory import build_detector

    return build_detector()
