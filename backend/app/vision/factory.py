"""Build the configured defect detector, with graceful fallback.

Selection order:
- ``INSPECTION_DETECTOR=heuristic`` (default): dependency-free stub.
- ``INSPECTION_DETECTOR=yolo``: Ultralytics YOLO model.
- ``INSPECTION_DETECTOR=vlm``: OpenAI-compatible vision model.
- ``INSPECTION_DETECTOR=auto``: prefer YOLO, then VLM, then heuristic, based on
  which is configured.

If a requested backend is misconfigured or its dependencies are missing, the
factory logs a warning and falls back to the heuristic detector so the service
stays available.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Optional

from backend.app.vision.config import VisionConfig
from backend.app.vision.defect_detector import DefectDetector, HeuristicDefectDetector

logger = logging.getLogger(__name__)


def _installed(*modules: str) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in modules)


def _build_yolo(config: VisionConfig) -> DefectDetector:
    if not config.yolo_model_path:
        raise RuntimeError("INSPECTION_YOLO_MODEL is not set.")
    if not _installed("ultralytics", "PIL"):
        raise RuntimeError("ultralytics/Pillow are not installed.")
    from backend.app.vision.yolo_detector import YOLODefectDetector

    return YOLODefectDetector(config=config)


def _build_vlm(config: VisionConfig) -> DefectDetector:
    if not config.vlm_api_key:
        raise RuntimeError("INSPECTION_VLM_API_KEY (or OPENAI_API_KEY) is not set.")
    if not _installed("openai"):
        raise RuntimeError("openai is not installed.")
    from backend.app.vision.vlm_detector import VLMDefectDetector

    return VLMDefectDetector(config=config)


def _auto(config: VisionConfig) -> DefectDetector:
    for builder, configured in ((_build_yolo, config.yolo_model_path),
                                (_build_vlm, config.vlm_api_key)):
        if not configured:
            continue
        try:
            return builder(config)
        except Exception as exc:  # noqa: BLE001 - try the next backend
            logger.warning("auto: %s unavailable (%s)", builder.__name__, exc)
    return HeuristicDefectDetector()


def build_detector(config: Optional[VisionConfig] = None) -> DefectDetector:
    """Construct the detector named by ``config`` (defaults to env config)."""
    config = config or VisionConfig.from_env()
    backend = (config.backend or "heuristic").lower()

    try:
        if backend == "heuristic":
            return HeuristicDefectDetector()
        if backend == "yolo":
            return _build_yolo(config)
        if backend == "vlm":
            return _build_vlm(config)
        if backend == "auto":
            return _auto(config)
        logger.warning(
            "Unknown INSPECTION_DETECTOR=%r; using heuristic detector.", backend
        )
    except Exception as exc:  # noqa: BLE001 - fallback must be robust
        logger.warning(
            "Failed to build '%s' detector (%s); falling back to heuristic.",
            backend,
            exc,
        )
    return HeuristicDefectDetector()
