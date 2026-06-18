"""Shared helpers for vision-language-model defect detectors.

Both the OpenAI-compatible backend and the local Qwen backend use the same
structured-output prompt and the same JSON-to-``DefectObservation`` parsing, so
that logic lives here to avoid duplication.
"""

from __future__ import annotations

import io
import json
import re
from typing import Any, Optional

from backend.app.schemas.inspection import BoundingBox, DefectObservation
from backend.app.vision.labels import canonical_defect_type, describe

SYSTEM_PROMPT = (
    "You are an industrial visual inspection model. Analyze the image and "
    "report a single primary surface defect. Respond ONLY with a JSON object "
    "with keys: defect_type (one of: crack, corrosion, scratch, dent, none, "
    "unknown), description (string), confidence (0..1 float), location "
    "(short string or null), bounding_box (object with normalized x, y, "
    "width, height in 0..1, or null). Do not include any other text."
)

USER_PROMPT = "Inspect this industrial surface image."


def open_pil_image(image_bytes: bytes) -> Any:
    """Open RGB image bytes via Pillow (imported lazily)."""
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise RuntimeError(
            "Pillow is required for image-based detectors. Install it with: "
            "pip install -r backend/requirements-vision.txt"
        ) from exc
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def extract_json(content: str) -> dict:
    """Parse a JSON object from a model response, tolerating code fences."""
    text = content.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def parse_bbox(raw: Any) -> Optional[BoundingBox]:
    """Build a clipped, normalized BoundingBox from a model payload."""
    if not isinstance(raw, dict):
        return None
    try:
        x = max(0.0, min(1.0, float(raw["x"])))
        y = max(0.0, min(1.0, float(raw["y"])))
        w = float(raw["width"])
        h = float(raw["height"])
    except (KeyError, TypeError, ValueError):
        return None
    w = max(0.0, min(1.0 - x, w))
    h = max(0.0, min(1.0 - y, h))
    if w <= 0.0 or h <= 0.0:
        return None
    return BoundingBox(x=x, y=y, width=w, height=h)


def observation_from_payload(content: str, hints: dict) -> DefectObservation:
    """Convert a raw model text response into a ``DefectObservation``.

    Unparseable responses yield a low-confidence ``unknown`` so the workflow
    routes them to human review instead of failing hard.
    """
    try:
        data = extract_json(content)
    except (json.JSONDecodeError, ValueError):
        return DefectObservation(
            defect_type="unknown",
            description="VLM response could not be parsed.",
            confidence=0.0,
            location=hints.get("location"),
        )

    defect_type = canonical_defect_type(data.get("defect_type"))
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return DefectObservation(
        defect_type=defect_type,
        description=str(data.get("description") or describe(defect_type)),
        confidence=max(0.0, min(1.0, confidence)),
        location=hints.get("location") or data.get("location"),
        bounding_box=parse_bbox(data.get("bounding_box")),
        length_mm=hints.get("length_mm"),
        area_ratio=hints.get("area_ratio"),
        load_bearing_area=bool(hints.get("load_bearing_area", False)),
    )
