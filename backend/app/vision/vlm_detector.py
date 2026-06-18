"""Vision-Language-Model defect detector (Phase 2).

Sends the image plus a structured-output prompt to an OpenAI-compatible Chat
Completions endpoint and parses the JSON response into a ``DefectObservation``.
Works with any compatible provider via ``INSPECTION_VLM_BASE_URL`` (OpenAI,
local servers, Qwen-VL gateways, etc.).

The ``openai`` SDK is imported lazily and the client can be injected for tests.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Optional

from backend.app.schemas.inspection import BoundingBox, DefectObservation
from backend.app.vision.config import VisionConfig
from backend.app.vision.defect_detector import DefectDetector
from backend.app.vision.labels import canonical_defect_type, describe

_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

_SYSTEM_PROMPT = (
    "You are an industrial visual inspection model. Analyze the image and "
    "report a single primary surface defect. Respond ONLY with a JSON object "
    "with keys: defect_type (one of: crack, corrosion, scratch, dent, none, "
    "unknown), description (string), confidence (0..1 float), location "
    "(short string or null), bounding_box (object with normalized x, y, "
    "width, height in 0..1, or null). Do not include any other text."
)


def _mime_for(filename: str) -> str:
    return _MIME_BY_SUFFIX.get(Path(filename).suffix.lower(), "image/png")


def _extract_json(content: str) -> dict:
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


def _parse_bbox(raw: Any) -> Optional[BoundingBox]:
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


class VLMDefectDetector(DefectDetector):
    """Defect detector backed by an OpenAI-compatible vision model."""

    def __init__(
        self,
        config: Optional[VisionConfig] = None,
        client: Any = None,
    ) -> None:
        self.config = config or VisionConfig()
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:  # pragma: no cover - without dep only
                raise RuntimeError(
                    "openai is required for the VLM detector. Install it with: "
                    "pip install -r backend/requirements-vision.txt"
                ) from exc
            if not self.config.vlm_api_key:
                raise RuntimeError(
                    "No VLM API key configured. Set INSPECTION_VLM_API_KEY "
                    "(or OPENAI_API_KEY)."
                )
            kwargs: dict[str, Any] = {"api_key": self.config.vlm_api_key}
            if self.config.vlm_base_url:
                kwargs["base_url"] = self.config.vlm_base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def detect(
        self,
        *,
        filename: str,
        image_bytes: Optional[bytes] = None,
        hints: Optional[dict] = None,
    ) -> DefectObservation:
        hints = hints or {}
        if not image_bytes:
            return DefectObservation(
                defect_type="unknown",
                description="No image data provided to the VLM detector.",
                confidence=0.0,
                location=hints.get("location"),
            )

        client = self._get_client()
        data_uri = (
            f"data:{_mime_for(filename)};base64,"
            + base64.b64encode(image_bytes).decode("ascii")
        )
        response = client.chat.completions.create(
            model=self.config.vlm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Inspect this industrial surface image.",
                        },
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        return self._parse_observation(content, hints)

    def _parse_observation(self, content: str, hints: dict) -> DefectObservation:
        try:
            data = _extract_json(content)
        except (json.JSONDecodeError, ValueError):
            # Unparseable response -> low-confidence unknown so the workflow
            # routes it to human review instead of failing hard.
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
            bounding_box=_parse_bbox(data.get("bounding_box")),
            length_mm=hints.get("length_mm"),
            area_ratio=hints.get("area_ratio"),
            load_bearing_area=bool(hints.get("load_bearing_area", False)),
        )
