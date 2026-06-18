"""VLM defect detector backed by an OpenAI-compatible API (Phase 2).

Sends the image plus a structured-output prompt to an OpenAI-compatible Chat
Completions endpoint and parses the JSON response into a ``DefectObservation``.
Works with any compatible server via ``INSPECTION_VLM_BASE_URL`` -- including
local Qwen-VL deployments served by Ollama or vLLM. For fully in-process local
Qwen inference (no server), use ``QwenVLDefectDetector`` instead.

The ``openai`` SDK is imported lazily and the client can be injected for tests.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Optional

from backend.app.schemas.inspection import DefectObservation
from backend.app.vision.config import VisionConfig
from backend.app.vision.defect_detector import DefectDetector
from backend.app.vision.vlm_common import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    observation_from_payload,
)

_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def _mime_for(filename: str) -> str:
    return _MIME_BY_SUFFIX.get(Path(filename).suffix.lower(), "image/png")


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
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        return observation_from_payload(content, hints)
