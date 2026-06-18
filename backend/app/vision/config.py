"""Configuration for vision backend selection.

Values are read from environment variables so the deployment can switch between
the heuristic stub, a YOLO model, or a vision-language model without code
changes. Defaults keep the dependency-free heuristic detector active.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DetectorBackend = str  # "heuristic" | "yolo" | "vlm" | "auto"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class VisionConfig:
    """Resolved vision configuration."""

    backend: DetectorBackend = "heuristic"
    confidence_threshold: float = 0.25

    # YOLO settings.
    yolo_model_path: str | None = None
    yolo_class_map: dict[str, str] = field(default_factory=dict)

    # VLM settings (OpenAI-compatible Chat Completions API).
    vlm_model: str = "gpt-4o-mini"
    vlm_api_key: str | None = None
    vlm_base_url: str | None = None

    @classmethod
    def from_env(cls) -> "VisionConfig":
        backend = (_env("INSPECTION_DETECTOR", "heuristic") or "heuristic").lower()

        class_map: dict[str, str] = {}
        raw_map = _env("INSPECTION_YOLO_CLASS_MAP")
        if raw_map:
            # Format: "0=crack,1=corrosion" or "crack=crack,rust=corrosion".
            for pair in raw_map.split(","):
                if "=" in pair:
                    key, val = pair.split("=", 1)
                    class_map[key.strip()] = val.strip()

        return cls(
            backend=backend,
            confidence_threshold=_env_float("INSPECTION_CONFIDENCE_THRESHOLD", 0.25),
            yolo_model_path=_env("INSPECTION_YOLO_MODEL"),
            yolo_class_map=class_map,
            vlm_model=_env("INSPECTION_VLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
            vlm_api_key=_env("INSPECTION_VLM_API_KEY") or _env("OPENAI_API_KEY"),
            vlm_base_url=_env("INSPECTION_VLM_BASE_URL"),
        )
