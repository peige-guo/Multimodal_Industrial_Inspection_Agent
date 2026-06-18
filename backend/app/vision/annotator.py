"""Draw the detected bounding box onto the image (Phase 2).

Produces an annotated copy of the input image with the defect box and label,
saved under an output directory, and returns its path. Pillow is imported
lazily; if it is unavailable the function returns ``None`` so the workflow can
continue without an annotated image.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

from backend.app.schemas.inspection import DefectObservation

logger = logging.getLogger(__name__)

_SEVERITY_COLOR = (220, 38, 38)  # red box; readable on most surfaces


def annotate_defect(
    *,
    image_bytes: bytes,
    observation: DefectObservation,
    output_dir: str | Path,
    inspection_id: str,
) -> Optional[str]:
    """Render the bounding box + label and save the annotated image.

    Returns the saved file path, or ``None`` when there is no box to draw or
    Pillow is not installed.
    """
    if observation.bounding_box is None or not image_bytes:
        return None

    try:
        from PIL import Image, ImageDraw  # type: ignore
    except ImportError:  # pragma: no cover - exercised only without dep
        logger.warning("Pillow not installed; skipping image annotation.")
        return None

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - invalid/unsupported image bytes
        logger.warning("Could not open image for annotation: %s", exc)
        return None

    width, height = image.size
    box = observation.bounding_box
    left = box.x * width
    top = box.y * height
    right = (box.x + box.width) * width
    bottom = (box.y + box.height) * height

    draw = ImageDraw.Draw(image)
    draw.rectangle([left, top, right, bottom], outline=_SEVERITY_COLOR, width=3)
    label = f"{observation.defect_type} {observation.confidence:.0%}"
    draw.text((left + 3, max(0, top - 12)), label, fill=_SEVERITY_COLOR)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{inspection_id}_annotated.png"
    image.save(out_path)
    return str(out_path)
