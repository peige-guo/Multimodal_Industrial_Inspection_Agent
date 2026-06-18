"""YOLO-based defect detector (Phase 2).

Wraps an Ultralytics YOLO model behind the :class:`DefectDetector` interface.
Heavy dependencies (``ultralytics``, ``Pillow``) are imported lazily so this
module can be imported even when they are not installed; the import only fails
when a YOLO detection is actually attempted.

The model object can be injected (``model=...``) which keeps the class unit
testable without the real dependency.
"""

from __future__ import annotations

import io
from typing import Any, Callable, Optional

from backend.app.schemas.inspection import BoundingBox, DefectObservation
from backend.app.vision.config import VisionConfig
from backend.app.vision.defect_detector import DefectDetector
from backend.app.vision.labels import canonical_defect_type, describe


def _load_yolo(model_path: str) -> Any:
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise RuntimeError(
            "ultralytics is required for the YOLO detector. "
            "Install it with: pip install -r backend/requirements-vision.txt"
        ) from exc
    return YOLO(model_path)


def _open_image(image_bytes: bytes) -> Any:
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise RuntimeError(
            "Pillow is required for the YOLO detector. "
            "Install it with: pip install -r backend/requirements-vision.txt"
        ) from exc
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _bbox_from_xyxyn(coords: list[float]) -> Optional[BoundingBox]:
    """Build a normalized BoundingBox from [x1, y1, x2, y2] in [0, 1]."""
    x1, y1, x2, y2 = (float(c) for c in coords[:4])
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    width, height = x2 - x1, y2 - y1
    if width <= 0.0 or height <= 0.0:
        return None
    return BoundingBox(x=x1, y=y1, width=width, height=height)


class YOLODefectDetector(DefectDetector):
    """Defect detector backed by an Ultralytics YOLO model."""

    def __init__(
        self,
        config: Optional[VisionConfig] = None,
        model: Any = None,
        image_loader: Optional[Callable[[bytes], Any]] = None,
    ) -> None:
        self.config = config or VisionConfig()
        self._model = model
        # Overridable so the detector is testable without Pillow installed.
        self._image_loader = image_loader or _open_image

    def _get_model(self) -> Any:
        if self._model is None:
            if not self.config.yolo_model_path:
                raise RuntimeError(
                    "No YOLO model configured. Set INSPECTION_YOLO_MODEL to a "
                    "model path/name (e.g. yolov8n.pt or a fine-tuned weights file)."
                )
            self._model = _load_yolo(self.config.yolo_model_path)
        return self._model

    def _map_class(self, class_name: str) -> str:
        mapped = self.config.yolo_class_map.get(class_name, class_name)
        return canonical_defect_type(mapped)

    def detect(
        self,
        *,
        filename: str,
        image_bytes: Optional[bytes] = None,
        hints: Optional[dict] = None,
    ) -> DefectObservation:
        hints = hints or {}
        if not image_bytes:
            # No pixels to analyze; emit a low-confidence result so the
            # workflow flags it for human review.
            return DefectObservation(
                defect_type="unknown",
                description="No image data provided to the YOLO detector.",
                confidence=0.0,
                location=hints.get("location"),
            )

        model = self._get_model()
        image = self._image_loader(image_bytes)
        results = model(image, verbose=False) if _accepts_verbose(model) else model(image)

        best = self._select_best(results)
        if best is None:
            return DefectObservation(
                defect_type="none",
                description="No defect detected above the confidence threshold.",
                confidence=float(hints.get("confidence", 0.6)),
                location=hints.get("location"),
                load_bearing_area=bool(hints.get("load_bearing_area", False)),
            )

        class_name, confidence, bbox = best
        defect_type = self._map_class(class_name)
        area_ratio = hints.get("area_ratio")
        if area_ratio is None and bbox is not None:
            area_ratio = round(min(1.0, bbox.width * bbox.height), 4)

        return DefectObservation(
            defect_type=defect_type,
            description=describe(defect_type),
            confidence=max(0.0, min(1.0, confidence)),
            location=hints.get("location"),
            bounding_box=bbox,
            length_mm=hints.get("length_mm"),
            area_ratio=area_ratio,
            load_bearing_area=bool(hints.get("load_bearing_area", False)),
        )

    def _select_best(
        self, results: Any
    ) -> Optional[tuple[str, float, Optional[BoundingBox]]]:
        """Pick the highest-confidence detection across all result frames."""
        threshold = self.config.confidence_threshold
        best: Optional[tuple[str, float, Optional[BoundingBox]]] = None

        for frame in _iter(results):
            names = getattr(frame, "names", None) or {}
            boxes = getattr(frame, "boxes", None)
            if boxes is None:
                continue
            for box in _iter(boxes):
                conf = _scalar(getattr(box, "conf", None))
                cls_id = _scalar(getattr(box, "cls", None))
                if conf is None or cls_id is None:
                    continue
                if conf < threshold:
                    continue
                class_name = str(names.get(int(cls_id), int(cls_id)))
                bbox = None
                xyxyn = getattr(box, "xyxyn", None)
                if xyxyn is not None:
                    coords = _to_list(xyxyn)
                    if coords:
                        bbox = _bbox_from_xyxyn(coords)
                if best is None or conf > best[1]:
                    best = (class_name, float(conf), bbox)
        return best


def _accepts_verbose(model: Any) -> bool:
    # Real ultralytics models accept verbose=; simple test doubles may not.
    return model.__class__.__module__.startswith("ultralytics")


def _iter(obj: Any):
    try:
        return list(obj)
    except TypeError:
        return [obj]


def _scalar(value: Any) -> Optional[float]:
    """Coerce a possibly-tensor/array/list scalar into a float."""
    if value is None:
        return None
    # Tensors/arrays expose .item() or are indexable.
    item = getattr(value, "item", None)
    try:
        if callable(item):
            try:
                return float(value.item())
            except (ValueError, RuntimeError, TypeError):
                pass
        if hasattr(value, "__len__") and len(value) > 0:
            return float(value[0])
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_list(value: Any) -> list[float]:
    """Flatten a tensor/array/nested list into a flat list of floats."""
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    flat: list[float] = []
    stack = [value]
    while stack:
        cur = stack.pop(0)
        if isinstance(cur, (list, tuple)):
            stack[0:0] = list(cur)
        else:
            try:
                flat.append(float(cur))
            except (TypeError, ValueError):
                continue
    return flat
