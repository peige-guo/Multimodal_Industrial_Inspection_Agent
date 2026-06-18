"""Tests for the YOLO detector using lightweight fake model objects.

These avoid the real ultralytics/Pillow dependencies by injecting a fake model
and a pass-through image loader.
"""

from backend.app.vision.config import VisionConfig
from backend.app.vision.yolo_detector import YOLODefectDetector


class _FakeBox:
    def __init__(self, cls_id, conf, xyxyn):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxyn = [xyxyn]


class _FakeFrame:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


class _FakeModel:
    def __init__(self, frames):
        self._frames = frames

    def __call__(self, image, **kwargs):
        return self._frames


def _detector(frames, **cfg):
    config = VisionConfig(yolo_model_path="fake.pt", confidence_threshold=0.25, **cfg)
    return YOLODefectDetector(
        config=config,
        model=_FakeModel(frames),
        image_loader=lambda b: b,  # pass bytes straight through
    )


def test_yolo_picks_highest_confidence_detection():
    frames = [
        _FakeFrame(
            names={0: "crack", 1: "scratch"},
            boxes=[
                _FakeBox(1, 0.40, [0.0, 0.0, 0.2, 0.2]),
                _FakeBox(0, 0.91, [0.1, 0.2, 0.5, 0.6]),
            ],
        )
    ]
    obs = _detector(frames).detect(filename="img.png", image_bytes=b"x")
    assert obs.defect_type == "crack"
    assert obs.confidence == 0.91
    assert obs.bounding_box is not None
    assert abs(obs.bounding_box.x - 0.1) < 1e-6
    assert abs(obs.bounding_box.width - 0.4) < 1e-6
    assert obs.area_ratio is not None


def test_yolo_filters_below_threshold_returns_none_defect():
    frames = [_FakeFrame(names={0: "crack"}, boxes=[_FakeBox(0, 0.10, [0, 0, 0.3, 0.3])])]
    obs = _detector(frames).detect(filename="img.png", image_bytes=b"x")
    assert obs.defect_type == "none"


def test_yolo_class_map_translates_raw_label():
    frames = [_FakeFrame(names={0: "rost"}, boxes=[_FakeBox(0, 0.8, [0, 0, 0.3, 0.3])])]
    det = _detector(frames, yolo_class_map={"rost": "corrosion"})
    obs = det.detect(filename="img.png", image_bytes=b"x")
    assert obs.defect_type == "corrosion"


def test_yolo_no_image_is_low_confidence_unknown():
    obs = _detector([]).detect(filename="img.png", image_bytes=None)
    assert obs.defect_type == "unknown"
    assert obs.confidence == 0.0


def test_yolo_hints_override_load_bearing_and_length():
    frames = [_FakeFrame(names={0: "crack"}, boxes=[_FakeBox(0, 0.8, [0, 0, 0.3, 0.3])])]
    obs = _detector(frames).detect(
        filename="img.png",
        image_bytes=b"x",
        hints={"load_bearing_area": True, "length_mm": 9.0, "location": "weld"},
    )
    assert obs.load_bearing_area is True
    assert obs.length_mm == 9.0
    assert obs.location == "weld"
