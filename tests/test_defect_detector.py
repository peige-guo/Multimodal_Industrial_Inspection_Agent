from backend.app.vision.defect_detector import (
    HeuristicDefectDetector,
    get_default_detector,
)


def test_classifies_from_filename():
    det = HeuristicDefectDetector()
    obs = det.detect(filename="pipe_crack_001.png")
    assert obs.defect_type == "crack"
    assert obs.confidence >= 0.7


def test_unknown_filename_low_confidence():
    det = HeuristicDefectDetector()
    obs = det.detect(filename="IMG_2024.png")
    assert obs.defect_type == "unknown"
    assert obs.confidence < 0.7


def test_hints_override_classification():
    det = HeuristicDefectDetector()
    obs = det.detect(
        filename="random.png",
        hints={
            "defect_type": "corrosion",
            "confidence": 0.95,
            "length_mm": 12.0,
            "load_bearing_area": True,
            "location": "weld seam",
        },
    )
    assert obs.defect_type == "corrosion"
    assert obs.confidence == 0.95
    assert obs.length_mm == 12.0
    assert obs.load_bearing_area is True
    assert obs.location == "weld seam"


def test_deterministic_output():
    det = get_default_detector()
    a = det.detect(filename="scratch.jpg")
    b = det.detect(filename="scratch.jpg")
    assert a.model_dump() == b.model_dump()
