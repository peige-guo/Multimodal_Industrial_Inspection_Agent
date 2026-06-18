"""Tests for the local Qwen-VL detector.

A fake ``runner`` and pass-through ``image_loader`` are injected so the tests
run without torch/transformers/qwen-vl-utils/Pillow installed.
"""

from backend.app.vision.config import VisionConfig
from backend.app.vision.qwen_detector import QwenVLDefectDetector


def _detector(content, capture=None):
    def runner(image, system_prompt, user_prompt):
        if capture is not None:
            capture["image"] = image
            capture["system"] = system_prompt
            capture["user"] = user_prompt
        return content

    return QwenVLDefectDetector(
        config=VisionConfig(backend="qwen", qwen_model="Qwen/Qwen2.5-VL-3B-Instruct"),
        runner=runner,
        image_loader=lambda b: f"img:{b!r}",
    )


def test_qwen_parses_clean_json():
    content = (
        '{"defect_type": "corrosion", "description": "rust patch", '
        '"confidence": 0.82, "location": "flange", '
        '"bounding_box": {"x": 0.2, "y": 0.3, "width": 0.25, "height": 0.15}}'
    )
    obs = _detector(content).detect(filename="part.png", image_bytes=b"x")
    assert obs.defect_type == "corrosion"
    assert obs.confidence == 0.82
    assert obs.location == "flange"
    assert obs.bounding_box is not None


def test_qwen_parses_json_in_code_fence():
    content = '```json\n{"defect_type": "crack", "confidence": 0.6}\n```'
    obs = _detector(content).detect(filename="part.png", image_bytes=b"x")
    assert obs.defect_type == "crack"
    assert obs.confidence == 0.6


def test_qwen_unparseable_is_low_confidence_unknown():
    obs = _detector("I cannot tell").detect(filename="part.png", image_bytes=b"x")
    assert obs.defect_type == "unknown"
    assert obs.confidence == 0.0


def test_qwen_no_image_short_circuits():
    obs = _detector("{}").detect(filename="part.png", image_bytes=None)
    assert obs.defect_type == "unknown"
    assert obs.confidence == 0.0


def test_qwen_hints_override_fields():
    content = '{"defect_type": "crack", "confidence": 0.7}'
    obs = _detector(content).detect(
        filename="part.png",
        image_bytes=b"x",
        hints={"load_bearing_area": True, "length_mm": 6.0, "location": "weld"},
    )
    assert obs.load_bearing_area is True
    assert obs.length_mm == 6.0
    assert obs.location == "weld"


def test_qwen_runner_receives_loaded_image_and_prompts():
    capture: dict = {}
    det = _detector('{"defect_type": "none", "confidence": 0.9}', capture=capture)
    det.detect(filename="part.png", image_bytes=b"abc")
    assert capture["image"] == "img:b'abc'"
    assert "JSON" in capture["system"]
    assert capture["user"]
