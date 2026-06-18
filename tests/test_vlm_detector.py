"""Tests for the VLM detector using a fake OpenAI-compatible client."""

from backend.app.vision.config import VisionConfig
from backend.app.vision.vlm_detector import VLMDefectDetector


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _FakeCompletions:
    def __init__(self, content):
        self._content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Response(self._content)


class _FakeChat:
    def __init__(self, content):
        self.completions = _FakeCompletions(content)


class _FakeClient:
    def __init__(self, content):
        self.chat = _FakeChat(content)


def _detector(content):
    return VLMDefectDetector(
        config=VisionConfig(vlm_model="test-model", vlm_api_key="k"),
        client=_FakeClient(content),
    )


def test_vlm_parses_clean_json():
    content = (
        '{"defect_type": "crack", "description": "linear crack", '
        '"confidence": 0.88, "location": "weld", '
        '"bounding_box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.1}}'
    )
    obs = _detector(content).detect(filename="img.png", image_bytes=b"x")
    assert obs.defect_type == "crack"
    assert obs.confidence == 0.88
    assert obs.location == "weld"
    assert obs.bounding_box is not None


def test_vlm_parses_json_in_code_fence():
    content = '```json\n{"defect_type": "corrosion", "confidence": 0.7}\n```'
    obs = _detector(content).detect(filename="img.png", image_bytes=b"x")
    assert obs.defect_type == "corrosion"
    assert obs.confidence == 0.7


def test_vlm_unparseable_response_is_low_confidence_unknown():
    obs = _detector("sorry, I cannot help").detect(filename="img.png", image_bytes=b"x")
    assert obs.defect_type == "unknown"
    assert obs.confidence == 0.0


def test_vlm_unknown_label_maps_to_unknown():
    content = '{"defect_type": "weird_thing", "confidence": 0.5}'
    obs = _detector(content).detect(filename="img.png", image_bytes=b"x")
    assert obs.defect_type == "unknown"


def test_vlm_sends_image_data_uri():
    det = _detector('{"defect_type": "none", "confidence": 0.9}')
    det.detect(filename="part.jpg", image_bytes=b"\xff\xd8\xff")
    call = det._client.chat.completions.calls[0]
    user_content = call["messages"][1]["content"]
    image_part = next(p for p in user_content if p["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_vlm_no_image_short_circuits():
    obs = _detector("{}").detect(filename="img.png", image_bytes=None)
    assert obs.defect_type == "unknown"
    assert obs.confidence == 0.0
