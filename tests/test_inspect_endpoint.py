import io
import json

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)

STANDARD_PATH = "data/sample_standards/pipeline_surface_defect_standard.md"


def _standard_bytes() -> bytes:
    with open(STANDARD_PATH, "rb") as f:
        return f.read()


def test_inspect_basic_crack():
    files = {
        "image": ("pipe_crack_001.png", b"\x89PNG_fake_bytes", "image/png"),
        "standard": (
            "standard.md",
            _standard_bytes(),
            "text/markdown",
        ),
    }
    resp = client.post("/api/inspect", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["defect_type"] == "crack"
    assert "severity_level" in body
    assert "recommended_action" in body
    assert body["standard_evidence"]


def test_inspect_with_hints_and_sensor():
    files = {
        "image": ("pipe.png", b"fake", "image/png"),
        "standard": ("standard.md", _standard_bytes(), "text/markdown"),
        "sensor_csv": ("sensors.csv", b"temperature,vibration\n90,3\n", "text/csv"),
    }
    data = {
        "object_name": "pipe_42",
        "vision_hints": json.dumps(
            {"defect_type": "crack", "confidence": 0.9, "length_mm": 9, "load_bearing_area": True}
        ),
    }
    resp = client.post("/api/inspect", files=files, data=data)
    assert resp.status_code == 200
    body = resp.json()
    assert body["object_name"] == "pipe_42"
    assert body["severity_level"] == "critical"
    assert body["recommended_action"] == "stop_machine"
    assert body["requires_human_review"] is True


def test_inspect_unsupported_standard_returns_415():
    files = {
        "image": ("pipe.png", b"fake", "image/png"),
        "standard": ("standard.docx", b"binary", "application/octet-stream"),
    }
    resp = client.post("/api/inspect", files=files)
    assert resp.status_code == 415


def test_inspect_invalid_hints_returns_422():
    files = {
        "image": ("pipe.png", b"fake", "image/png"),
        "standard": ("standard.md", _standard_bytes(), "text/markdown"),
    }
    resp = client.post("/api/inspect", files=files, data={"vision_hints": "{not json"})
    assert resp.status_code == 422
