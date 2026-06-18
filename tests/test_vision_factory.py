"""Tests for detector selection and graceful fallback."""

from backend.app.vision.config import VisionConfig
from backend.app.vision.defect_detector import HeuristicDefectDetector
from backend.app.vision.factory import build_detector


def test_default_is_heuristic():
    det = build_detector(VisionConfig(backend="heuristic"))
    assert isinstance(det, HeuristicDefectDetector)


def test_unknown_backend_falls_back_to_heuristic():
    det = build_detector(VisionConfig(backend="does-not-exist"))
    assert isinstance(det, HeuristicDefectDetector)


def test_yolo_without_model_path_falls_back():
    det = build_detector(VisionConfig(backend="yolo", yolo_model_path=None))
    assert isinstance(det, HeuristicDefectDetector)


def test_vlm_without_api_key_falls_back():
    det = build_detector(VisionConfig(backend="vlm", vlm_api_key=None))
    assert isinstance(det, HeuristicDefectDetector)


def test_auto_with_nothing_configured_is_heuristic():
    det = build_detector(VisionConfig(backend="auto"))
    assert isinstance(det, HeuristicDefectDetector)


def test_from_env_reads_backend(monkeypatch):
    monkeypatch.setenv("INSPECTION_DETECTOR", "yolo")
    monkeypatch.setenv("INSPECTION_YOLO_MODEL", "weights.pt")
    monkeypatch.setenv("INSPECTION_YOLO_CLASS_MAP", "0=crack,1=corrosion")
    monkeypatch.setenv("INSPECTION_CONFIDENCE_THRESHOLD", "0.5")
    cfg = VisionConfig.from_env()
    assert cfg.backend == "yolo"
    assert cfg.yolo_model_path == "weights.pt"
    assert cfg.yolo_class_map == {"0": "crack", "1": "corrosion"}
    assert cfg.confidence_threshold == 0.5


def test_yolo_built_when_dependencies_present(monkeypatch):
    # Simulate ultralytics + Pillow being installed without importing them.
    import backend.app.vision.factory as factory

    monkeypatch.setattr(factory, "_installed", lambda *mods: True)

    built = {}

    class _Sentinel(HeuristicDefectDetector):
        pass

    def fake_yolo_ctor(config):
        built["called"] = True
        return _Sentinel()

    # Patch the lazy import target by swapping the builder's inner import.
    monkeypatch.setattr(factory, "_build_yolo", lambda config: fake_yolo_ctor(config))
    det = build_detector(VisionConfig(backend="yolo", yolo_model_path="w.pt"))
    assert built.get("called") is True
    assert isinstance(det, _Sentinel)
