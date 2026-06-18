from backend.app.agents.inspection_agent import decide
from backend.app.schemas.inspection import (
    DefectObservation,
    RecommendedAction,
    SensorReading,
    SeverityLevel,
    StandardEvidence,
)

_EVIDENCE = [StandardEvidence(clause_id="3.2", text="serious crack", score=2.0)]


def _obs(**kwargs):
    base = dict(defect_type="crack", description="d", confidence=0.9)
    base.update(kwargs)
    return DefectObservation(**base)


def test_minor_defect_is_released():
    d = decide(_obs(defect_type="scratch", confidence=0.9), _EVIDENCE)
    assert d.severity_level == SeverityLevel.LOW
    assert d.recommended_action == RecommendedAction.RELEASE
    assert d.requires_human_review is False


def test_serious_crack_high_severity_repair_and_review():
    d = decide(_obs(length_mm=8.0, load_bearing_area=True), _EVIDENCE)
    assert d.severity_level == SeverityLevel.HIGH
    assert d.recommended_action == RecommendedAction.REPAIR
    assert d.requires_human_review is True
    assert any("Severity" in r for r in d.human_review_reasons)


def test_low_confidence_triggers_review():
    d = decide(_obs(confidence=0.5), _EVIDENCE)
    assert d.requires_human_review is True
    assert any("confidence" in r.lower() for r in d.human_review_reasons)


def test_missing_evidence_triggers_review():
    d = decide(_obs(confidence=0.9), [])
    assert d.requires_human_review is True
    assert any("standard clause" in r.lower() for r in d.human_review_reasons)


def test_critical_sensor_stops_machine():
    readings = [SensorReading(name="temperature", value=95.0, unit="C")]
    d = decide(_obs(confidence=0.9), _EVIDENCE, readings)
    assert d.severity_level == SeverityLevel.CRITICAL
    assert d.recommended_action == RecommendedAction.STOP_MACHINE
    assert d.requires_human_review is True


def test_conflicting_sensor_vs_no_defect():
    readings = [SensorReading(name="vibration", value=20.0, unit="mm/s")]
    d = decide(_obs(defect_type="none", confidence=0.95), [], readings)
    assert d.severity_level == SeverityLevel.CRITICAL
    assert any("conflicting" in r.lower() for r in d.human_review_reasons)
