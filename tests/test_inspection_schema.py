import pytest
from pydantic import ValidationError

from backend.app.schemas.inspection import (
    DefectObservation,
    InspectionReport,
    RecommendedAction,
    SensorReading,
    SeverityLevel,
    StandardEvidence,
)


def test_severity_rank_ordering():
    assert SeverityLevel.LOW.rank < SeverityLevel.MEDIUM.rank
    assert SeverityLevel.MEDIUM.rank < SeverityLevel.HIGH.rank
    assert SeverityLevel.HIGH.rank < SeverityLevel.CRITICAL.rank


def test_defect_observation_confidence_bounds():
    with pytest.raises(ValidationError):
        DefectObservation(defect_type="crack", description="x", confidence=1.5)


def test_inspection_report_defaults_and_markdown():
    report = InspectionReport(
        object_name="pipe_section_A",
        input_data_summary="1 image, 1 standard",
        defect_type="crack",
        defect_location="top weld",
        confidence=0.92,
        severity_level=SeverityLevel.HIGH,
        recommended_action=RecommendedAction.REPAIR,
        risk_explanation="Crack near weld exceeds 5mm.",
        standard_evidence=[
            StandardEvidence(clause_id="3.2", text="Serious crack clause", score=2.0)
        ],
        sensor_readings=[SensorReading(name="temperature", value=85.0, unit="C")],
        requires_human_review=True,
        human_review_reasons=["high severity"],
    )

    assert report.inspection_id.startswith("insp_")
    assert report.inspection_time.tzinfo is not None

    md = report.to_markdown()
    assert "Inspection Report" in md
    assert "crack" in md
    assert "3.2" in md
    assert "temperature" in md


def test_report_serializes_to_json():
    report = InspectionReport(
        object_name="obj",
        input_data_summary="summary",
        defect_type="scratch",
        confidence=0.5,
        severity_level=SeverityLevel.LOW,
        recommended_action=RecommendedAction.RELEASE,
        risk_explanation="minor",
        requires_human_review=False,
    )
    payload = report.model_dump(mode="json")
    assert payload["severity_level"] == "low"
    assert payload["recommended_action"] == "release"
