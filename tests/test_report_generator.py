from backend.app.rag.document_loader import load_document_text
from backend.app.services.report_generator import build_report
from backend.app.services.workflow import run_inspection
from backend.app.agents.inspection_agent import decide
from backend.app.schemas.inspection import (
    DefectObservation,
    InspectionReport,
    SeverityLevel,
    StandardEvidence,
)

SAMPLE = "data/sample_standards/pipeline_surface_defect_standard.md"


def test_build_report_contains_required_fields():
    obs = DefectObservation(
        defect_type="crack", description="d", confidence=0.9, location="weld"
    )
    evidence = [StandardEvidence(clause_id="3.2", text="serious crack", score=2.0)]
    decision = decide(obs, evidence)
    report = build_report(
        object_name="pipe",
        image_name="pipe_crack.png",
        observation=obs,
        decision=decision,
        standard_evidence=evidence,
        standard_name="std.md",
    )
    assert isinstance(report, InspectionReport)
    assert report.defect_type == "crack"
    assert report.object_name == "pipe"
    assert "image" in report.input_data_summary


def test_full_workflow_on_sample_standard():
    report = run_inspection(
        image_name="pipeline_crack_serious.png",
        standard_text=load_document_text(SAMPLE),
        standard_name="pipeline_surface_defect_standard.md",
        vision_hints={
            "confidence": 0.9,
            "length_mm": 8.0,
            "load_bearing_area": True,
            "location": "weld seam",
        },
    )
    assert report.defect_type == "crack"
    assert report.severity_level == SeverityLevel.HIGH
    assert report.standard_evidence  # retrieved something
    assert report.requires_human_review is True


def test_workflow_with_sensor_csv_stops_machine():
    csv = "temperature,vibration\n85,2.0\n"
    report = run_inspection(
        image_name="pipe_crack.png",
        standard_text=load_document_text(SAMPLE),
        sensor_csv=csv,
        vision_hints={"confidence": 0.9},
    )
    assert report.severity_level == SeverityLevel.CRITICAL
    assert report.recommended_action.value == "stop_machine"
    assert any(r.name == "temperature" for r in report.sensor_readings)


def test_markdown_render_smoke():
    report = run_inspection(
        image_name="scratch.png",
        standard_text=load_document_text(SAMPLE),
        vision_hints={"confidence": 0.9},
    )
    md = report.to_markdown()
    assert "Inspection Report" in md
