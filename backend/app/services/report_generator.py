"""Assemble the final structured InspectionReport from workflow outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from backend.app.schemas.inspection import (
    DefectObservation,
    InspectionReport,
    SensorReading,
    StandardEvidence,
)

if TYPE_CHECKING:
    from backend.app.agents.inspection_agent import InspectionDecision


def _summarize_inputs(
    *,
    image_name: str,
    standard_name: Optional[str],
    sensor_readings: list[SensorReading],
) -> str:
    parts = [f"image '{image_name}'"]
    if standard_name:
        parts.append(f"standard '{standard_name}'")
    if sensor_readings:
        parts.append(f"{len(sensor_readings)} sensor reading(s)")
    return ", ".join(parts)


def build_report(
    *,
    object_name: str,
    image_name: str,
    observation: DefectObservation,
    decision: "InspectionDecision",
    standard_evidence: list[StandardEvidence],
    sensor_readings: Optional[list[SensorReading]] = None,
    standard_name: Optional[str] = None,
    annotated_image_path: Optional[str] = None,
) -> InspectionReport:
    """Combine all pieces into the final report object."""
    sensor_readings = sensor_readings or []
    return InspectionReport(
        object_name=object_name,
        input_data_summary=_summarize_inputs(
            image_name=image_name,
            standard_name=standard_name,
            sensor_readings=sensor_readings,
        ),
        defect_type=observation.defect_type,
        defect_location=observation.location,
        confidence=observation.confidence,
        severity_level=decision.severity_level,
        recommended_action=decision.recommended_action,
        risk_explanation=decision.risk_explanation,
        standard_evidence=standard_evidence,
        sensor_readings=sensor_readings,
        requires_human_review=decision.requires_human_review,
        human_review_reasons=decision.human_review_reasons,
        annotated_image_path=annotated_image_path,
    )
