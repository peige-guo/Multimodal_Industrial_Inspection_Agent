"""Decision engine: combine vision, standard evidence, and sensors.

This is the transparent, rule-based core described in the system design. It
maps a defect observation plus retrieved standard clauses plus sensor readings
into a severity level, a recommended action, a human-review flag, and a
human-readable risk explanation.

The rules are intentionally explicit (no LLM call) so behavior is deterministic
and test-covered. An LLM can later be layered on for explanation/ambiguous
cases without changing this contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.app.schemas.inspection import (
    DefectObservation,
    RecommendedAction,
    SensorReading,
    SeverityLevel,
    StandardEvidence,
)
from backend.app.services.sensor_loader import evaluate_sensor_abnormality

CONFIDENCE_REVIEW_THRESHOLD = 0.70
SERIOUS_CRACK_LENGTH_MM = 5.0
CORROSION_MEDIUM_AREA_RATIO = 0.10


@dataclass
class InspectionDecision:
    severity_level: SeverityLevel
    recommended_action: RecommendedAction
    risk_explanation: str
    requires_human_review: bool
    human_review_reasons: list[str] = field(default_factory=list)


def _base_severity(obs: DefectObservation) -> tuple[SeverityLevel, str]:
    """Derive severity from the defect observation against standard rules."""
    dtype = obs.defect_type.lower()

    if dtype in {"none", "no_defect"}:
        return SeverityLevel.LOW, "No defect detected on the inspected surface."

    if dtype == "crack":
        long_enough = (obs.length_mm or 0.0) > SERIOUS_CRACK_LENGTH_MM
        if long_enough and obs.load_bearing_area:
            return (
                SeverityLevel.HIGH,
                f"Crack longer than {SERIOUS_CRACK_LENGTH_MM} mm in a load-bearing "
                f"area (clause 3.2: serious defect).",
            )
        if long_enough or obs.load_bearing_area:
            return (
                SeverityLevel.MEDIUM,
                "Crack exceeds size limit or sits in a sensitive area.",
            )
        return SeverityLevel.LOW, "Short crack outside load-bearing areas."

    if dtype == "corrosion":
        area = obs.area_ratio or 0.0
        if obs.load_bearing_area:
            return (
                SeverityLevel.HIGH,
                "Corrosion near a weld or pressure boundary (clause 3.4).",
            )
        if area > CORROSION_MEDIUM_AREA_RATIO:
            return (
                SeverityLevel.MEDIUM,
                f"Corrosion covers more than "
                f"{int(CORROSION_MEDIUM_AREA_RATIO * 100)}% of the local area.",
            )
        return SeverityLevel.LOW, "Localized superficial corrosion."

    if dtype == "scratch":
        return SeverityLevel.LOW, "Shallow scratch without material separation."

    # Unknown / unclassified anomaly.
    return (
        SeverityLevel.MEDIUM,
        "Unclassified surface anomaly requires confirmation.",
    )


def _action_for(severity: SeverityLevel, sensor_critical: bool) -> RecommendedAction:
    if sensor_critical:
        return RecommendedAction.STOP_MACHINE
    return {
        SeverityLevel.LOW: RecommendedAction.RELEASE,
        SeverityLevel.MEDIUM: RecommendedAction.REINSPECT,
        SeverityLevel.HIGH: RecommendedAction.REPAIR,
        SeverityLevel.CRITICAL: RecommendedAction.STOP_MACHINE,
    }[severity]


def decide(
    observation: DefectObservation,
    standard_evidence: list[StandardEvidence],
    sensor_readings: list[SensorReading] | None = None,
) -> InspectionDecision:
    """Produce the inspection decision from all available evidence."""
    sensor_readings = sensor_readings or []
    severity, explanation = _base_severity(observation)

    sensor_critical, sensor_reasons = evaluate_sensor_abnormality(sensor_readings)
    if sensor_critical:
        severity = SeverityLevel.CRITICAL
        explanation = (
            f"{explanation} Critical sensor abnormality detected: "
            f"{'; '.join(sensor_reasons)} (clause 3.3)."
        )

    review_reasons: list[str] = []
    if observation.confidence < CONFIDENCE_REVIEW_THRESHOLD:
        review_reasons.append(
            f"Model confidence {observation.confidence:.2f} is below "
            f"{CONFIDENCE_REVIEW_THRESHOLD:.2f}."
        )
    if not standard_evidence and observation.defect_type.lower() not in {
        "none",
        "no_defect",
    }:
        review_reasons.append("No matching standard clause found for the defect.")
    if severity.rank >= SeverityLevel.HIGH.rank:
        review_reasons.append(f"Severity is {severity.value}.")
    if sensor_critical and observation.defect_type.lower() in {"none", "no_defect"}:
        review_reasons.append(
            "Sensor data indicates critical risk but vision found no defect "
            "(conflicting evidence)."
        )

    action = _action_for(severity, sensor_critical)

    return InspectionDecision(
        severity_level=severity,
        recommended_action=action,
        risk_explanation=explanation,
        requires_human_review=len(review_reasons) > 0,
        human_review_reasons=review_reasons,
    )
