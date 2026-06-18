"""Pydantic models that define the inspection input/output contract.

These schemas are the single source of truth for data that flows through the
inspection workflow: vision observation -> standard evidence -> decision ->
final structured report.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class SeverityLevel(str, Enum):
    """Defect severity, ordered from least to most severe."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Numeric rank so severities can be compared and combined."""
        order = {
            SeverityLevel.LOW: 0,
            SeverityLevel.MEDIUM: 1,
            SeverityLevel.HIGH: 2,
            SeverityLevel.CRITICAL: 3,
        }
        return order[self]


class RecommendedAction(str, Enum):
    """Action recommended to the operator after inspection."""

    RELEASE = "release"
    REINSPECT = "reinspect"
    REPAIR = "repair"
    REJECT = "reject"
    STOP_MACHINE = "stop_machine"


class BoundingBox(BaseModel):
    """Normalized bounding box in [0, 1] coordinates relative to image size."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)


class DefectObservation(BaseModel):
    """What the vision component saw in the image."""

    defect_type: str = Field(description="Detected defect class, e.g. 'crack'.")
    description: str = Field(description="Human-readable visual description.")
    confidence: float = Field(ge=0.0, le=1.0)
    location: Optional[str] = Field(
        default=None, description="Coarse text location, e.g. 'top-left weld'."
    )
    bounding_box: Optional[BoundingBox] = None
    # Optional measured attributes used by rule-based severity logic.
    length_mm: Optional[float] = Field(default=None, ge=0.0)
    area_ratio: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Fraction of local area affected."
    )
    load_bearing_area: bool = False


class StandardEvidence(BaseModel):
    """A retrieved clause from the inspection standard document."""

    clause_id: Optional[str] = Field(
        default=None, description="Clause identifier such as '3.2' if available."
    )
    title: Optional[str] = None
    text: str
    score: float = Field(ge=0.0, description="Relevance score from the retriever.")


class SensorReading(BaseModel):
    """A single numeric sensor measurement."""

    name: str
    value: float
    unit: Optional[str] = None


class InspectionReport(BaseModel):
    """Final structured report returned by the workflow."""

    inspection_id: str = Field(default_factory=lambda: f"insp_{uuid4().hex[:12]}")
    object_name: str
    inspection_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    input_data_summary: str

    defect_type: str
    defect_location: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)

    severity_level: SeverityLevel
    recommended_action: RecommendedAction
    risk_explanation: str

    standard_evidence: list[StandardEvidence] = Field(default_factory=list)
    sensor_readings: list[SensorReading] = Field(default_factory=list)

    requires_human_review: bool
    human_review_reasons: list[str] = Field(default_factory=list)

    annotated_image_path: Optional[str] = None

    def to_markdown(self) -> str:
        """Render the report as a readable Markdown summary."""
        evidence_lines = (
            "\n".join(
                f"- **{e.clause_id or 'clause'}** {e.title or ''}: {e.text}".strip()
                for e in self.standard_evidence
            )
            or "- _No matching standard clause found._"
        )
        sensor_lines = (
            "\n".join(
                f"- {s.name}: {s.value}{(' ' + s.unit) if s.unit else ''}"
                for s in self.sensor_readings
            )
            or "- _No sensor data provided._"
        )
        review_lines = (
            "\n".join(f"- {r}" for r in self.human_review_reasons)
            or "- _None._"
        )
        return (
            f"# Inspection Report {self.inspection_id}\n\n"
            f"- **Object:** {self.object_name}\n"
            f"- **Time:** {self.inspection_time.isoformat()}\n"
            f"- **Input summary:** {self.input_data_summary}\n\n"
            f"## Defect\n"
            f"- **Type:** {self.defect_type}\n"
            f"- **Location:** {self.defect_location or 'n/a'}\n"
            f"- **Confidence:** {self.confidence:.2f}\n\n"
            f"## Decision\n"
            f"- **Severity:** {self.severity_level.value}\n"
            f"- **Recommended action:** {self.recommended_action.value}\n"
            f"- **Risk:** {self.risk_explanation}\n\n"
            f"## Standard Evidence\n{evidence_lines}\n\n"
            f"## Sensor Readings\n{sensor_lines}\n\n"
            f"## Human Review\n"
            f"- **Required:** {'yes' if self.requires_human_review else 'no'}\n"
            f"{review_lines}\n"
        )
