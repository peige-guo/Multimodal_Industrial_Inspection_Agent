"""End-to-end inspection workflow orchestration.

Ties together the vision adapter, document loader, retriever, decision engine,
and report generator. This is the single entry point used by the API layer and
is easy to unit test without HTTP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from backend.app.agents.inspection_agent import decide
from backend.app.rag.document_loader import load_document_bytes
from backend.app.rag.retriever import StandardRetriever
from backend.app.schemas.inspection import InspectionReport
from backend.app.services.report_generator import build_report
from backend.app.services.sensor_loader import parse_sensor_csv
from backend.app.vision.defect_detector import DefectDetector, get_default_detector


def _build_query(observation) -> str:
    """Construct a retrieval query from the observation."""
    parts = [observation.defect_type, observation.description]
    if observation.length_mm:
        parts.append(f"length {observation.length_mm} mm")
    if observation.load_bearing_area:
        parts.append("load bearing pressure boundary weld")
    if observation.location:
        parts.append(observation.location)
    return " ".join(p for p in parts if p)


def run_inspection(
    *,
    image_name: str,
    image_bytes: Optional[bytes] = None,
    standard_text: Optional[str] = None,
    standard_bytes: Optional[bytes] = None,
    standard_name: Optional[str] = None,
    sensor_csv: Optional[bytes | str] = None,
    object_name: Optional[str] = None,
    vision_hints: Optional[dict] = None,
    detector: Optional[DefectDetector] = None,
    top_k: int = 3,
) -> InspectionReport:
    """Run the full inspection pipeline and return a structured report."""
    detector = detector or get_default_detector()

    # 1. Vision analysis.
    observation = detector.detect(
        filename=image_name, image_bytes=image_bytes, hints=vision_hints
    )

    # 2. Standard RAG.
    if standard_text is None and standard_bytes is not None:
        standard_text = load_document_bytes(
            standard_bytes, standard_name or "standard.md"
        )
    standard_evidence = []
    if standard_text:
        retriever = StandardRetriever.from_text(standard_text)
        standard_evidence = retriever.retrieve(_build_query(observation), top_k=top_k)

    # 3. Sensor check.
    sensor_readings = parse_sensor_csv(sensor_csv) if sensor_csv else []

    # 4. Decision engine.
    decision = decide(observation, standard_evidence, sensor_readings)

    # 5. Report generation.
    resolved_object = object_name or Path(image_name).stem
    return build_report(
        object_name=resolved_object,
        image_name=image_name,
        observation=observation,
        decision=decision,
        standard_evidence=standard_evidence,
        sensor_readings=sensor_readings,
        standard_name=standard_name,
    )
