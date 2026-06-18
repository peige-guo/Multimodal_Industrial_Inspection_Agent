"""`/api/inspect` endpoint: run the full multimodal inspection workflow."""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.rag.document_loader import UnsupportedDocumentError
from backend.app.schemas.inspection import InspectionReport
from backend.app.services.workflow import run_inspection

router = APIRouter(prefix="/api", tags=["inspection"])


@router.post("/inspect", response_model=InspectionReport)
async def inspect(
    image: UploadFile = File(..., description="Industrial defect image."),
    standard: UploadFile = File(..., description="Inspection standard (txt/md/pdf)."),
    sensor_csv: Optional[UploadFile] = File(
        default=None, description="Optional sensor CSV."
    ),
    object_name: Optional[str] = Form(default=None),
    vision_hints: Optional[str] = Form(
        default=None, description="Optional JSON object of detector hints."
    ),
) -> InspectionReport:
    image_bytes = await image.read()
    standard_bytes = await standard.read()
    sensor_data = await sensor_csv.read() if sensor_csv is not None else None

    hints = None
    if vision_hints:
        try:
            hints = json.loads(vision_hints)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=422, detail=f"vision_hints must be valid JSON: {exc}"
            ) from exc
        if not isinstance(hints, dict):
            raise HTTPException(
                status_code=422, detail="vision_hints must be a JSON object."
            )

    try:
        report = run_inspection(
            image_name=image.filename or "image.png",
            image_bytes=image_bytes,
            standard_bytes=standard_bytes,
            standard_name=standard.filename,
            sensor_csv=sensor_data,
            object_name=object_name,
            vision_hints=hints,
        )
    except UnsupportedDocumentError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    return report
