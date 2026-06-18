"""Tests for the bounding-box annotator and its workflow wiring.

Assertions branch on whether Pillow is installed so they hold in both
environments: with Pillow an annotated file is produced; without it the
annotator degrades gracefully to ``None``.
"""

import importlib.util
import os

from backend.app.rag.document_loader import load_document_text
from backend.app.schemas.inspection import BoundingBox, DefectObservation
from backend.app.services.workflow import run_inspection
from backend.app.vision.annotator import annotate_defect

_HAS_PIL = importlib.util.find_spec("PIL") is not None
SAMPLE = "data/sample_standards/pipeline_surface_defect_standard.md"


def _png_bytes() -> bytes:
    # Minimal 1x1 PNG.
    import base64

    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )


def test_no_bounding_box_returns_none(tmp_path):
    obs = DefectObservation(defect_type="crack", description="d", confidence=0.9)
    result = annotate_defect(
        image_bytes=_png_bytes(),
        observation=obs,
        output_dir=tmp_path,
        inspection_id="insp_test",
    )
    assert result is None


def test_annotate_with_bbox(tmp_path):
    obs = DefectObservation(
        defect_type="crack",
        description="d",
        confidence=0.9,
        bounding_box=BoundingBox(x=0.1, y=0.1, width=0.3, height=0.2),
    )
    result = annotate_defect(
        image_bytes=_png_bytes(),
        observation=obs,
        output_dir=tmp_path,
        inspection_id="insp_test",
    )
    if _HAS_PIL:
        assert result is not None
        assert os.path.exists(result)
    else:
        assert result is None


def test_workflow_annotate_sets_path(tmp_path):
    report = run_inspection(
        image_name="pipe_crack.png",
        image_bytes=_png_bytes(),
        standard_text=load_document_text(SAMPLE),
        vision_hints={
            "confidence": 0.9,
            "bounding_box": {"x": 0.2, "y": 0.2, "width": 0.3, "height": 0.3},
        },
        annotate=True,
        output_dir=str(tmp_path),
    )
    if _HAS_PIL:
        assert report.annotated_image_path is not None
        assert os.path.exists(report.annotated_image_path)
    else:
        assert report.annotated_image_path is None


def test_workflow_no_annotate_leaves_path_none(tmp_path):
    report = run_inspection(
        image_name="pipe_crack.png",
        image_bytes=_png_bytes(),
        standard_text=load_document_text(SAMPLE),
        vision_hints={"confidence": 0.9},
        annotate=False,
    )
    assert report.annotated_image_path is None
