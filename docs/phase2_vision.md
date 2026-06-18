# Phase 2: Stronger Vision (YOLO / VLM)

Phase 2 replaces the heuristic stub with real model backends behind the same
`DefectDetector` interface. The rest of the pipeline (RAG, decision engine,
report) is unchanged: every backend normalizes its output to the same canonical
`DefectObservation`.

## Backends

| Backend     | Class                   | Dependencies            | Notes |
|-------------|-------------------------|-------------------------|-------|
| `heuristic` | `HeuristicDefectDetector` | none (default)        | Deterministic stub for dev/tests. |
| `yolo`      | `YOLODefectDetector`    | `ultralytics`, `pillow` | Ultralytics YOLOv8/YOLOv11 detection + bbox. |
| `vlm`       | `VLMDefectDetector`     | `openai`                | OpenAI-compatible vision model, JSON output. |
| `auto`      | (resolves at runtime)   | as configured           | YOLO if a model is set, else VLM if a key is set, else heuristic. |

All canonical defect types: `crack`, `corrosion`, `scratch`, `dent`, `none`,
`unknown` (see `backend/app/vision/labels.py`).

## Install

```bash
pip install -r backend/requirements.txt -r backend/requirements-vision.txt
```

The base app never imports these packages; they are loaded lazily only when a
model backend is actually used. If a backend is selected but its dependency or
configuration is missing, the factory logs a warning and falls back to the
heuristic detector so the service stays available.

## Configuration (environment variables)

| Variable | Purpose | Example |
|----------|---------|---------|
| `INSPECTION_DETECTOR` | Backend: `heuristic` \| `yolo` \| `vlm` \| `auto` | `yolo` |
| `INSPECTION_CONFIDENCE_THRESHOLD` | Min detection confidence | `0.25` |
| `INSPECTION_YOLO_MODEL` | YOLO weights path/name | `runs/defect/best.pt` |
| `INSPECTION_YOLO_CLASS_MAP` | Map raw classes to canonical types | `0=crack,1=corrosion` |
| `INSPECTION_VLM_MODEL` | VLM model id | `gpt-4o-mini` |
| `INSPECTION_VLM_API_KEY` | API key (falls back to `OPENAI_API_KEY`) | `sk-...` |
| `INSPECTION_VLM_BASE_URL` | Custom OpenAI-compatible endpoint | `https://...` |

### Examples

YOLO with a fine-tuned model:

```bash
export INSPECTION_DETECTOR=yolo
export INSPECTION_YOLO_MODEL=runs/defect/best.pt
export INSPECTION_YOLO_CLASS_MAP="0=crack,1=corrosion,2=scratch"
uvicorn backend.app.main:app --reload
```

VLM via OpenAI-compatible API:

```bash
export INSPECTION_DETECTOR=vlm
export INSPECTION_VLM_MODEL=gpt-4o-mini
export INSPECTION_VLM_API_KEY=sk-...
uvicorn backend.app.main:app --reload
```

## Annotated images

Pass `annotate=true` to `POST /api/inspect` (or `annotate=True` to
`run_inspection`) to draw the detected bounding box on the image. The annotated
PNG is written to the output directory (default `outputs/`) and its path is
returned as `annotated_image_path`. Requires Pillow; without it the field stays
`null`.

## Extending

- **New model:** subclass `DefectDetector`, normalize output via
  `backend/app/vision/labels.canonical_defect_type`, and register it in
  `backend/app/vision/factory.build_detector`.
- **Segmentation (SAM):** add masks to `DefectObservation` and feed mask area
  into `area_ratio` for the corrosion severity rules.
- **Evaluation:** see `docs/evaluation.md` for vision metrics (defect accuracy,
  localization IoU, FP/FN) on MVTec AD / NEU / KolektorSDD.
