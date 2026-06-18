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
| `qwen`      | `QwenVLDefectDetector`  | `transformers`, `torch`, `qwen-vl-utils`, `pillow` | **Local Qwen2.5-VL**, fully in-process, no server. |
| `vlm`       | `VLMDefectDetector`     | `openai`                | Any OpenAI-compatible server (incl. Qwen via Ollama/vLLM). |
| `auto`      | (resolves at runtime)   | as configured           | YOLO if a model is set, else local Qwen if installed, else a VLM server if a key is set, else heuristic. |

Both VLM backends share the same prompt and JSON parsing (`vision/vlm_common.py`).

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
| `INSPECTION_QWEN_MODEL` | Local Qwen-VL model id/path | `Qwen/Qwen2.5-VL-3B-Instruct` |
| `INSPECTION_QWEN_DEVICE` | `device_map` for loading | `auto`, `cuda`, `cpu`, `mps` |
| `INSPECTION_QWEN_MAX_NEW_TOKENS` | Generation length cap | `256` |
| `INSPECTION_VLM_MODEL` | VLM server model id | `qwen2.5vl` |
| `INSPECTION_VLM_API_KEY` | API key (falls back to `OPENAI_API_KEY`) | `ollama` |
| `INSPECTION_VLM_BASE_URL` | OpenAI-compatible endpoint | `http://localhost:11434/v1` |

### Examples

Local Qwen-VL (recommended local option, in-process, no server):

```bash
pip install -r backend/requirements.txt -r backend/requirements-vision.txt
export INSPECTION_DETECTOR=qwen
export INSPECTION_QWEN_MODEL=Qwen/Qwen2.5-VL-3B-Instruct   # 7B/72B also supported
export INSPECTION_QWEN_DEVICE=auto                         # cuda / mps / cpu
uvicorn backend.app.main:app --reload
```

The model weights download from Hugging Face on first use and are cached. Pick
the variant that fits your hardware (3B for modest GPUs/CPU, 7B/72B for more).

Qwen-VL via a local server (Ollama / vLLM) using the OpenAI-compatible backend:

```bash
# e.g. `ollama run qwen2.5vl` then:
export INSPECTION_DETECTOR=vlm
export INSPECTION_VLM_MODEL=qwen2.5vl
export INSPECTION_VLM_BASE_URL=http://localhost:11434/v1
export INSPECTION_VLM_API_KEY=ollama   # any non-empty string for local servers
uvicorn backend.app.main:app --reload
```

YOLO with a fine-tuned model:

```bash
export INSPECTION_DETECTOR=yolo
export INSPECTION_YOLO_MODEL=runs/defect/best.pt
export INSPECTION_YOLO_CLASS_MAP="0=crack,1=corrosion,2=scratch"
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
