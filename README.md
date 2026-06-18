# Multimodal Industrial Inspection Agent

A multimodal intelligent inspection agent for industrial quality inspection scenarios. The system combines images, inspection standard documents, and structured sensor data to perform defect identification, standards-based evidence retrieval, risk grading, report generation, and human review recommendations.

## One-Sentence Positioning

Multimodal_Industrial_Inspection_Agent is a multimodal intelligent inspection agent for industrial quality inspection. It combines images, videos, textual standards, and sensor data to identify, explain, classify, and report equipment or product defects.

## MVP Goal

Upload one industrial defect image, one inspection standard document, and an optional sensor CSV. The system automatically:

1. Identifies and describes the defect
2. Retrieves relevant inspection standards
3. Determines the severity level
4. Provides risk explanation and recommended action
5. Generates a structured Inspection Report
6. Flags whether human review is required

## MVP Scope

Included:
- Image upload
- Inspection standard document parsing
- RAG-based retrieval of standard evidence
- Defect description and simplified localization
- Agent-based decision making
- Structured report generation
- Frontend result display

Not included yet:
- Real-time video streaming
- Large-scale model training
- Multi-sensor time-series fusion
- Enterprise permission system
- Automated work order system
- Edge device deployment

## Recommended Tech Stack

Backend: Python, FastAPI, LangGraph, Chroma/FAISS, PyMuPDF, Pydantic

Vision: OpenAI/Gemini/Qwen-VL/LLaVA for the MVP; YOLO + SAM for phase 2

Frontend: React / Next.js

Data: MVTec AD, DAGM, NEU surface defect, KolektorSDD

## Current Directory Layout

```text
backend/app/main.py              # FastAPI app: /health + /api/inspect, CORS
backend/app/api/inspect.py       # /api/inspect multipart endpoint
backend/app/schemas/inspection.py# Pydantic data contract (report, defect, etc.)
backend/app/rag/document_loader.py # txt/md/pdf text extraction
backend/app/rag/retriever.py     # clause chunker + lexical retriever
backend/app/vision/defect_detector.py # detector interface + heuristic impl
backend/app/vision/yolo_detector.py # Ultralytics YOLO backend (Phase 2)
backend/app/vision/qwen_detector.py # local Qwen2.5-VL backend (Phase 2)
backend/app/vision/vlm_detector.py  # OpenAI-compatible VLM backend (Phase 2)
backend/app/vision/vlm_common.py # shared VLM prompt + JSON parsing
backend/app/vision/factory.py    # env-driven backend selection + fallback
backend/app/vision/annotator.py  # bounding-box image annotation
backend/app/agents/inspection_agent.py # severity/action/human-review decision
backend/app/services/sensor_loader.py  # sensor CSV parsing + thresholds
backend/app/services/report_generator.py # assemble InspectionReport
backend/app/services/workflow.py # end-to-end orchestration
frontend/index.html              # static upload + report UI (no build step)
frontend/app.js
frontend/styles.css
data/sample_standards/           # demo inspection standard
data/sample_reports/             # example report JSON + sample sensor CSV
notebooks/
docs/
```

## Quickstart

```bash
uv venv .venv
uv pip install -p .venv/bin/python -r backend/requirements.txt
.venv/bin/python -m pytest tests/ -q
.venv/bin/uvicorn backend.app.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health   # -> {"status":"ok"}
```

Run a full inspection:

```bash
curl -X POST http://127.0.0.1:8000/api/inspect \
  -F "image=@your_defect.png" \
  -F "standard=@data/sample_standards/pipeline_surface_defect_standard.md" \
  -F "sensor_csv=@data/sample_reports/sample_sensors.csv" \
  -F 'vision_hints={"defect_type":"crack","confidence":0.9,"length_mm":8,"load_bearing_area":true}'
```

The response is a structured `InspectionReport` (see `data/sample_reports/example_report.json`).

### Frontend

The frontend is a dependency-free static page. Start the backend, then serve the UI:

```bash
python -m http.server 5173 --directory frontend
# open http://127.0.0.1:5173
```

Upload an image + standard (+ optional sensor CSV), then view the rendered report.

## How It Works

The `/api/inspect` request runs this deterministic pipeline (`services/workflow.py`):

1. **Vision** — `DefectDetector` returns a `DefectObservation` (heuristic by default; YOLO or VLM backends in Phase 2, see below).
2. **RAG** — the standard document is parsed and chunked into clauses, then relevant clauses are retrieved lexically.
3. **Sensors** — optional CSV is parsed; the latest reading per column is checked against critical thresholds.
4. **Decision** — transparent rules assign severity, recommended action, and the human-review flag.
5. **Report** — everything is assembled into an `InspectionReport` (JSON + Markdown).

Human review is triggered when confidence < 0.70, no standard clause matches, severity is high/critical, or sensor and visual evidence conflict.

## Vision Backends (Phase 2)

The vision step is pluggable behind the `DefectDetector` interface and selected via environment variables (defaults to the dependency-free heuristic):

```bash
pip install -r backend/requirements.txt -r backend/requirements-vision.txt

# Local Qwen-VL (in-process, no server)
export INSPECTION_DETECTOR=qwen
export INSPECTION_QWEN_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
export INSPECTION_QWEN_DEVICE=auto   # cuda / mps / cpu

# or YOLO
export INSPECTION_DETECTOR=yolo
export INSPECTION_YOLO_MODEL=runs/defect/best.pt

# or Qwen-VL via a local OpenAI-compatible server (Ollama/vLLM)
export INSPECTION_DETECTOR=vlm
export INSPECTION_VLM_MODEL=qwen2.5vl
export INSPECTION_VLM_BASE_URL=http://localhost:11434/v1
export INSPECTION_VLM_API_KEY=ollama
```

Backends: `heuristic` (default), `yolo` (Ultralytics), `qwen` (local Qwen2.5-VL via transformers), `vlm` (OpenAI-compatible server, including Qwen on Ollama/vLLM), `auto`. If a selected backend's dependency/config is missing, the factory falls back to the heuristic detector so the service stays up. Pass `annotate=true` to `/api/inspect` to save a bounding-box image (`annotated_image_path`). Full details in `docs/phase2_vision.md`.

## Extending the MVP

- **Real vision:** subclass `DefectDetector` (VLM/YOLO/SAM), normalize labels via `vision/labels.py`, and register it in `vision/factory.py`.
- **Embedding retrieval:** swap `StandardRetriever` for a FAISS/Chroma-backed retriever with the same `retrieve()` contract.
- **LLM explanations:** layer an LLM on top of the rule-based `decide()` for ambiguous cases.

See `docs/mvp_implementation_plan.md`, `docs/system_design.md`, and `docs/phase2_vision.md` for the full design.
