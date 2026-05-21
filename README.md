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
backend/app/main.py
backend/app/api/
backend/app/agents/
backend/app/vision/
backend/app/rag/
backend/app/schemas/
backend/app/services/
frontend/app/
frontend/components/
data/sample_images/
data/sample_standards/
data/sample_reports/
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
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## Next Step

Implement the backend core loop first: schema -> document parser -> retriever -> detector adapter -> inspection agent -> report generator -> FastAPI endpoint. See `docs/mvp_implementation_plan.md`.
