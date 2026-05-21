# MVP Implementation Plan

> For Hermes: Use subagent-driven-development skill to implement this plan task-by-task if execution is requested.

**Goal:** Build an end-to-end MVP where a user uploads an industrial defect image and an inspection standard document, then receives a structured defect/risk/report result.

**Architecture:** FastAPI backend orchestrates a deterministic inspection workflow. The MVP uses adapter interfaces for vision and LLM/RAG so local stubs can be tested first, then replaced by OpenAI/Gemini/Qwen-VL/YOLO/SAM later.

**Tech Stack:** Python, FastAPI, Pydantic, PyMuPDF, FAISS/Chroma optional, LangGraph optional, React/Next.js optional for UI.

---

## Task 1: Initialize Python backend package

Objective: Create importable backend package and a minimal health endpoint.

Files:
- Create: `backend/app/__init__.py`
- Create: `backend/app/main.py`
- Create: `backend/requirements.txt`
- Create: `tests/test_health.py`

Verification:
- Run `pytest tests/test_health.py -v`
- Expected: health endpoint returns `{"status": "ok"}`

## Task 2: Define inspection schemas

Objective: Define Pydantic models for input metadata, defect evidence, standard evidence, and final report.

Files:
- Create: `backend/app/schemas/inspection.py`
- Create: `tests/test_inspection_schema.py`

Key models:
- `SeverityLevel`: low, medium, high, critical
- `RecommendedAction`: release, reinspect, repair, reject, stop_machine
- `DefectObservation`
- `StandardEvidence`
- `InspectionReport`

Verification:
- Run `pytest tests/test_inspection_schema.py -v`

## Task 3: Implement document loader

Objective: Extract text from `.txt`, `.md`, and `.pdf` standard documents.

Files:
- Create: `backend/app/rag/document_loader.py`
- Create: `tests/test_document_loader.py`

Acceptance criteria:
- `.txt` and `.md` are read as UTF-8 text
- `.pdf` is parsed via PyMuPDF
- unsupported extension raises clear error

Verification:
- Run `pytest tests/test_document_loader.py -v`

## Task 4: Implement standard chunker and retriever

Objective: Split standard text into clauses and retrieve relevant clauses using simple lexical matching first.

Files:
- Create: `backend/app/rag/retriever.py`
- Create: `tests/test_retriever.py`

Acceptance criteria:
- Chunks preserve clause identifiers where possible
- Query `crack length 5mm` retrieves crack-related clauses
- No-match case returns empty evidence and triggers human review later

Verification:
- Run `pytest tests/test_retriever.py -v`

## Task 5: Implement vision adapter stub

Objective: Create a testable interface for defect detection without binding to one model yet.

Files:
- Create: `backend/app/vision/defect_detector.py`
- Create: `tests/test_defect_detector.py`

MVP behavior:
- For now, return a deterministic `DefectObservation` based on filename or provided metadata
- Later replace with VLM/YOLO/SAM implementation

Verification:
- Run `pytest tests/test_defect_detector.py -v`

## Task 6: Implement severity and review logic

Objective: Convert observation + retrieved standard + sensor values into severity, action, and human-review flag.

Files:
- Create: `backend/app/agents/inspection_agent.py`
- Create: `tests/test_inspection_agent.py`

Rules:
- confidence < 0.70 -> requires_human_review = true
- no standard evidence -> requires_human_review = true
- high/critical severity -> requires_human_review = true
- critical sensor threshold -> stop_machine

Verification:
- Run `pytest tests/test_inspection_agent.py -v`

## Task 7: Implement report generator

Objective: Generate a structured report object and Markdown summary.

Files:
- Create: `backend/app/services/report_generator.py`
- Create: `tests/test_report_generator.py`

Verification:
- Run `pytest tests/test_report_generator.py -v`

## Task 8: Add `/api/inspect` endpoint

Objective: Expose the full backend workflow through FastAPI.

Files:
- Modify: `backend/app/main.py`
- Create: `tests/test_inspect_endpoint.py`

Request:
- multipart image file
- multipart standard document file
- optional sensor CSV

Response:
- `InspectionReport` JSON

Verification:
- Run `pytest tests/test_inspect_endpoint.py -v`

## Task 9: Add sample standard and sample report

Objective: Make the demo understandable without external data.

Files:
- Create: `data/sample_standards/pipeline_surface_defect_standard.md`
- Create: `data/sample_reports/example_report.json`

Verification:
- User can run the endpoint with sample files and see a realistic report.

## Task 10: Add minimal frontend

Objective: Provide a simple upload + report display UI.

Files:
- Create: `frontend/app/page.tsx`
- Create: `frontend/components/ImageUploader.tsx`
- Create: `frontend/components/InspectionReport.tsx`
- Create: `frontend/components/DefectViewer.tsx`

Verification:
- Upload image/document
- See report fields rendered clearly

## Definition of Done

- `pytest tests/ -q` passes
- README explains MVP and commands
- `/api/inspect` returns structured report
- Sample standard and sample report exist
- Human-review logic is explicit and test-covered
