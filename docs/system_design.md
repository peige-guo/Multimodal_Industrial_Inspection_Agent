# System Design: Multimodal Industrial Inspection Agent

## 1. Problem

The key challenge in industrial quality inspection is not simply identifying one defect. The real requirement is a complete workflow: detect an abnormality -> classify the defect type -> query the applicable standard -> explain the root cause or risk -> assign a risk level -> generate a report -> recommend whether human review is required.

## 2. MVP Input / Output

### Inputs

- Industrial defect image: jpg/png
- Inspection standard document: pdf/docx/md/txt, with md/txt/pdf prioritized for the MVP
- Optional sensor CSV: fields such as temperature, vibration, pressure, and current

### Output

Structured Inspection Report:

- inspection_id
- object_name
- inspection_time
- input_data_summary
- defect_type
- defect_location
- severity_level
- confidence
- standard_evidence
- risk_explanation
- recommended_action
- requires_human_review
- annotated_image_path

## 3. Architecture

```text
Frontend Upload Page
        |
        v
FastAPI /api/inspect
        |
        v
Inspection Agent / LangGraph workflow
        |
        +--> Vision Adapter: defect detection / VLM description
        |
        +--> Document Loader: parse standards
        |
        +--> Retriever: retrieve relevant clauses
        |
        +--> Severity Engine: rule + LLM judgment
        |
        +--> Report Generator: structured report
        v
Inspection Report JSON + UI rendering
```

## 4. Agent Workflow

1. Validate uploaded inputs
2. Extract text from the standard document
3. Chunk and index standard clauses
4. Analyze the image for candidate defect type, location, and confidence
5. Retrieve standard clauses relevant to the defect type and sensor abnormality
6. Combine visual evidence, standard evidence, and sensor values
7. Assign severity: low / medium / high / critical
8. Recommend action: release / re-inspect / repair / reject / stop-machine
9. Mark human review if confidence is low, evidence conflicts, or severity is high/critical
10. Generate the report

## 5. Severity Logic for MVP

Use a transparent hybrid strategy:

- Rule-based thresholds from retrieved standard text when explicit limits exist
- LLM/VLM reasoning only for explanation and ambiguous cases
- Human review when:
  - confidence < 0.70
  - no relevant standard clause found
  - severity is high or critical
  - sensor data contradicts visual result

## 6. Phase Roadmap

### Phase 1: MVP

- Single image inspection
- Standard document RAG
- JSON report
- Basic web UI

### Phase 2: Stronger Vision

- YOLOv8/YOLOv11 fine-tuning
- SAM segmentation mask
- Annotated defect image
- Evaluation on MVTec/NEU/KolektorSDD

### Phase 3: Feedback Loop

- Store user feedback
- Track model performance
- Improve prompts and rules
- Build inspection knowledge base

### Phase 4: Industrialization

- Video frame sampling
- Sensor time-series anomaly detection
- PostgreSQL
- Auth and audit logs
- Deployment with Docker
