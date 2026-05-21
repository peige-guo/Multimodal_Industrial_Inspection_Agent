# Evaluation Plan

## MVP Evaluation

Evaluate the system at workflow level, not only model accuracy.

## Metrics

### Vision
- defect type accuracy
- localization IoU if bounding boxes/masks are available
- false positive rate
- false negative rate

### RAG
- standard clause recall@k
- citation correctness
- unsupported-claim rate

### Decision
- severity classification accuracy
- recommended action accuracy
- human-review trigger precision/recall

### Report
- completeness of required fields
- evidence traceability
- human readability

## Test Data

Start with:
- MVTec AD
- NEU surface defect dataset
- KolektorSDD
- synthetic standard documents written for each defect class

## Feedback Loop

Store reviewer feedback:

- correct / incorrect
- wrong defect type
- wrong severity
- wrong standard evidence
- wrong recommendation

Use feedback to improve prompts, thresholds, retrieval, and eventually fine-tuning.
