# Agent Workflow

## Core Flow

```text
Input package
  - image
  - standard document
  - optional sensor CSV
        |
        v
1. Validate inputs
        |
        v
2. Vision analysis
        - defect type
        - approximate location
        - confidence
        - visual explanation
        |
        v
3. Standard RAG
        - parse document
        - chunk clauses
        - retrieve relevant rules
        |
        v
4. Sensor check
        - threshold abnormality
        - contradiction with visual evidence
        |
        v
5. Decision engine
        - severity level
        - recommended action
        - human review flag
        |
        v
6. Report generation
```

## Decision Output Contract

The agent must always output:

- What defect was found
- Where it appears
- Why it matters
- Which standard clause supports the judgment
- How severe it is
- What action is recommended
- Whether a human must review it

## Human Review Triggers

- Low model confidence
- Missing or weak standard evidence
- Conflicting visual/sensor evidence
- High business/safety risk
- Critical severity
