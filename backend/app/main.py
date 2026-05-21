from fastapi import FastAPI

app = FastAPI(
    title="Multimodal Industrial Inspection Agent",
    description="MVP API for industrial defect inspection workflows.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
