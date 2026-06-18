from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.inspect import router as inspect_router

app = FastAPI(
    title="Multimodal Industrial Inspection Agent",
    description="MVP API for industrial defect inspection workflows.",
    version="0.1.0",
)

# Permissive CORS for the local MVP frontend. Tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inspect_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
