import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from backend.api.routes import router
from backend.config import BASE_DIR

app = FastAPI(
    title="Quran Reciter Recognition API",
    description="API for Quran reciter recognition using ECAPA-TDNN",
    version="1.0.0"
)

app.include_router(router)

frontend_dir = os.path.join(BASE_DIR, "frontend")

app.mount(
    "/",
    StaticFiles(
        directory=frontend_dir,
        html=True
    ),
    name="frontend"
)

@app.get("/health")
def health():
    return {
        "status": "ok"
    }