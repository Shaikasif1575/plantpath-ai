"""FastAPI application entry point for PlantPath AI."""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure backend package is importable when run from repo root or backend/
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# Project root (parent of backend/)
_PROJECT_ROOT = _BACKEND_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.database import Database  # noqa: E402
from app.ml.torch_classifier import TorchClassifier  # noqa: E402
from app.services.disease_service import DiseaseService  # noqa: E402
from app.services.prediction_service import PredictionService  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("plantpath")


class AppState:
    settings = None
    classifier: TorchClassifier | None = None
    prediction_service: PredictionService | None = None
    disease_service: DiseaseService | None = None
    database: Database | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    AppState.settings = settings

    # Resolve paths relative to project root for convenience
    def _resolve(p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        return (_PROJECT_ROOT / path).resolve()

    model_path = _resolve(settings.model_path)
    classes_path = _resolve(settings.model_classes_path)
    audit_path = _resolve(settings.model_audit_path)
    metrics_path = _resolve(settings.model_metrics_path)
    diseases_path = _resolve(settings.diseases_path)
    db_path = _resolve(settings.db_path)

    # Override settings paths with resolved absolute paths
    settings.model_path = str(model_path)
    settings.model_classes_path = str(classes_path)
    settings.model_audit_path = str(audit_path)
    settings.model_metrics_path = str(metrics_path)
    settings.diseases_path = str(diseases_path)
    settings.db_path = str(db_path)

    classifier = TorchClassifier()
    classifier.configure(
        model_path=model_path,
        classes_path=classes_path,
        preprocessing=settings.preprocessing,
        audit_path=audit_path,
    )

    AppState.classifier = classifier
    AppState.prediction_service = PredictionService(settings, classifier)
    AppState.disease_service = DiseaseService(diseases_path)
    AppState.database = Database(db_path)

    loaded = classifier.is_loaded()
    logger.info(
        "PlantPath AI backend started | demo_mode=%s model_loaded=%s model_path=%s",
        settings.demo_mode,
        loaded,
        model_path,
    )
    if not loaded:
        logger.error(
            "Model not available for live inference: %s", classifier.get_error()
        )
    yield
    logger.info("PlantPath AI backend shutting down.")


app = FastAPI(
    title="PlantPath AI",
    description=(
        "Explainable deep learning for plant disease detection, classification, "
        "and (future) severity assessment. Research and educational prototype."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected server error occurred. Please try again.",
            "error_type": type(exc).__name__,
        },
    )


def _cors_origins():
    settings = get_settings()
    origins = settings.cors_list
    if not origins:
        origins = ["*"]
    return origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register routes
from app.api import health, model, diseases, predict, explain, history  # noqa: E402

app.include_router(health.router, tags=["health"])
app.include_router(model.router, prefix="/api", tags=["model"])
app.include_router(diseases.router, prefix="/api", tags=["diseases"])
app.include_router(predict.router, prefix="/api", tags=["predict"])
app.include_router(explain.router, prefix="/api", tags=["explain"])
app.include_router(history.router, prefix="/api", tags=["history"])


@app.get("/")
async def root():
    return {
        "name": "PlantPath AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
