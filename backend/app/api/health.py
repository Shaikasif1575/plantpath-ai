"""Health check endpoint."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from app.main import AppState
from app.schemas import HealthResponse

router = APIRouter()


@router.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = AppState.settings
    classifier = AppState.classifier
    demo = settings.demo_mode if settings else False
    loaded = classifier.is_loaded() if classifier else False
    info = classifier.get_model_info() if classifier else {}
    gradcam = False
    severity = False
    if AppState.prediction_service is not None:
        gradcam = AppState.prediction_service.gradcam.is_available()
        severity = AppState.prediction_service.severity.is_available()
    classes = classifier.get_classes() if classifier else []
    if demo:
        detail = "Demo mode: no real inference is performed."
    elif loaded:
        detail = "Live model loaded; real inference enabled."
    else:
        detail = (
            "No verified model is connected; predictions are unavailable rather than "
            f"simulated. Reason: {classifier.get_error() if classifier else 'unknown'}"
        )
    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        demo_mode=demo,
        model_name="PlantPath AI",
        model_version="1.0.0",
        gradcam_available=gradcam,
        severity_available=severity,
        leaf_validation_available=False,
        class_count=len(classes) if classes else None,
        timestamp=datetime.now(timezone.utc).isoformat(),
        detail=detail,
    )