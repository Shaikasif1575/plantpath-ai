"""Model metadata endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from app.main import AppState

router = APIRouter()


@router.get("/model-info")
async def model_info():
    ps = AppState.prediction_service
    settings = AppState.settings
    classifier = AppState.classifier
    if ps is None or classifier is None:
        return {
            "name": "PlantPath AI",
            "version": "1.0.0",
            "loaded": False,
            "demo_mode": bool(settings.demo_mode) if settings else False,
            "classes": [],
            "error": "service not ready",
        }
    meta = ps.get_model_metadata()
    meta["low_confidence_threshold"] = settings.confidence_low
    meta["segmentation_available"] = ps.severity.is_available()
    meta["segmentation_status"] = "available" if ps.severity.is_available() else "unavailable"
    meta["gradcam_available"] = ps.gradcam.is_available()
    meta.setdefault("severity", ps.severity.get_method_info())
    return meta


@router.get("/crops")
async def crops():
    """Crops supported by the connected model, plus the reference library.

    ``items`` reflects the real model class set and is empty when no verified
    model with a resolved class mapping is connected. ``reference_library`` is
    educational content only and is never presented as model-supported.
    """
    classifier = AppState.classifier
    ds = AppState.disease_service
    class_names = classifier.get_classes() if classifier else []
    supported = ds.get_crops(class_names) if ds else []
    reference = ds.get_crops(ds.all_class_names()) if ds else []
    return {
        "items": supported,
        "total": len(supported),
        "supported_classes": class_names,
        "reference_library": {
            "note": (
                "Educational reference content only. These are not the classes the "
                "connected model supports."
            ),
            "items": reference,
            "total": len(reference),
        },
    }