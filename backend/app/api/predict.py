"""Prediction endpoint."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.main import AppState
from app.utils import safe_filename

logger = logging.getLogger("plantpath.api.predict")

router = APIRouter()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@router.post("/predict")
async def predict(image: UploadFile = File(...)):
    settings = AppState.settings
    ps = AppState.prediction_service
    db = AppState.database
    classifier = AppState.classifier

    if ps is None or classifier is None:
        raise HTTPException(status_code=503, detail="Prediction service is not ready.")

    filename = safe_filename(image.filename or "image.jpg")
    ext = Path(filename).suffix.lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Use JPG, JPEG, PNG, or WEBP.",
        )

    try:
        data = await image.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.")

    if not data:
        raise HTTPException(status_code=400, detail="Empty upload. Please select an image.")

    if not settings.demo_mode and not classifier.is_loaded():
        raise HTTPException(
            status_code=503,
            detail=(
                "No verified model is connected, so predictions cannot be generated. "
                f"Reason: {classifier.get_error()}"
            ),
        )

    try:
        result = ps.predict(data, filename, include_explain=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        logger.warning("Prediction unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("Unexpected prediction error")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an unexpected error.",
        )

    if db is not None and result.get("top_prediction"):
        try:
            db.insert_prediction(
                filename=filename,
                crop=result.get("crop", "") or "",
                disease=result.get("disease", "") or "",
                confidence=float(result["top_prediction"].get("probability", 0.0)),
                model=result.get("model", {}).get("architecture", "PlantPath AI"),
                mode="demo" if result.get("demo_mode") else "real",
                condition=result.get("condition"),
                top_predictions=result.get("top_predictions"),
            )
        except Exception:
            logger.warning("Failed to write prediction history", exc_info=True)

    return result