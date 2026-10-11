"""Grad-CAM explain endpoint."""
from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.main import AppState
from app.utils import safe_filename

logger = logging.getLogger("plantpath.api.explain")

router = APIRouter()

UNAVAILABLE = {
    "available": False,
    "message": "Explainability unavailable for this model.",
}


@router.post("/explain")
async def explain(
    image: UploadFile = File(...),
    target_class: str = Form(None),
):
    """Generate Grad-CAM for an uploaded image using the connected model."""
    ps = AppState.prediction_service
    classifier = AppState.classifier
    if ps is None or classifier is None:
        raise HTTPException(status_code=503, detail="Services are not ready.")

    if not classifier.is_loaded():
        return {
            "available": False,
            "message": "Explainability unavailable for this model.",
        }

    filename = safe_filename(image.filename or "image.jpg")
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")

    from app.services.image_validation import load_image_rgb, validate_image

    img, validation = validate_image(
        data, filename, AppState.settings.max_upload_bytes
    )
    if img is None:
        raise HTTPException(status_code=400, detail=validation.get("error") or "Invalid image.")

    arr = load_image_rgb(img)
    class_names = classifier.get_classes()
    if not class_names:
        return UNAVAILABLE

    class_index = 0
    class_name = class_names[0]
    if target_class and target_class.strip():
        tc = target_class.strip()
        if tc.isdigit() and int(tc) < len(class_names):
            class_index = int(tc)
        elif tc in class_names:
            class_index = class_names.index(tc)
        else:
            matches = [i for i, c in enumerate(class_names) if tc.lower() in c.lower()]
            if matches:
                class_index = matches[0]
    else:
        try:
            result = classifier.predict(arr)
            class_index = result["class_index"]
            class_name = result["class_name"]
        except Exception:
            logger.warning("Could not resolve top class for Grad-CAM", exc_info=True)

    if class_index < len(class_names):
        class_name = class_names[class_index]

    return ps.gradcam.generate(arr, class_index, class_name)