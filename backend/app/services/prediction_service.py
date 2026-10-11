"""Prediction orchestration service.

Live mode runs real inference with the connected trained classifier and returns
its actual output. It never substitutes a simulated disease, confidence or
severity value. If no verified model is connected, live prediction fails with a
clear error instead of inventing a result.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import Settings, load_json
from app.ml.gradcam_service import GradCamService
from app.ml.severity_service import SeverityService
from app.ml.torch_classifier import parse_class_name
from app.schemas import (
    GradCamResult,
    PredictionItem,
    SeverityResult,
)

logger = logging.getLogger("plantpath.services.prediction")

LOW_CONFIDENCE_MESSAGE = (
    "Low-confidence prediction — please upload a clearer image or an image from a "
    "supported plant/disease category."
)

MODEL_UNAVAILABLE_MESSAGE = (
    "No verified model is connected, so predictions cannot be generated. "
    "No result is shown rather than an invented one."
)


class PredictionService:
    def __init__(self, settings: Settings, classifier) -> None:
        self.settings = settings
        self.classifier = classifier
        self.gradcam = GradCamService(classifier)
        seg_path = settings.resolved_segmentation_model_path
        self.severity = SeverityService(seg_path if seg_path.exists() else None)
        self._metrics = load_json(settings.resolved_metrics_path) or {}

    def is_demo(self) -> bool:
        return bool(self.settings.demo_mode)

    def _classify(
        self, probs: List[float], class_names: List[str]
    ) -> Tuple[PredictionItem, List[PredictionItem]]:
        indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        top_items: List[PredictionItem] = []
        for rank, (idx, p) in enumerate(indexed[:3], start=1):
            cname = class_names[idx] if idx < len(class_names) else f"class_{idx}"
            crop, disease, _condition = parse_class_name(cname)
            top_items.append(
                PredictionItem(
                    class_name=cname,
                    crop=crop,
                    disease=disease,
                    probability=round(float(p), 4),
                    rank=rank,
                )
            )
        return top_items[0], top_items

    def _confidence_band(self, confidence: float) -> Tuple[str, str]:
        low = self.settings.confidence_low
        med = self.settings.confidence_medium
        if confidence >= med:
            band = "high"
            note = (
                "The model assigned relatively high probability to the top class. "
                "This is still not a guarantee of diagnostic correctness."
            )
        elif confidence >= low:
            band = "medium"
            note = (
                "The model is moderately confident. Consider capturing another image "
                "with the leaf clearly visible and well-lit."
            )
        else:
            band = "low"
            note = LOW_CONFIDENCE_MESSAGE
        return band, note

    def predict(
        self,
        image_bytes: bytes,
        filename: str,
        include_explain: bool = True,
    ) -> Dict[str, Any]:
        from app.services.image_validation import validate_image, load_image_rgb

        max_bytes = self.settings.max_upload_bytes
        img, validation = validate_image(image_bytes, filename, max_bytes)
        if img is None:
            raise ValueError(validation.get("error") or "Invalid image.")

        arr = load_image_rgb(img)
        demo = self.is_demo()

        if demo:
            return self._demo_result(arr, validation)
        if not self.classifier.is_loaded():
            raise RuntimeError(
                f"{MODEL_UNAVAILABLE_MESSAGE} Reason: {self.classifier.get_error()}"
            )

        result = self.classifier.predict(arr)
        class_names = self.classifier.get_classes()
        probs = [t["confidence"] for t in result["top_predictions"]]
        top_pred, top_preds = self._classify(probs, class_names)
        confidence = top_pred.probability
        band, note = self._confidence_band(confidence)
        crop, disease, condition = parse_class_name(top_pred.class_name)
        low_confidence = confidence < self.settings.confidence_low

        if include_explain and self.gradcam.is_available():
            explain = GradCamResult(**self.gradcam.generate(
                arr, result["class_index"], top_pred.class_name
            ))
        else:
            explain = GradCamResult(
                available=False,
                message="Explainability unavailable for this model.",
            )

        severity = SeverityResult(**self.severity.analyze(arr))

        model_info = self.classifier.get_model_info()
        model_info.update({"demo_mode": False, "status": "live"})

        limitations = [
            "This model is limited to the classes represented by the connected classifier.",
            "PlantVillage is a controlled benchmark dataset; field performance may differ.",
            "Model confidence does not guarantee diagnostic correctness.",
            "Images of plants or diseases outside the trained class set may be misclassified.",
            "Severity assessment is unavailable: no compatible segmentation model is connected.",
        ]

        return {
            "status": "ok",
            "demo_mode": False,
            "live_model": True,
            "leaf_validation_available": False,
            "validation": validation,
            "crop": crop,
            "disease": disease,
            "condition": condition,
            "detected_disease": None if condition == "healthy" else disease,
            "top_prediction": top_pred.model_dump(),
            "top_predictions": [p.model_dump() for p in top_preds],
            "confidence_band": band,
            "confidence_note": note,
            "low_confidence": low_confidence,
            "low_confidence_message": LOW_CONFIDENCE_MESSAGE if low_confidence else None,
            "supported_classes": list(class_names),
            "model": model_info,
            "explainability": explain.model_dump(),
            "severity": severity.model_dump(),
            "limitations": limitations,
        }

    def _demo_result(self, arr: np.ndarray, validation: Dict[str, Any]) -> Dict[str, Any]:
        """Explicit demo mode only (DEMO_MODE=true). Clearly labelled as not real."""
        explain = GradCamResult(
            available=False,
            message="Explainability is unavailable in demo mode (no trained model connected).",
        )
        severity = SeverityResult(
            available=False,
            message="Severity assessment is currently unavailable for this model.",
        )
        return {
            "status": "ok",
            "demo_mode": True,
            "live_model": False,
            "leaf_validation_available": False,
            "validation": validation,
            "crop": None,
            "disease": None,
            "condition": None,
            "detected_disease": None,
            "top_prediction": None,
            "top_predictions": [],
            "confidence_band": None,
            "confidence_note": None,
            "low_confidence": True,
            "low_confidence_message": LOW_CONFIDENCE_MESSAGE,
            "supported_classes": [],
            "model": {
                "loaded": False,
                "demo_mode": True,
                "status": "demo",
                "architecture": "Demo mode: no trained model connected",
                "framework": "None",
                "classes": [],
            },
            "explainability": explain.model_dump(),
            "severity": severity.model_dump(),
            "limitations": [
                "DEMO MODE is active: no real inference is performed and no prediction is produced.",
            ],
        }

    def get_model_metadata(self) -> Dict[str, Any]:
        info = self.classifier.get_model_info()
        metrics = self._metrics.get("metrics", {}) if isinstance(self._metrics, dict) else {}
        training = self._metrics.get("training", {}) if isinstance(self._metrics, dict) else {}
        if not metrics:
            metrics = {"accuracy": None, "precision": None, "recall": None, "f1": None}
        if not training:
            training = {
                "training_accuracy": None,
                "validation_accuracy": None,
                "training_loss": None,
                "validation_loss": None,
            }
        loaded = self.classifier.is_loaded()
        return {
            **info,
            "demo_mode": self.is_demo(),
            "gradcam_available": self.gradcam.is_available(),
            "severity": self.severity.get_method_info(),
            "metrics": metrics,
            "training": training,
            "limitations": [
                "PlantVillage is a controlled benchmark dataset.",
                "Performance on field-collected images is not established by dataset metrics alone.",
                "Unknown / out-of-distribution classes are not detected by this classifier.",
                "Severity requires a segmentation model; none is connected.",
            ],
            "class_names": self.classifier.get_classes(),
            "comparison": [],
        }